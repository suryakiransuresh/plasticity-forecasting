#!/usr/bin/env python3
"""Read-only baseline knowledge audit for the 100-edit RippleEdits dataset.

Scores the *full canonical answer token sequence* for acquisition, protected,
and ripple probes.  It does not define a known/unknown cutoff and never
updates, optimizes, or saves the model.

Run from the repository root:
  python experiments/02_external_validation/02_qwen_baseline_knowledge_audit.py
"""

import argparse
import csv
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
CACHE_DIR = ROOT / "results" / "cache" / "huggingface"
DEFAULT_DATASET = ROOT / "data" / "external_validation" / "rippleedits_qwen_100.json"
DEFAULT_ROWS = ROOT / "results" / "external_validation" / "rippleedits_qwen_baseline_knowledge_audit.csv"
DEFAULT_SUMMARY = ROOT / "results" / "external_validation" / "rippleedits_qwen_baseline_knowledge_audit_summary.json"
MIN_GROUP_SIZE = 5


def choose_device(requested):
    if requested != "auto":
        return torch.device(requested)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def make_rows(dataset):
    """Flatten one scoreable canonical-answer row per benchmark probe."""
    rows = []
    for edit in dataset["edits"]:
        common = {
            "edit_id": edit["edit_id"],
            "source_subset": edit["source"]["subset"],
            "example_type": edit["source"]["example_type"],
            "edit_relation": edit["metadata"]["relation"],
        }
        acquisition = edit["acquisition"]
        rows.append({
            **common, "role": "acquisition", "axis": "acquisition", "probe_id": "acquisition",
            "prompt": acquisition["prompt"], "canonical_answer": acquisition["canonical_answer"],
            "score_answer": acquisition["score_answer"],
            "dataset_prompt_tokens": acquisition["prompt_tokens"],
            "dataset_answer_tokens": acquisition["answer_tokens"],
        })
        for role_key, role_name in (("protected", "protected"), ("ripples", "ripple")):
            for axis, probes in edit[role_key].items():
                for probe in probes:
                    rows.append({
                        **common, "role": role_name, "axis": axis,
                        "probe_id": f"{axis}:{probe['case_index']}:{probe['query_index']}",
                        "prompt": probe["prompt"], "canonical_answer": probe["answer"]["canonical"],
                        "score_answer": probe["score_answer"],
                        "dataset_prompt_tokens": probe["prompt_tokens"],
                        "dataset_answer_tokens": probe["answer_tokens"],
                    })
    return rows


def prepare_tokens(rows, tokenizer):
    """Validate the answer boundary again and retain exact loss spans."""
    for row in rows:
        prompt_ids = tokenizer(row["prompt"], add_special_tokens=False).input_ids
        full_ids = tokenizer(row["prompt"] + row["score_answer"], add_special_tokens=False).input_ids
        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError(f"prompt is not a token prefix for {row['edit_id']} {row['probe_id']}")
        answer_tokens = len(full_ids) - len(prompt_ids)
        if answer_tokens < 1:
            raise ValueError(f"empty answer span for {row['edit_id']} {row['probe_id']}")
        if (len(prompt_ids), answer_tokens) != (row["dataset_prompt_tokens"], row["dataset_answer_tokens"]):
            raise ValueError(f"tokenization differs from constructed dataset for {row['edit_id']} {row['probe_id']}")
        row["input_ids"] = full_ids
        row["answer_start"] = len(prompt_ids)
        row["prompt_tokens"] = len(prompt_ids)
        row["answer_tokens"] = answer_tokens


def score_batch(model, batch, pad_token_id, device):
    max_length = max(len(row["input_ids"]) for row in batch)
    input_ids = torch.full((len(batch), max_length), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(batch), max_length), dtype=torch.long, device=device)
    for i, row in enumerate(batch):
        ids = torch.tensor(row["input_ids"], dtype=torch.long, device=device)
        input_ids[i, :len(ids)] = ids
        attention_mask[i, :len(ids)] = 1
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        log_probs = torch.log_softmax(logits.float(), dim=-1)
    for i, row in enumerate(batch):
        start, end = row["answer_start"], len(row["input_ids"])
        # Causal logits at token t-1 score target token t.  This span includes
        # every canonical answer token, rather than only its first token.
        positions = torch.arange(start - 1, end - 1, device=device)
        targets = input_ids[i, start:end]
        nll = -log_probs[i, positions, targets]
        row["canonical_answer_nll"] = float(nll.sum().cpu())
        row["canonical_answer_nll_per_token"] = float(nll.mean().cpu())


def percentile(sorted_values, percent):
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * percent / 100.0
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def describe(rows):
    values = sorted(row["canonical_answer_nll_per_token"] for row in rows)
    mean = sum(values) / len(values)
    return {
        "n": len(rows),
        "answer_tokens_total": sum(row["answer_tokens"] for row in rows),
        "mean_nll_per_token": mean,
        "median_nll_per_token": percentile(values, 50),
        "std_nll_per_token": math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)),
        "min_nll_per_token": values[0],
        "max_nll_per_token": values[-1],
        "percentiles_nll_per_token": {str(p): percentile(values, p) for p in (10, 25, 50, 75, 90)},
    }


def grouped_descriptions(rows, key, min_group_size=MIN_GROUP_SIZE):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return {
        str(name): describe(group)
        for name, group in sorted(groups.items()) if len(group) >= min_group_size
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--rows-output", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-group-size", type=int, default=MIN_GROUP_SIZE)
    args = parser.parse_args()
    if args.batch_size < 1 or args.min_group_size < 1:
        parser.error("--batch-size and --min-group-size must be positive")

    dataset = json.loads(args.dataset.read_text())
    rows = make_rows(dataset)
    expected = {"acquisition": 100, "protected": 505, "ripple": 773}
    observed = {role: sum(row["role"] == role for row in rows) for role in expected}
    if observed != expected:
        raise ValueError(f"unexpected dataset role counts: {observed}; expected {expected}")
    device = choose_device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS was requested but is unavailable")

    print(f"Preparing {len(rows)} canonical-answer scores on {device} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=str(args.cache_dir), local_files_only=True)
    prepare_tokens(rows, tokenizer)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer has neither pad nor EOS token")
    print(f"Loading {args.model} in eval mode (no updates) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir=str(args.cache_dir), local_files_only=True).to(device).eval()
    start = time.perf_counter()
    for offset in range(0, len(rows), args.batch_size):
        score_batch(model, rows[offset:offset + args.batch_size], pad_token_id, device)
        completed = min(offset + args.batch_size, len(rows))
        if completed % 100 == 0 or completed == len(rows):
            print(f"Scored {completed}/{len(rows)}", flush=True)

    for row in rows:
        del row["input_ids"]
        del row["answer_start"]
    args.rows_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "edit_id", "source_subset", "example_type", "edit_relation", "role", "axis", "probe_id",
        "prompt", "canonical_answer", "score_answer", "prompt_tokens", "answer_tokens",
        "canonical_answer_nll", "canonical_answer_nll_per_token",
    ]
    temporary_rows_output = args.rows_output.with_suffix(args.rows_output.suffix + ".tmp")
    written_rows = 0
    with temporary_rows_output.open("w", newline="", encoding="utf-8") as handle:
        # ``dataset_*_tokens`` are construction-time consistency checks.  Keep
        # them in memory for validation but do not expose duplicate columns in
        # the row-level audit file.
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            written_rows += 1
        handle.flush()
        os.fsync(handle.fileno())
    if written_rows != len(rows):
        raise RuntimeError(f"wrote {written_rows} CSV rows; expected {len(rows)}")
    # Replace only a complete, count-checked file; this prevents a failed run
    # from leaving a misleading header-only audit CSV behind.
    temporary_rows_output.replace(args.rows_output)
    with args.rows_output.open(newline="", encoding="utf-8") as handle:
        verified_rows = sum(1 for _ in csv.DictReader(handle))
    if verified_rows != len(rows):
        raise RuntimeError(f"CSV verification found {verified_rows} data rows; expected {len(rows)}")
    summary = {
        "purpose": "Baseline knowledge characterization only; no known/unknown threshold and no model update.",
        "dataset": str(args.dataset), "rows": str(args.rows_output), "model": args.model,
        "device": str(device), "batch_size": args.batch_size, "runtime_seconds": round(time.perf_counter() - start, 3),
        "scoring": "sum and mean negative log likelihood over every canonical answer token; lower is better",
        "counts_by_role": observed,
        "overall": describe(rows),
        "by_role": grouped_descriptions(rows, "role", 1),
        "by_protected_axis": grouped_descriptions([row for row in rows if row["role"] == "protected"], "axis", 1),
        "by_source_subset": grouped_descriptions(rows, "source_subset", args.min_group_size),
        "by_relation": grouped_descriptions(rows, "edit_relation", args.min_group_size),
        "grouping_rule": f"source/relation groups are reported when n >= {args.min_group_size}",
    }
    args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {args.rows_output}", flush=True)
    print(f"Wrote {args.summary_output}", flush=True)


if __name__ == "__main__":
    main()
