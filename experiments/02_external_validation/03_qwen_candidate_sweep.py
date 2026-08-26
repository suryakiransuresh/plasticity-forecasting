#!/usr/bin/env python3
"""Four-candidate, non-forecasting RippleEdits sweep for Qwen2.5-0.5B.

Updates are applied only to disposable model copies.  The frozen validation
dataset, baseline audit, and model checkpoint are never modified.
"""

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
CACHE_DIR = ROOT / "results" / "cache" / "huggingface"
DATASET = ROOT / "data" / "external_validation" / "rippleedits_qwen_100.json"
BASELINE = ROOT / "results" / "external_validation" / "rippleedits_qwen_baseline_knowledge_audit.csv"
ROWS_OUTPUT = ROOT / "results" / "external_validation" / "rippleedits_qwen_candidate_sweep.csv"
SUMMARY_OUTPUT = ROOT / "results" / "external_validation" / "rippleedits_qwen_candidate_sweep_summary.json"
UPDATE_STEPS = 1
KNOWN_NLL_PER_TOKEN = 4.0
CANDIDATES = (
    {"name": "full_lr_1e-6", "mode": "full", "lr": 1e-6},
    {"name": "full_lr_1e-5", "mode": "full", "lr": 1e-5},
    {"name": "early_lr_1e-5", "mode": "early", "lr": 1e-5, "layers": list(range(0, 8))},
    {"name": "late_lr_1e-5", "mode": "late", "lr": 1e-5, "layers": list(range(16, 24))},
)


def choose_device(requested):
    if requested != "auto":
        return torch.device(requested)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def edit_seed(edit_id):
    """Stable per-edit seed; deliberately independent of candidate name."""
    return int(hashlib.sha256(edit_id.encode()).hexdigest()[:8], 16)


def reset_seed(seed):
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def targets_for_edit(edit):
    common = {"edit_id": edit["edit_id"], "source_subset": edit["source"]["subset"],
              "relation": edit["metadata"]["relation"]}
    acquisition = edit["acquisition"]
    targets = [{**common, "role": "acquisition", "axis": "acquisition", "probe_id": "acquisition",
                "prompt": acquisition["prompt"], "answer": acquisition["score_answer"]}]
    for container, role in (("protected", "protected"), ("ripples", "ripple")):
        for axis, probes in edit[container].items():
            for probe in probes:
                targets.append({**common, "role": role, "axis": axis,
                                "probe_id": f"{axis}:{probe['case_index']}:{probe['query_index']}",
                                "prompt": probe["prompt"], "answer": probe["score_answer"]})
    return targets


def baseline_map(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1378:
        raise ValueError(f"baseline CSV has {len(rows)} data rows; expected 1378")
    output = {}
    for row in rows:
        key = (row["edit_id"], row["role"], row["axis"], row["probe_id"])
        if key in output:
            raise ValueError(f"duplicate baseline row: {key}")
        output[key] = float(row["canonical_answer_nll_per_token"])
    return output


def prepare(targets, tokenizer):
    for target in targets:
        prompt_ids = tokenizer(target["prompt"], add_special_tokens=False).input_ids
        full_ids = tokenizer(target["prompt"] + target["answer"], add_special_tokens=False).input_ids
        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError(f"token boundary failed: {target['edit_id']} {target['probe_id']}")
        target["input_ids"], target["answer_start"] = full_ids, len(prompt_ids)
        target["answer_tokens"] = len(full_ids) - len(prompt_ids)
        if target["answer_tokens"] < 1:
            raise ValueError("empty answer token span")


def score(model, targets, pad_id, device, batch_size):
    model.eval()
    for offset in range(0, len(targets), batch_size):
        batch = targets[offset:offset + batch_size]
        max_length = max(len(item["input_ids"]) for item in batch)
        input_ids = torch.full((len(batch), max_length), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros_like(input_ids)
        for i, item in enumerate(batch):
            ids = torch.tensor(item["input_ids"], dtype=torch.long, device=device)
            input_ids[i, :len(ids)] = ids
            attention_mask[i, :len(ids)] = 1
        with torch.inference_mode():
            log_probs = torch.log_softmax(model(input_ids=input_ids, attention_mask=attention_mask).logits.float(), dim=-1)
        for i, item in enumerate(batch):
            start, end = item["answer_start"], len(item["input_ids"])
            positions = torch.arange(start - 1, end - 1, device=device)
            item["nll_after"] = float((-log_probs[i, positions, input_ids[i, start:end]]).mean().cpu())


def configure_trainable(model, candidate):
    for parameter in model.parameters():
        parameter.requires_grad = candidate["mode"] == "full"
    if candidate["mode"] == "full":
        return
    layers = getattr(model.model, "layers", None)
    if layers is None or len(layers) < 24:
        raise ValueError("expected Qwen decoder layers 0--23")
    for index in candidate["layers"]:
        for parameter in layers[index].parameters():
            parameter.requires_grad = True


def update(model, target, candidate, seed, device):
    reset_seed(seed)
    configure_trainable(model, candidate)
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    before = {id(parameter): parameter.detach().clone() for parameter in params}
    optimizer = torch.optim.AdamW(params, lr=candidate["lr"], weight_decay=0.0)
    model.train()
    for _ in range(UPDATE_STEPS):
        optimizer.zero_grad(set_to_none=True)
        ids = torch.tensor(target["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
        labels = ids.clone()
        labels[:, :target["answer_start"]] = -100
        model(input_ids=ids, labels=labels).loss.backward()
        optimizer.step()
    norm = math.sqrt(sum((parameter.detach() - before[id(parameter)]).float().pow(2).sum().item()
                         for parameter in params))
    del optimizer, before
    return norm, sum(parameter.numel() for parameter in params)


def mean(values):
    return sum(values) / len(values) if values else None


def make_result(edit, candidate, targets, baseline, update_norm, trainable_parameters):
    acquisition = next(target for target in targets if target["role"] == "acquisition")
    protected, ripple = [], []
    for target in targets:
        key = (target["edit_id"], target["role"], target["axis"], target["probe_id"])
        before = baseline[key]
        detail = {"axis": target["axis"], "probe_id": target["probe_id"], "baseline_nll_per_token": before,
                  "after_nll_per_token": target["nll_after"], "loss_change": target["nll_after"] - before,
                  "answer_tokens": target["answer_tokens"]}
        if target["role"] == "protected":
            detail["known_at_baseline"] = before <= KNOWN_NLL_PER_TOKEN
            protected.append(detail)
        elif target["role"] == "ripple":
            ripple.append(detail)
    known_damage = [item["loss_change"] for item in protected if item["known_at_baseline"]]
    all_damage = [item["loss_change"] for item in protected]
    return {
        "edit_id": edit["edit_id"], "candidate": candidate["name"], "candidate_mode": candidate["mode"],
        "learning_rate": candidate["lr"], "update_steps": UPDATE_STEPS, "seed": edit_seed(edit["edit_id"]),
        "source_subset": edit["source"]["subset"], "example_type": edit["source"]["example_type"],
        "relation": edit["metadata"]["relation"], "trainable_parameters": trainable_parameters, "update_norm": update_norm,
        "acquisition_loss_before": baseline[(edit["edit_id"], "acquisition", "acquisition", "acquisition")],
        "acquisition_loss_after": acquisition["nll_after"],
        "acquisition_gain": baseline[(edit["edit_id"], "acquisition", "acquisition", "acquisition")] - acquisition["nll_after"],
        "known_protected_probes": len(known_damage), "all_protected_probes": len(protected),
        "D_known": mean(known_damage), "D_all": mean(all_damage),
        "ripple_probes": len(ripple),
        "ripple_loss_before": mean([item["baseline_nll_per_token"] for item in ripple]),
        "ripple_loss_after": mean([item["after_nll_per_token"] for item in ripple]),
        "ripple_loss_change": mean([item["loss_change"] for item in ripple]),
        "protected_probe_losses": json.dumps(protected, ensure_ascii=False),
        "ripple_probe_losses": json.dumps(ripple, ensure_ascii=False),
    }


def summary(rows):
    metrics = ("acquisition_gain", "D_known", "D_all", "ripple_loss_before", "ripple_loss_after",
               "ripple_loss_change", "update_norm", "known_protected_probes")
    def describe(group):
        result = {"n": len(group)}
        for metric in metrics:
            values = [float(row[metric]) for row in group if row[metric] not in ("", None)]
            result[f"mean_{metric}"] = mean(values)
        return result
    def by(keys):
        groups = defaultdict(list)
        for row in rows:
            groups[tuple(row[key] for key in keys)].append(row)
        return [{**dict(zip(keys, name)), **describe(group)} for name, group in sorted(groups.items())]
    expected = 100 * len(CANDIDATES)
    per_edit = defaultdict(list)
    for row in rows:
        per_edit[row["edit_id"]].append(row)
    valid_candidates = all(len(group) == 4 and {r["candidate"] for r in group} == {c["name"] for c in CANDIDATES}
                           for group in per_edit.values()) and len(per_edit) == 100
    shared_seeds = all(len({row["seed"] for row in group}) == 1 for group in per_edit.values())
    return {
        "purpose": "Candidate sweep only; no forecasting was performed.", "model": MODEL_NAME,
        "update_steps": UPDATE_STEPS, "known_protected_nll_per_token_cutoff": KNOWN_NLL_PER_TOKEN,
        "ripple_is_excluded_from_damage": True, "expected_rows": expected, "completed_rows": len(rows),
        "complete": len(rows) == expected, "validation": {"four_candidates_per_edit": valid_candidates,
                                                               "same_seed_per_edit": shared_seeds},
        "candidate_level": by(("candidate",)), "source_level": by(("candidate", "source_subset")),
        "relation_level": by(("candidate", "relation")),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--rows-output", type=Path, default=ROWS_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=SUMMARY_OUTPUT)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    device = choose_device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS was requested but is unavailable")
    dataset = json.loads(args.dataset.read_text())
    baseline = baseline_map(args.baseline)
    all_targets = [target for edit in dataset["edits"] for target in targets_for_edit(edit)]
    if len(all_targets) != 1378 or any((t["edit_id"], t["role"], t["axis"], t["probe_id"]) not in baseline for t in all_targets):
        raise ValueError("frozen dataset and baseline audit do not align")
    args.rows_output.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if args.rows_output.exists() and not args.no_resume:
        with args.rows_output.open(newline="", encoding="utf-8") as handle: existing = list(csv.DictReader(handle))
    elif args.rows_output.exists():
        args.rows_output.unlink()
    completed = {(row["edit_id"], row["candidate"]) for row in existing}
    if len(completed) != len(existing): raise ValueError("duplicate completed edit/candidate rows")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=str(CACHE_DIR), local_files_only=True)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=str(CACHE_DIR), local_files_only=True).to(device).eval()
    fieldnames = None
    print(f"Device: {device}; completed: {len(existing)}/400; no forecasting", flush=True)
    with args.rows_output.open("a", newline="", encoding="utf-8") as handle:
        for edit in dataset["edits"]:
            prepared = targets_for_edit(edit); prepare(prepared, tokenizer)
            for candidate in CANDIDATES:
                if (edit["edit_id"], candidate["name"]) in completed: continue
                model = copy.deepcopy(base_model)
                norm, trainable = update(model, prepared[0], candidate, edit_seed(edit["edit_id"]), device)
                score(model, prepared, pad_id, device, args.batch_size)
                row = make_result(edit, candidate, prepared, baseline, norm, trainable)
                if fieldnames is None:
                    fieldnames = list(row)
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    if handle.tell() == 0: writer.writeheader()
                writer.writerow(row); handle.flush()
                existing.append(row); completed.add((edit["edit_id"], candidate["name"]))
                args.summary_output.write_text(json.dumps(summary(existing), indent=2) + "\n")
                print(f"{len(existing)}/400 {edit['edit_id']} {candidate['name']} G={row['acquisition_gain']:.4f} D_known={row['D_known']}", flush=True)
                del model; gc.collect()
                if device.type == "mps": torch.mps.empty_cache()
    report = summary(existing)
    args.summary_output.write_text(json.dumps(report, indent=2) + "\n")
    if not (report["complete"] and report["validation"]["four_candidates_per_edit"] and report["validation"]["same_seed_per_edit"]):
        raise RuntimeError("candidate-sweep completion validation failed")
    print(f"Complete: 400 rows written to {args.rows_output}")


if __name__ == "__main__":
    main()
