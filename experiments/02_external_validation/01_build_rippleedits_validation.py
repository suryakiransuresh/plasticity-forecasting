#!/usr/bin/env python3
"""Build a non-editing, 100-edit RippleEdits validation set for Qwen2.5-0.5B.

The official axes have deliberately different roles here:
  * acquisition gain (G): the counterfactual/recent ``edit`` fact;
  * protected damage (D): Relation_Specificity and Forgetfulness -- unrelated
    facts and the benchmark's previous-storage check that must *not* change;
  * ripple outcomes: Logical_Generalization, Subject_Aliasing,
    Compositionality_I and Compositionality_II.  These are
    expected consequences of the edit and are never folded into D.

Run from the repository root (this script only reads data and writes datasets):
  python experiments/02_external_validation/01_build_rippleedits_validation.py
"""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "data" / "external_validation" / "RippleEdits" / "data" / "benchmark"
OUT_DIR = ROOT / "data" / "external_validation"
DEFAULT_JSON = OUT_DIR / "rippleedits_qwen_100.json"
DEFAULT_CSV = OUT_DIR / "rippleedits_qwen_100.csv"
DEFAULT_SUMMARY = ROOT / "results" / "external_validation" / "rippleedits_qwen_100_summary.json"
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
CACHE_DIR = ROOT / "results" / "cache" / "huggingface"

# The independent set is intentionally the only source of D.  Every other
# official axis is an edit consequence, not collateral damage.
PROTECTED_AXES = ("Relation_Specificity", "Forgetfulness")
RIPPLE_AXES = (
    "Logical_Generalization",
    "Subject_Aliasing",
    "Compositionality_I",
    "Compositionality_II",
)
QUOTAS = {"random": 34, "recent": 33, "popular": 33}


def stable_order(items, seed, key):
    return sorted(items, key=lambda item: hashlib.sha256(f"{seed}|{key(item)}".encode()).hexdigest())


def split_edit_prompt(text):
    """Return the exact prompt prefix and a space-prefixed full answer.

    The explicit leading answer space makes the token boundary auditable and
    permits answer-only loss over every answer token (not just token one).
    """
    text = text.strip()
    separator = next((value for value in (" is ", " follows ") if value in text), None)
    if separator is None:
        raise ValueError(f"unrecognised edit form: {text!r}")
    prefix, answer = text.rsplit(separator, 1)
    answer = answer.rstrip(".?!").strip()
    if not prefix.strip() or not answer:
        raise ValueError(f"empty edit prompt/answer: {text!r}")
    # Empty entity labels appear in a small fraction of RECENT records and
    # cannot yield a meaningful language-model prompt.
    if "  " in prefix:
        raise ValueError(f"malformed subject label: {text!r}")
    return prefix + separator.rstrip(), " " + answer


def answer_payload(query):
    answers = query.get("answers", [])
    if not answers or not answers[0].get("value", "").strip():
        raise ValueError(f"query lacks a canonical answer: {query!r}")
    # Keep the benchmark's complete answer/alias structure; canonical answer
    # is the target for full-sequence NLL, aliases remain available for exact
    # match/generation metrics later.
    return {
        "canonical": answers[0]["value"].strip(),
        "aliases": [alias for answer in answers for alias in answer.get("aliases", []) if alias.strip()],
        "all_answers": answers,
    }


def tokenize_boundary(tokenizer, prompt, answer):
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    full_ids = tokenizer(prompt + answer, add_special_tokens=False).input_ids
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError(f"tokenizer changes prompt at boundary: {prompt!r} + {answer!r}")
    answer_count = len(full_ids) - len(prompt_ids)
    if answer_count < 1:
        raise ValueError(f"answer has no tokens: {answer!r}")
    return {"prompt_tokens": len(prompt_ids), "answer_tokens": answer_count, "boundary_ok": True}


def flatten_axis(axis, cases, tokenizer):
    probes = []
    for case_index, case in enumerate(cases):
        test_queries = case.get("test_queries", [])
        if not test_queries:
            continue
        for query_index, query in enumerate(test_queries):
            prompt = query.get("prompt", "").strip()
            answer = answer_payload(query)
            if not prompt:
                raise ValueError(f"empty {axis} probe prompt")
            # Query prompts conventionally omit the answer separator.
            score_answer = " " + answer["canonical"]
            probe = {
                "axis": axis,
                "case_index": case_index,
                "query_index": query_index,
                "prompt": prompt,
                "score_answer": score_answer,
                "answer": answer,
                "query_type": query.get("query_type"),
                "subject_id": query.get("subject_id"),
                "relation": query.get("relation"),
                "target_ids": query.get("target_ids", []),
                "phrase": query.get("phrase"),
                "second_relation": query.get("second_relation"),
                "second_hop_target_ids": query.get("second_hop_target_ids", []),
                "test_condition": case.get("test_condition"),
                "condition_queries": case.get("condition_queries", []),
            }
            probe.update(tokenize_boundary(tokenizer, prompt, score_answer))
            probes.append(probe)
    return probes


def map_record(record, source_subset, source_index, tokenizer):
    edit_prompt, edit_answer = split_edit_prompt(record["edit"]["prompt"])
    edit = record["edit"]
    result = {
        "edit_id": f"RE_{source_subset.upper()}_{source_index:04d}",
        "source": {"subset": source_subset, "index": source_index, "example_type": record["example_type"]},
        "metadata": {
            "relation": edit["relation"], "subject_id": edit["subject_id"], "target_id": edit["target_id"],
            "has_original_fact": "original_fact" in edit,
        },
        "acquisition": {
            "role": "G_acquisition_target", "prompt": edit_prompt, "score_answer": edit_answer,
            "canonical_answer": edit_answer.lstrip(), "original_edit_prompt": edit["prompt"],
        },
        "protected": {
            axis: flatten_axis(axis, record.get(axis, []), tokenizer) for axis in PROTECTED_AXES
        },
        "ripples": {axis: flatten_axis(axis, record.get(axis, []), tokenizer) for axis in RIPPLE_AXES},
    }
    result["acquisition"].update(tokenize_boundary(tokenizer, edit_prompt, edit_answer))
    if "original_fact" in edit:
        result["original_fact"] = edit["original_fact"]
    result["counts"] = {
        "protected_probes": sum(len(probes) for probes in result["protected"].values()),
        "ripple_probes": sum(len(probes) for probes in result["ripples"].values()),
        "ripple_probes_by_axis": {axis: len(probes) for axis, probes in result["ripples"].items()},
    }
    if not result["counts"]["protected_probes"] or not result["counts"]["ripple_probes"]:
        raise ValueError("record must have at least one protected and one ripple probe")
    return result


def select_balanced(candidates, quotas, seed):
    """Use equal source quotas and relation round-robin within each source."""
    selected, used_prompts = [], set()
    for subset, quota in quotas.items():
        buckets = defaultdict(list)
        for item in candidates[subset]:
            buckets[item["metadata"]["relation"]].append(item)
        for relation in buckets:
            buckets[relation] = stable_order(buckets[relation], seed, lambda item: item["edit_id"])
        relation_order = stable_order(list(buckets), seed, str)
        cursor = defaultdict(int)
        subset_selected = []
        while len(subset_selected) < quota:
            progressed = False
            # One edit per relation before taking a second: this makes relation
            # leave-one-category-out evaluation less dominated by any relation.
            for relation in relation_order:
                while cursor[relation] < len(buckets[relation]):
                    item = buckets[relation][cursor[relation]]
                    cursor[relation] += 1
                    if item["acquisition"]["original_edit_prompt"] in used_prompts:
                        continue
                    subset_selected.append(item)
                    used_prompts.add(item["acquisition"]["original_edit_prompt"])
                    progressed = True
                    break
                if len(subset_selected) == quota:
                    break
            if not progressed:
                raise RuntimeError(f"only found {len(subset_selected)} usable {subset} records; need {quota}")
        selected.extend(subset_selected)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, default=BENCHMARK_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=str(args.cache_dir), local_files_only=True)
    candidates, rejected = {}, Counter()
    for subset in QUOTAS:
        records = json.loads((args.benchmark_dir / f"{subset}.json").read_text())
        valid = []
        for index, record in enumerate(records):
            try:
                valid.append(map_record(record, subset, index, tokenizer))
            except (KeyError, TypeError, ValueError) as exc:
                rejected[f"{subset}:{str(exc).split(':')[0]}"] += 1
        candidates[subset] = valid
        print(f"{subset}: {len(valid)}/{len(records)} records usable")

    selected = select_balanced(candidates, QUOTAS, args.seed)
    if len(selected) != 100:
        raise RuntimeError(f"selection size is {len(selected)}, expected 100")
    if any(not item["acquisition"]["boundary_ok"] for item in selected):
        raise RuntimeError("an acquisition boundary check failed")

    relation_counts = Counter(item["metadata"]["relation"] for item in selected)
    source_counts = Counter(item["source"]["subset"] for item in selected)
    protected_by_axis = Counter(
        axis for item in selected for axis, probes in item["protected"].items() for _ in probes
    )
    ripple_by_axis = Counter(axis for item in selected for axis, probes in item["ripples"].items() for _ in probes)
    protected_total = sum(item["counts"]["protected_probes"] for item in selected)
    ripple_total = sum(item["counts"]["ripple_probes"] for item in selected)
    payload = {
        "schema_version": 1,
        "purpose": "External validation dataset only; no model update has been run.",
        "model_tokenizer": args.model,
        "selection": {"seed": args.seed, "quotas": QUOTAS, "strategy": "equal source quotas; deterministic relation round-robin"},
        "role_mapping": {
            "G": "edit prompt and counterfactual/recent target",
            "D": "Relation_Specificity plus Forgetfulness (previous-storage); facts expected to remain unchanged",
            "ripple": list(RIPPLE_AXES),
            "not_damage": list(RIPPLE_AXES),
            "loss": "full canonical answer-token negative log likelihood; aliases retained for later exact-match metrics",
        },
        "edits": selected,
    }
    summary = {
        "dataset": str(args.output_json), "edits": len(selected), "source_counts": dict(source_counts),
        "unique_relations": len(relation_counts), "max_edits_per_relation": max(relation_counts.values()),
        "relation_counts": dict(sorted(relation_counts.items())),
        "protected_probes": protected_total, "protected_probes_by_axis": dict(protected_by_axis),
        "ripple_probes": ripple_total, "ripple_probes_by_axis": dict(ripple_by_axis),
        "all_acquisition_boundaries_valid": True,
        "all_probe_boundaries_valid": all(
            probe["boundary_ok"] for item in selected for probes in item["protected"].values() for probe in probes
        ) and all(
            probe["boundary_ok"] for item in selected for probes in item["ripples"].values() for probe in probes
        ),
        "rejected_records": dict(sorted(rejected.items())),
        "role_mapping": payload["role_mapping"],
    }
    for path in (args.output_json, args.output_csv, args.summary):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    fields = ["edit_id", "source_subset", "example_type", "relation", "subject_id", "target_id", "edit_prompt", "edit_answer", "edit_answer_tokens", "protected_probes", "ripple_probes", "ripple_probes_by_axis", "has_original_fact"]
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in selected:
            writer.writerow({
                "edit_id": item["edit_id"], "source_subset": item["source"]["subset"],
                "example_type": item["source"]["example_type"], "relation": item["metadata"]["relation"],
                "subject_id": item["metadata"]["subject_id"], "target_id": item["metadata"]["target_id"],
                "edit_prompt": item["acquisition"]["prompt"], "edit_answer": item["acquisition"]["score_answer"],
                "edit_answer_tokens": item["acquisition"]["answer_tokens"],
                "protected_probes": item["counts"]["protected_probes"], "ripple_probes": item["counts"]["ripple_probes"],
                "ripple_probes_by_axis": json.dumps(item["counts"]["ripple_probes_by_axis"], sort_keys=True),
                "has_original_fact": item["metadata"]["has_original_fact"],
            })
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.summary}")
    print(f"100 edits: protected={protected_total}, ripple={ripple_total}, relations={len(relation_counts)}")


if __name__ == "__main__":
    main()
