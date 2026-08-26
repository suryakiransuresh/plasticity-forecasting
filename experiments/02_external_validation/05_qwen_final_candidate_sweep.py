#!/usr/bin/env python3
"""Final frozen four-candidate Qwen RippleEdits sweep; no forecasting."""

import argparse
import copy
import csv
import gc
import json
import math
from collections import defaultdict
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


HERE = Path(__file__).resolve().parent
SPEC = spec_from_file_location("qwen_calibration_helpers", HERE / "04_qwen_update_strength_calibration.py")
H = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(H)

ROWS_OUTPUT = H.ROOT / "results" / "external_validation" / "rippleedits_qwen_final_candidate_sweep.csv"
SUMMARY_OUTPUT = H.ROOT / "results" / "external_validation" / "rippleedits_qwen_final_candidate_sweep_summary.json"
CANDIDATES = (
    {"name": "full_lr_1e-5_steps1", "scope": "full", "lr": 1e-5, "steps": 1},
    {"name": "early_lr_1e-5_steps10", "scope": "early_0_7", "lr": 1e-5, "steps": 10},
    {"name": "late_lr_1e-5_steps10", "scope": "late_16_23", "lr": 1e-5, "steps": 10},
    {"name": "full_lr_2e-5_steps5", "scope": "full", "lr": 2e-5, "steps": 5},
)


def mean(values):
    return sum(values) / len(values) if values else None


def final_summary(rows):
    metrics = ("acquisition_gain", "D_known", "D_all", "ripple_loss_change", "update_norm")
    def describe(group):
        report = {"n": len(group), "stable_n": sum(row["numerical_stability"] == "stable" for row in group),
                  "unstable_n": sum(row["numerical_stability"] != "stable" for row in group)}
        for metric in metrics:
            values = [float(row[metric]) for row in group if row[metric] not in ("", None) and math.isfinite(float(row[metric]))]
            report[f"mean_{metric}"] = mean(values)
        return report
    def grouped(keys):
        groups = defaultdict(list)
        for row in rows: groups[tuple(row[key] for key in keys)].append(row)
        return [{**dict(zip(keys, key)), **describe(group)} for key, group in sorted(groups.items())]
    by_edit = defaultdict(list)
    for row in rows: by_edit[row["edit_id"]].append(row)
    expected = {candidate["name"] for candidate in CANDIDATES}
    return {
        "purpose": "Final frozen 100-edit candidate sweep only; no forecasting.", "model": H.MODEL_NAME,
        "baseline_known_nll_per_token_cutoff": H.KNOWN_NLL_PER_TOKEN, "ripple_excluded_from_damage": True,
        "candidate_definitions": CANDIDATES, "expected_rows": 400, "completed_rows": len(rows),
        "validation": {
            "csv_exactly_400_rows": len(rows) == 400, "unique_edit_ids": len(by_edit) == 100,
            "unique_candidate_names": len({row["candidate"] for row in rows}) == 4,
            "each_edit_has_all_4_candidates": len(by_edit) == 100 and all(len(group) == 4 and {row["candidate"] for row in group} == expected for group in by_edit.values()),
            "same_seed_per_edit": all(len({row["seed"] for row in group}) == 1 for group in by_edit.values()),
        },
        "candidate_level": grouped(("candidate",)), "source_level": grouped(("candidate", "source_subset")),
        "relation_level": grouped(("candidate", "relation")),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dataset", type=Path, default=H.DATASET)
    parser.add_argument("--baseline", type=Path, default=H.BASELINE)
    args = parser.parse_args()
    device = H.device_for(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available(): parser.error("MPS was requested but is unavailable")
    dataset = json.loads(args.dataset.read_text())
    edits = dataset["edits"]
    if len(edits) != 100: raise ValueError(f"frozen dataset has {len(edits)} edits, expected 100")
    baseline = H.load_baseline(args.baseline)
    needed = {(item["edit_id"], item["role"], item["axis"], item["probe_id"]) for edit in edits for item in H.targets(edit)}
    if not needed.issubset(baseline): raise ValueError("baseline audit does not cover the frozen dataset")
    ROWS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for path in (ROWS_OUTPUT, SUMMARY_OUTPUT):
        if path.exists(): path.unlink()
    tokenizer = AutoTokenizer.from_pretrained(H.MODEL_NAME, cache_dir=str(H.CACHE_DIR), local_files_only=True)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    base_model = AutoModelForCausalLM.from_pretrained(H.MODEL_NAME, cache_dir=str(H.CACHE_DIR), local_files_only=True).to(device).eval()
    rows = []
    print(f"Final sweep: 100 edits x 4 frozen candidates on {device}; no forecasting", flush=True)
    with ROWS_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = None
        for edit in edits:
            items = H.targets(edit); H.tokenise(items, tokenizer)
            for candidate in CANDIDATES:
                model = copy.deepcopy(base_model); status = "stable"
                try:
                    norm, count = H.update(model, items[0], candidate, H.edit_seed(edit["edit_id"]), device)
                    H.score(model, items, pad_id, device, args.batch_size)
                except (FloatingPointError, RuntimeError) as exc:
                    # Preserve the requested candidate with its instability;
                    # never retry it at a different hyperparameter setting.
                    status, norm, count = f"error:{type(exc).__name__}", float("nan"), 0
                    for item in items: item["after"] = float("nan")
                row = H.build_row(edit, candidate, items, baseline, norm, count, status)
                if writer is None: writer = csv.DictWriter(handle, fieldnames=list(row)); writer.writeheader()
                writer.writerow(row); handle.flush(); rows.append(row)
                print(f"{len(rows)}/400 {edit['edit_id']} {candidate['name']} {row['numerical_stability']}", flush=True)
                del model; gc.collect()
                if device.type == "mps": torch.mps.empty_cache()
    with ROWS_OUTPUT.open(newline="", encoding="utf-8") as handle: checked = list(csv.DictReader(handle))
    report = final_summary(checked); SUMMARY_OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    if not all(report["validation"].values()): raise RuntimeError(f"final-sweep validation failed: {report['validation']}")
    print(f"Complete: 400 rows written to {ROWS_OUTPUT}", flush=True)


if __name__ == "__main__": main()
