#!/usr/bin/env python3
"""Independent 20-edit, 10-configuration Qwen update-strength calibration.

Updates apply only to disposable copies; this neither forecasts nor saves a
model checkpoint.  It deliberately does not call the four-candidate loop.
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
ROWS_OUTPUT = ROOT / "results" / "external_validation" / "rippleedits_qwen_update_strength_calibration.csv"
SUMMARY_OUTPUT = ROOT / "results" / "external_validation" / "rippleedits_qwen_update_strength_calibration_summary.json"
KNOWN_NLL_PER_TOKEN = 4.0
SUBSET_QUOTAS = {"random": 7, "recent": 7, "popular": 6}
CANDIDATES = (
    {"name": "full_lr_1e-5_steps1", "scope": "full", "lr": 1e-5, "steps": 1},
    {"name": "full_lr_1e-5_steps3", "scope": "full", "lr": 1e-5, "steps": 3},
    {"name": "full_lr_1e-5_steps5", "scope": "full", "lr": 1e-5, "steps": 5},
    {"name": "full_lr_1e-5_steps10", "scope": "full", "lr": 1e-5, "steps": 10},
    {"name": "full_lr_2e-5_steps5", "scope": "full", "lr": 2e-5, "steps": 5},
    {"name": "full_lr_5e-5_steps3", "scope": "full", "lr": 5e-5, "steps": 3},
    {"name": "early_lr_1e-5_steps5", "scope": "early_0_7", "lr": 1e-5, "steps": 5},
    {"name": "early_lr_1e-5_steps10", "scope": "early_0_7", "lr": 1e-5, "steps": 10},
    {"name": "late_lr_1e-5_steps5", "scope": "late_16_23", "lr": 1e-5, "steps": 5},
    {"name": "late_lr_1e-5_steps10", "scope": "late_16_23", "lr": 1e-5, "steps": 10},
)


def hash_order(value): return hashlib.sha256(value.encode()).hexdigest()
def edit_seed(edit_id): return int(hash_order(edit_id)[:8], 16)
def mean(values): return sum(values) / len(values) if values else None
def finite(value): return value is not None and math.isfinite(value)


def device_for(requested):
    return torch.device(requested) if requested != "auto" else torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def seed_everything(seed):
    torch.manual_seed(seed)
    if torch.backends.mps.is_available(): torch.mps.manual_seed(seed)


def calibration_subset(edits):
    """Deterministic 7/7/6 source balance and relation round-robin."""
    output = []
    for source, quota in SUBSET_QUOTAS.items():
        buckets = defaultdict(list)
        for edit in edits:
            if edit["source"]["subset"] == source: buckets[edit["metadata"]["relation"]].append(edit)
        for bucket in buckets.values(): bucket.sort(key=lambda item: hash_order(item["edit_id"]))
        relations = sorted(buckets, key=hash_order); indices = defaultdict(int); selected = []
        while len(selected) < quota:
            moved = False
            for relation in relations:
                if indices[relation] < len(buckets[relation]):
                    selected.append(buckets[relation][indices[relation]]); indices[relation] += 1; moved = True
                    if len(selected) == quota: break
            if not moved: raise RuntimeError(f"insufficient {source} edits")
        output.extend(selected)
    if len(output) != 20 or len({item["edit_id"] for item in output}) != 20: raise RuntimeError("subset is not 20 distinct edits")
    return output


def targets(edit):
    common = {"edit_id": edit["edit_id"], "source_subset": edit["source"]["subset"], "relation": edit["metadata"]["relation"]}
    acquisition = edit["acquisition"]
    result = [{**common, "role": "acquisition", "axis": "acquisition", "probe_id": "acquisition", "prompt": acquisition["prompt"], "answer": acquisition["score_answer"]}]
    for section, role in (("protected", "protected"), ("ripples", "ripple")):
        for axis, probes in edit[section].items():
            for probe in probes:
                result.append({**common, "role": role, "axis": axis, "probe_id": f"{axis}:{probe['case_index']}:{probe['query_index']}", "prompt": probe["prompt"], "answer": probe["score_answer"]})
    return result


def load_baseline(path):
    with path.open(newline="", encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
    if len(rows) != 1378: raise ValueError(f"baseline audit contains {len(rows)}, expected 1378 rows")
    result = {}
    for row in rows:
        key = (row["edit_id"], row["role"], row["axis"], row["probe_id"])
        if key in result: raise ValueError(f"duplicate baseline probe {key}")
        result[key] = float(row["canonical_answer_nll_per_token"])
    return result


def tokenise(items, tokenizer):
    for item in items:
        prefix = tokenizer(item["prompt"], add_special_tokens=False).input_ids
        full = tokenizer(item["prompt"] + item["answer"], add_special_tokens=False).input_ids
        if full[:len(prefix)] != prefix or len(full) == len(prefix): raise ValueError(f"invalid answer boundary {item['edit_id']} {item['probe_id']}")
        item["input_ids"], item["answer_start"], item["answer_tokens"] = full, len(prefix), len(full) - len(prefix)


def set_trainable(model, scope):
    for parameter in model.parameters(): parameter.requires_grad = scope == "full"
    if scope == "full": return
    layers = getattr(model.model, "layers", None)
    if layers is None or len(layers) != 24: raise ValueError("Qwen must expose 24 decoder layers")
    indices = range(0, 8) if scope == "early_0_7" else range(16, 24)
    for index in indices:
        for parameter in layers[index].parameters(): parameter.requires_grad = True


def update(model, acquisition, candidate, seed, device):
    seed_everything(seed); set_trainable(model, candidate["scope"])
    parameters = [p for p in model.parameters() if p.requires_grad]
    before = {id(p): p.detach().clone() for p in parameters}
    optimizer = torch.optim.AdamW(parameters, lr=candidate["lr"], weight_decay=0.0); model.train()
    for _ in range(candidate["steps"]):
        optimizer.zero_grad(set_to_none=True)
        ids = torch.tensor(acquisition["input_ids"], device=device).unsqueeze(0); labels = ids.clone(); labels[:, :acquisition["answer_start"]] = -100
        loss = model(input_ids=ids, labels=labels).loss
        if not torch.isfinite(loss): raise FloatingPointError("non-finite update loss")
        loss.backward(); optimizer.step()
    norm = math.sqrt(sum((p.detach() - before[id(p)]).float().pow(2).sum().item() for p in parameters))
    count = sum(p.numel() for p in parameters); del optimizer, before
    return norm, count


def score(model, items, pad_id, device, batch_size):
    model.eval()
    for offset in range(0, len(items), batch_size):
        batch = items[offset:offset + batch_size]; width = max(len(item["input_ids"]) for item in batch)
        ids = torch.full((len(batch), width), pad_id, dtype=torch.long, device=device); mask = torch.zeros_like(ids)
        for i, item in enumerate(batch):
            values = torch.tensor(item["input_ids"], device=device); ids[i, :len(values)] = values; mask[i, :len(values)] = 1
        with torch.inference_mode(): logits = torch.log_softmax(model(input_ids=ids, attention_mask=mask).logits.float(), dim=-1)
        for i, item in enumerate(batch):
            start, end = item["answer_start"], len(item["input_ids"]); positions = torch.arange(start - 1, end - 1, device=device)
            item["after"] = float((-logits[i, positions, ids[i, start:end]]).mean().cpu())


def build_row(edit, candidate, items, baseline, norm, count, status):
    key = lambda item: (item["edit_id"], item["role"], item["axis"], item["probe_id"])
    acquisition = items[0]; protected = [item for item in items if item["role"] == "protected"]; ripple = [item for item in items if item["role"] == "ripple"]
    before = baseline[key(acquisition)]; protected_delta = [item["after"] - baseline[key(item)] for item in protected]
    known_delta = [item["after"] - baseline[key(item)] for item in protected if baseline[key(item)] <= KNOWN_NLL_PER_TOKEN]
    ripple_before, ripple_after = [baseline[key(item)] for item in ripple], [item["after"] for item in ripple]
    if not all(finite(value) for value in [before, acquisition["after"], norm] + protected_delta + ripple_after): status = "nonfinite"
    return {"edit_id": edit["edit_id"], "candidate": candidate["name"], "learning_rate": candidate["lr"], "update_steps": candidate["steps"], "updated_layer_scope": candidate["scope"], "seed": edit_seed(edit["edit_id"]), "source_subset": edit["source"]["subset"], "relation": edit["metadata"]["relation"], "acquisition_loss_before": before, "acquisition_loss_after": acquisition["after"], "acquisition_gain": before - acquisition["after"], "known_protected_probes": len(known_delta), "D_known": mean(known_delta), "D_all": mean(protected_delta), "ripple_loss_before": mean(ripple_before), "ripple_loss_after": mean(ripple_after), "ripple_loss_change": mean([after - prior for prior, after in zip(ripple_before, ripple_after)]), "update_norm": norm, "trainable_parameters": count, "numerical_stability": status}


def make_summary(rows, subset):
    metric_names = ("acquisition_gain", "D_known", "D_all", "ripple_loss_change", "update_norm")
    def describe(group):
        result = {"n": len(group), "stable_n": sum(row["numerical_stability"] == "stable" for row in group)}
        for metric in metric_names:
            values = [float(row[metric]) for row in group if row[metric] not in (None, "")]; result[f"mean_{metric}"] = mean(values)
        return result
    def groups(keys):
        grouped = defaultdict(list)
        for row in rows: grouped[tuple(row[key] for key in keys)].append(row)
        return [{**dict(zip(keys, group_key)), **describe(group)} for group_key, group in sorted(grouped.items())]
    by_edit = defaultdict(list)
    for row in rows: by_edit[row["edit_id"]].append(row)
    wanted = {candidate["name"] for candidate in CANDIDATES}
    suffixes = all(any(candidate["name"].endswith(f"steps{step}") for candidate in CANDIDATES) for step in (1, 3, 5, 10))
    return {"purpose": "Independent 20-edit update-strength calibration; no forecasting.", "model": MODEL_NAME, "baseline_known_nll_per_token_cutoff": KNOWN_NLL_PER_TOKEN, "subset_edit_ids": [edit["edit_id"] for edit in subset], "subset_source_counts": {source: sum(edit["source"]["subset"] == source for edit in subset) for source in SUBSET_QUOTAS}, "candidate_definitions": CANDIDATES, "expected_rows": 200, "completed_rows": len(rows), "validation": {"csv_exactly_200_rows": len(rows) == 200, "unique_edit_ids": len(by_edit) == 20, "unique_candidate_names": len({row["candidate"] for row in rows}) == 10, "each_edit_has_all_10_candidates": len(by_edit) == 20 and all(len(group) == 10 and {row["candidate"] for row in group} == wanted for group in by_edit.values()), "same_seed_per_edit": all(len({row["seed"] for row in group}) == 1 for group in by_edit.values()), "step_suffixes_present": suffixes}, "candidate_level": groups(("candidate",)), "source_level": groups(("candidate", "source_subset")), "relation_level": groups(("candidate", "relation"))}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto"); parser.add_argument("--batch-size", type=int, default=8); parser.add_argument("--dataset", type=Path, default=DATASET); parser.add_argument("--baseline", type=Path, default=BASELINE); args = parser.parse_args()
    device = device_for(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available(): parser.error("MPS was requested but is unavailable")
    dataset = json.loads(args.dataset.read_text()); subset = calibration_subset(dataset["edits"]); baseline = load_baseline(args.baseline)
    required = {(item["edit_id"], item["role"], item["axis"], item["probe_id"]) for edit in subset for item in targets(edit)}
    if not required.issubset(baseline): raise ValueError("baseline audit does not cover calibration subset")
    ROWS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for path in (ROWS_OUTPUT, SUMMARY_OUTPUT):
        if path.exists(): path.unlink()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=str(CACHE_DIR), local_files_only=True); pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=str(CACHE_DIR), local_files_only=True).to(device).eval(); rows = []
    print(f"Calibration: 20 edits x 10 configurations on {device}; no forecasting", flush=True)
    with ROWS_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = None
        for edit in subset:
            items = targets(edit); tokenise(items, tokenizer)
            for candidate in CANDIDATES:
                model = copy.deepcopy(base_model); status = "stable"
                try: norm, count = update(model, items[0], candidate, edit_seed(edit["edit_id"]), device); score(model, items, pad_id, device, args.batch_size)
                except (FloatingPointError, RuntimeError) as exc:
                    status, norm, count = f"error:{type(exc).__name__}", float("nan"), 0
                    for item in items: item["after"] = float("nan")
                row = build_row(edit, candidate, items, baseline, norm, count, status)
                if writer is None: writer = csv.DictWriter(handle, fieldnames=list(row)); writer.writeheader()
                writer.writerow(row); handle.flush(); rows.append(row); print(f"{len(rows)}/200 {edit['edit_id']} {candidate['name']} {row['numerical_stability']}", flush=True)
                del model; gc.collect()
                if device.type == "mps": torch.mps.empty_cache()
    with ROWS_OUTPUT.open(newline="", encoding="utf-8") as handle: checked = list(csv.DictReader(handle))
    report = make_summary(checked, subset); SUMMARY_OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    if not all(report["validation"].values()): raise RuntimeError(f"calibration validation failed: {report['validation']}")
    print(f"Complete: {len(checked)} rows written to {ROWS_OUTPUT}", flush=True)


if __name__ == "__main__": main()
