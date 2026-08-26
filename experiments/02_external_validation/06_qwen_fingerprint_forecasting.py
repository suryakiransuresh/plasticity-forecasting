#!/usr/bin/env python3
"""Pre-update-only Qwen fingerprints and grouped held-out-edit forecasting.

Reads the final candidate outcomes as frozen data.  This script never creates
an optimizer or calls an update routine; every gradient is cleared immediately
after it is measured.
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
CACHE_DIR = ROOT / "results" / "cache" / "huggingface"
DATASET = ROOT / "data" / "external_validation" / "rippleedits_qwen_100.json"
BASELINE = ROOT / "results" / "external_validation" / "rippleedits_qwen_baseline_knowledge_audit.csv"
OUTCOMES = ROOT / "results" / "external_validation" / "rippleedits_qwen_final_candidate_sweep.csv"
FINGERPRINTS = ROOT / "results" / "external_validation" / "rippleedits_qwen_preupdate_fingerprints.csv"
FORECASTING_DATASET = ROOT / "results" / "external_validation" / "rippleedits_qwen_forecasting_dataset.csv"
SUMMARY = ROOT / "results" / "external_validation" / "rippleedits_qwen_fingerprint_forecasting_summary.json"
KNOWN_CUTOFF = 4.0
N_SPLITS = 5
RANDOM_STATE = 20260826


def device_for(value):
    return torch.device(value) if value != "auto" else torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def target_key(item): return (item["edit_id"], item["role"], item["axis"], item["probe_id"])
def l2(values): return math.sqrt(sum(value * value for value in values))


def load_baseline(path):
    with path.open(newline="", encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
    if len(rows) != 1378: raise ValueError(f"baseline audit has {len(rows)} rows, expected 1378")
    return {(row["edit_id"], row["role"], row["axis"], row["probe_id"]): float(row["canonical_answer_nll_per_token"]) for row in rows}


def edit_targets(edit):
    result = []
    acquisition = edit["acquisition"]
    result.append({"edit_id": edit["edit_id"], "role": "acquisition", "axis": "acquisition", "probe_id": "acquisition", "prompt": acquisition["prompt"], "answer": acquisition["score_answer"]})
    for section, role in (("protected", "protected"), ("ripples", "ripple")):
        for axis, probes in edit[section].items():
            for probe in probes:
                result.append({"edit_id": edit["edit_id"], "role": role, "axis": axis, "probe_id": f"{axis}:{probe['case_index']}:{probe['query_index']}", "prompt": probe["prompt"], "answer": probe["score_answer"]})
    return result


def encode(item, tokenizer, device):
    prefix = tokenizer(item["prompt"], add_special_tokens=False).input_ids
    full = tokenizer(item["prompt"] + item["answer"], add_special_tokens=False).input_ids
    if full[:len(prefix)] != prefix or len(full) == len(prefix): raise ValueError(f"invalid boundary: {item['edit_id']} {item['probe_id']}")
    ids = torch.tensor(full, dtype=torch.long, device=device).unsqueeze(0)
    labels = ids.clone(); labels[:, :len(prefix)] = -100
    return ids, labels, len(prefix), len(full)


def gradient_profile(model, tokenizer, device, item):
    """Full-answer-loss parameter-gradient magnitude/profile, before updates."""
    ids, labels, _, _ = encode(item, tokenizer, device)
    model.zero_grad(set_to_none=True)
    loss = model(input_ids=ids, labels=labels).loss
    loss.backward()
    layers = model.model.layers
    norms = []
    for layer in layers:
        norms.append(math.sqrt(sum(p.grad.detach().float().pow(2).sum().item() for p in layer.parameters() if p.grad is not None)))
    total = math.sqrt(sum(p.grad.detach().float().pow(2).sum().item() for p in model.parameters() if p.grad is not None))
    model.zero_grad(set_to_none=True)
    transformer = l2(norms)
    if transformer == 0 or not math.isfinite(transformer): raise ValueError("invalid transformer gradient norm")
    groups = [l2(norms[0:8]), l2(norms[8:16]), l2(norms[16:24])]
    squared = np.asarray(norms, dtype=float) ** 2 / transformer ** 2
    nonzero = squared[squared > 0]
    return {"acquisition_baseline_loss": float(loss.detach().cpu()), "total_grad_norm": total,
            **{f"layer_{i}_grad_norm": value for i, value in enumerate(norms)},
            "early_grad_norm": groups[0], "middle_grad_norm": groups[1], "late_grad_norm": groups[2],
            "early_grad_fraction": groups[0] ** 2 / transformer ** 2, "middle_grad_fraction": groups[1] ** 2 / transformer ** 2, "late_grad_fraction": groups[2] ** 2 / transformer ** 2,
            "peak_gradient_layer": int(np.argmax(norms)), "layer_concentration_hhi": float(np.sum(squared ** 2)),
            "layer_entropy": float(-np.sum(nonzero * np.log(nonzero))), "top3_gradient_fraction": float(np.sort(squared)[-3:].sum()),
            "gradient_profile_centroid": float(np.dot(np.arange(24), squared))}


def hidden_signature(model, tokenizer, device, item):
    """Layerwise hidden-state gradient signature of the full canonical loss."""
    ids, labels, start, end = encode(item, tokenizer, device)
    model.zero_grad(set_to_none=True)
    output = model(input_ids=ids, labels=labels, output_hidden_states=True, return_dict=True)
    states = output.hidden_states[1:]
    if len(states) != 24: raise ValueError(f"expected 24 hidden-state layers, got {len(states)}")
    for state in states: state.retain_grad()
    output.loss.backward()
    signature = [state.grad[0, start - 1:end - 1, :].mean(dim=0).detach().float().cpu() for state in states]
    model.zero_grad(set_to_none=True)
    return torch.cat(signature)


def cosine(left, right):
    return float(torch.nn.functional.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0)).item())


def alignment(edit, acquisition_signature, model, tokenizer, device, baseline):
    probes = [item for item in edit_targets(edit) if item["role"] == "protected" and baseline[target_key(item)] <= KNOWN_CUTOFF]
    values, by_axis = [], defaultdict(list)
    for probe in probes:
        value = cosine(acquisition_signature, hidden_signature(model, tokenizer, device, probe))
        values.append(value); by_axis[probe["axis"]].append(value)
    output = {"known_protected_probes": len(values), "alignment_mean": np.nan, "alignment_min": np.nan, "alignment_max": np.nan,
              "relation_specificity_alignment_mean": np.nan, "forgetfulness_alignment_mean": np.nan}
    if values: output.update({"alignment_mean": float(np.mean(values)), "alignment_min": float(np.min(values)), "alignment_max": float(np.max(values))})
    if by_axis["Relation_Specificity"]: output["relation_specificity_alignment_mean"] = float(np.mean(by_axis["Relation_Specificity"]))
    if by_axis["Forgetfulness"]: output["forgetfulness_alignment_mean"] = float(np.mean(by_axis["Forgetfulness"]))
    return output


def extract_fingerprints(dataset, baseline, tokenizer, model, device):
    existing = pd.read_csv(FINGERPRINTS) if FINGERPRINTS.exists() else pd.DataFrame()
    ids = {edit["edit_id"] for edit in dataset["edits"]}
    if not existing.empty and (existing["edit_id"].duplicated().any() or not set(existing["edit_id"]).issubset(ids)): raise ValueError("invalid resumable fingerprint file")
    rows = existing.to_dict("records") if not existing.empty else []; done = {row["edit_id"] for row in rows}
    for index, edit in enumerate(dataset["edits"], 1):
        if edit["edit_id"] in done: continue
        acquisition = edit_targets(edit)[0]
        profile = gradient_profile(model, tokenizer, device, acquisition)
        signature = hidden_signature(model, tokenizer, device, acquisition)
        row = {"edit_id": edit["edit_id"], "source_subset": edit["source"]["subset"], "relation": edit["metadata"]["relation"],
               **profile, **alignment(edit, signature, model, tokenizer, device, baseline)}
        numeric = [value for key, value in row.items() if key not in {"edit_id", "source_subset", "relation"} and not (isinstance(value, float) and math.isnan(value))]
        if not all(math.isfinite(float(value)) for value in numeric): raise ValueError(f"non-finite fingerprint: {edit['edit_id']}")
        rows.append(row); pd.DataFrame(rows).to_csv(FINGERPRINTS, index=False)
        print(f"Fingerprint {index}/100 {edit['edit_id']}", flush=True)
    frame = pd.DataFrame(rows)
    if len(frame) != 100: raise RuntimeError(f"fingerprints have {len(frame)} rows, expected 100")
    frame.to_csv(FINGERPRINTS, index=False); return frame


def pipeline(features, classifier=False):
    categorical = [feature for feature in features if feature in {"candidate", "updated_layer_scope"}]
    numeric = [feature for feature in features if feature not in categorical]
    estimator = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced") if classifier else RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    return Pipeline([("preprocess", ColumnTransformer([("categorical", OneHotEncoder(handle_unknown="ignore"), categorical), ("numeric", SimpleImputer(strategy="median"), numeric)])), ("model", estimator)])


def regression_cv(frame, feature_groups):
    results = []
    for target, subset in (("acquisition_gain", frame[frame["numerical_stability"] == "stable"]), ("D_known", frame[(frame["numerical_stability"] == "stable") & (frame["known_protected_probes"] > 0) & frame["D_known"].notna()])):
        if subset["edit_id"].nunique() < N_SPLITS: continue
        splitter = GroupKFold(n_splits=N_SPLITS)
        for group_name, features in feature_groups.items():
            fold_scores = []
            for train_idx, test_idx in splitter.split(subset, subset[target], subset["edit_id"]):
                train, test = subset.iloc[train_idx], subset.iloc[test_idx]
                model = pipeline(features); model.fit(train[features], train[target]); prediction = model.predict(test[features])
                fold_scores.append((mean_absolute_error(test[target], prediction), r2_score(test[target], prediction)))
            results.append({"analysis": "regression", "target": target, "feature_group": group_name, "folds": len(fold_scores), "mae": float(np.mean([x[0] for x in fold_scores])), "r2": float(np.mean([x[1] for x in fold_scores]))})
    return results


def stability_cv(frame, feature_groups):
    labels = (frame["numerical_stability"] != "stable").astype(int)
    class_counts = labels.value_counts(); group_classes = frame.assign(label=labels).groupby("edit_id")["label"].max().value_counts()
    if len(class_counts) != 2 or class_counts.min() < 10 or len(group_classes) != 2 or group_classes.min() < N_SPLITS: return {"available": False, "reason": "insufficient stable/nonfinite class support"}
    output = []
    for name, features in feature_groups.items():
        scores = []
        for train_idx, test_idx in GroupKFold(N_SPLITS).split(frame, labels, frame["edit_id"]):
            train, test = frame.iloc[train_idx], frame.iloc[test_idx]; train_y, test_y = labels.iloc[train_idx], labels.iloc[test_idx]
            if train_y.nunique() < 2 or test_y.nunique() < 2: continue
            model = pipeline(features, classifier=True); model.fit(train[features], train_y); probability = model.predict_proba(test[features])[:, 1]; prediction = model.predict(test[features])
            scores.append((roc_auc_score(test_y, probability), balanced_accuracy_score(test_y, prediction)))
        if scores: output.append({"feature_group": name, "folds": len(scores), "roc_auc": float(np.mean([x[0] for x in scores])), "balanced_accuracy": float(np.mean([x[1] for x in scores]))})
    return {"available": bool(output), "results": output}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto"); args = parser.parse_args()
    if not OUTCOMES.exists(): raise RuntimeError(f"Missing frozen final sweep: {OUTCOMES}")
    outcomes = pd.read_csv(OUTCOMES)
    if len(outcomes) != 400 or outcomes.duplicated(["edit_id", "candidate"]).any(): raise ValueError("final outcome table must have 400 unique edit/candidate rows")
    dataset = json.loads(DATASET.read_text()); baseline = load_baseline(BASELINE); device = device_for(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available(): parser.error("MPS was requested but is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=str(CACHE_DIR), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=str(CACHE_DIR), local_files_only=True).to(device).eval()
    FINGERPRINTS.parent.mkdir(parents=True, exist_ok=True)
    fingerprints = extract_fingerprints(dataset, baseline, tokenizer, model, device)
    joined = outcomes.merge(fingerprints, on="edit_id", validate="many_to_one", suffixes=("", "_fingerprint"))
    if len(joined) != 400 or joined["total_grad_norm"].isna().any(): raise RuntimeError("incomplete fingerprint/outcome join")
    joined.to_csv(FORECASTING_DATASET, index=False)
    magnitude = ["acquisition_baseline_loss", "total_grad_norm", *[f"layer_{i}_grad_norm" for i in range(24)], "early_grad_norm", "middle_grad_norm", "late_grad_norm", "early_grad_fraction", "middle_grad_fraction", "late_grad_fraction", "peak_gradient_layer", "layer_concentration_hhi", "layer_entropy", "top3_gradient_fraction", "gradient_profile_centroid"]
    alignment = ["alignment_mean", "alignment_min", "alignment_max", "relation_specificity_alignment_mean", "forgetfulness_alignment_mean", "known_protected_probes"]
    candidate = ["candidate", "learning_rate", "update_steps", "updated_layer_scope"]
    groups = {"candidate_only": candidate, "magnitude_profile_plus_candidate": magnitude + candidate, "alignment_plus_candidate": alignment + candidate, "full_fingerprint_plus_candidate": magnitude + alignment + candidate}
    results = regression_cv(joined, groups); stability = stability_cv(joined, groups)
    SUMMARY.write_text(json.dumps({"purpose": "Pre-update fingerprint forecasting evaluation only; no candidate selection.", "fingerprint_rows": len(fingerprints), "frozen_outcome_rows": len(outcomes), "joined_rows": len(joined), "cv": {"scheme": "GroupKFold by edit_id", "splits": N_SPLITS}, "feature_groups": groups, "regression_results": results, "secondary_stability_analysis": stability}, indent=2) + "\n")
    print(f"Wrote {FINGERPRINTS}, {FORECASTING_DATASET}, and {SUMMARY}")


if __name__ == "__main__": main()
