#!/usr/bin/env python3
"""OOF acquisition-only selection analysis over frozen Qwen outcomes.

This neither calls Qwen nor candidate-update code.  Damage is deliberately
retrospective: no damage prediction, threshold, or safety constraint is used.
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[2]
FORECASTING_DATASET = ROOT / "results" / "external_validation" / "rippleedits_qwen_forecasting_dataset.csv"
OUTCOMES = ROOT / "results" / "external_validation" / "rippleedits_qwen_final_candidate_sweep.csv"
ROWS_OUTPUT = ROOT / "results" / "external_validation" / "rippleedits_qwen_forecast_selection_oof.csv"
SUMMARY_OUTPUT = ROOT / "results" / "external_validation" / "rippleedits_qwen_forecast_selection_summary.json"
N_SPLITS, RANDOM_STATE = 5, 20260826


def mean(values): return float(np.mean(values)) if len(values) else None


def feature_groups(frame):
    magnitude = ["acquisition_baseline_loss", "total_grad_norm", *[f"layer_{i}_grad_norm" for i in range(24)],
                 "early_grad_norm", "middle_grad_norm", "late_grad_norm", "early_grad_fraction", "middle_grad_fraction", "late_grad_fraction",
                 "peak_gradient_layer", "layer_concentration_hhi", "layer_entropy", "top3_gradient_fraction", "gradient_profile_centroid"]
    alignment = ["alignment_mean", "alignment_min", "alignment_max", "relation_specificity_alignment_mean", "forgetfulness_alignment_mean", "known_protected_probes"]
    candidate = ["candidate", "learning_rate", "update_steps", "updated_layer_scope"]
    groups = {"candidate_only": candidate, "magnitude_profile_plus_candidate": magnitude + candidate,
              "full_fingerprint_plus_candidate": magnitude + alignment + candidate}
    missing = {name: sorted(set(columns) - set(frame.columns)) for name, columns in groups.items()}
    if any(missing.values()): raise ValueError(f"forecasting dataset lacks 06 feature columns: {missing}")
    return groups


def model_for(features):
    categorical = [name for name in features if name in {"candidate", "updated_layer_scope"}]
    numeric = [name for name in features if name not in categorical]
    return Pipeline([("preprocess", ColumnTransformer([("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
                                                          ("numeric", SimpleImputer(strategy="median"), numeric)])),
                     ("model", RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1))])


def stable(frame): return frame["numerical_stability"].eq("stable") & frame["acquisition_gain"].notna()


def fold_assignments(frame):
    edits = frame[["edit_id"]].drop_duplicates().reset_index(drop=True)
    if len(edits) != 100: raise ValueError(f"expected 100 edits, found {len(edits)}")
    splits = GroupKFold(n_splits=N_SPLITS).split(edits, groups=edits["edit_id"])
    result = {}
    for fold, (_, test_idx) in enumerate(splits, 1):
        for edit_id in edits.iloc[test_idx]["edit_id"]: result[edit_id] = fold
    return result


def oof_predictions(frame, features, assignments):
    result = frame.copy(); result["fold"] = result["edit_id"].map(assignments); result["predicted_acquisition_gain"] = np.nan
    for fold in sorted(result["fold"].unique()):
        train = result[(result["fold"] != fold) & stable(result)]
        test = result[result["fold"] == fold]
        model = model_for(features); model.fit(train[features], train["acquisition_gain"])
        result.loc[test.index, "predicted_acquisition_gain"] = model.predict(test[features])
    if result["predicted_acquisition_gain"].isna().any(): raise RuntimeError("incomplete OOF gain predictions")
    return result


def choose_rows(predicted, method):
    selections = []
    for edit_id, group in predicted.groupby("edit_id", sort=False):
        # No stability or damage constraint is applied to this forecast rank.
        chosen = group.sort_values(["predicted_acquisition_gain", "candidate"], ascending=[False, True]).iloc[0]
        oracle_pool = group[stable(group)]
        if oracle_pool.empty:
            oracle = None
        else:
            oracle = oracle_pool.sort_values(["acquisition_gain", "candidate"], ascending=[False, True]).iloc[0]
        train_fold = int(chosen["fold"])
        train = predicted[(predicted["fold"] != train_fold) & stable(predicted)]
        static_means = train.groupby("candidate")["acquisition_gain"].mean()
        static_candidate = static_means.sort_values(ascending=False).index[0]
        static_rows = group[(group["candidate"] == static_candidate) & stable(group)]
        static_gain = static_rows.iloc[0]["acquisition_gain"] if len(static_rows) else np.nan
        selections.append({"method": method, "edit_id": edit_id, "fold": train_fold, "source_subset": chosen["source_subset"],
                           "relation": chosen["relation"], "selected_candidate": chosen["candidate"],
                           "selected_predicted_gain": chosen["predicted_acquisition_gain"], "selected_realized_gain": chosen["acquisition_gain"],
                           "selected_D_known": chosen["D_known"], "selected_D_all": chosen["D_all"],
                           "selected_numerical_stability": chosen["numerical_stability"],
                           "oracle_stable_candidate": oracle["candidate"] if oracle is not None else "",
                           "oracle_stable_gain": oracle["acquisition_gain"] if oracle is not None else np.nan,
                           "top1_matches_stable_oracle": bool(oracle is not None and chosen["candidate"] == oracle["candidate"]),
                           "acquisition_regret": oracle["acquisition_gain"] - chosen["acquisition_gain"] if oracle is not None else np.nan,
                           "fold_trained_static_candidate": static_candidate, "static_realized_gain": static_gain,
                           "static_oracle_gap": oracle["acquisition_gain"] - static_gain if oracle is not None else np.nan})
    return selections


def describe(rows):
    # Every selected-vs-oracle comparison below uses precisely this same
    # eligible set.  A non-finite selected candidate is never silently
    # dropped from overall top-1 accuracy, but it is excluded from conditional
    # realized-gain comparisons because it has no finite realized gain.
    selected_stable = [row for row in rows if row["selected_numerical_stability"] == "stable"]
    eligible = [row for row in selected_stable if pd.notna(row["oracle_stable_gain"]) and pd.notna(row["selected_realized_gain"])]
    static_gap_rows = [row for row in eligible if pd.notna(row["static_realized_gain"])]
    selected = [row["selected_realized_gain"] for row in eligible]
    oracle = [row["oracle_stable_gain"] for row in eligible]
    static = [row["static_realized_gain"] for row in static_gap_rows]
    selected_gap = [row["selected_realized_gain"] for row in static_gap_rows]
    oracle_for_static = [row["oracle_stable_gain"] for row in static_gap_rows]
    gap = mean(oracle_for_static) - mean(static) if static else np.nan
    closed = (mean(selected_gap) - mean(static)) / gap if static and not math.isclose(gap, 0.0) else np.nan
    return {"total_edits": len(rows), "selected_stable_count": len(selected_stable),
            "selected_stable_rate": len(selected_stable) / len(rows) if rows else None,
            "selected_vs_oracle_eligible_edits": len(eligible),
            "conditional_realized_acquisition_gain_selected_stable": mean(selected),
            "oracle_stable_gain_on_selected_stable_subset": mean(oracle),
            "conditional_acquisition_regret_on_selected_stable_subset": mean([row["acquisition_regret"] for row in eligible]),
            "top1_candidate_selection_accuracy_overall": mean([bool(row["top1_matches_stable_oracle"]) for row in rows]),
            "top1_candidate_selection_accuracy_given_selected_stable": mean([bool(row["top1_matches_stable_oracle"]) for row in eligible]),
            "best_static_candidate_mean_realized_gain": mean(static), "fraction_oracle_gap_closed_vs_fold_trained_static": closed,
            "selected_D_known_retrospective": mean([row["selected_D_known"] for row in rows if pd.notna(row["selected_D_known"])]),
            "selected_D_all_retrospective": mean([row["selected_D_all"] for row in rows if pd.notna(row["selected_D_all"])]),
            "selected_instability_rate_retrospective": mean([row["selected_numerical_stability"] != "stable" for row in rows]),
            "candidate_choice_frequencies": dict(Counter(row["selected_candidate"] for row in rows)),
            "fold_trained_static_choice_frequencies": dict(Counter(row["fold_trained_static_candidate"] for row in rows))}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--reuse-oof", action="store_true", help="Regenerate accounting from existing OOF rows without fitting forecasters."); args = parser.parse_args()
    if not FORECASTING_DATASET.exists() or not OUTCOMES.exists(): raise RuntimeError("Run 05 and 06 first; frozen outcomes and joined forecasting data are required.")
    frame = pd.read_csv(FORECASTING_DATASET)
    if len(frame) != 400 or frame.duplicated(["edit_id", "candidate"]).any(): raise ValueError("expected the frozen 400-row joined dataset")
    if args.reuse_oof:
        if not ROWS_OUTPUT.exists(): raise RuntimeError("No existing OOF selection CSV to reuse.")
        output = pd.read_csv(ROWS_OUTPUT)
    else:
        groups = feature_groups(frame); assignments = fold_assignments(frame); selection_rows = []
        for name, features in groups.items(): selection_rows.extend(choose_rows(oof_predictions(frame, features, assignments), name))
        output = pd.DataFrame(selection_rows)
    if len(output) != 300 or output.duplicated(["method", "edit_id"]).any(): raise RuntimeError("expected 300 unique OOF selections")
    ROWS_OUTPUT.parent.mkdir(parents=True, exist_ok=True); output.to_csv(ROWS_OUTPUT, index=False)
    summary = {"purpose": "Acquisition-only retrospective OOF ranking; these are NOT safety-constrained selections because Qwen damage forecasting failed.",
               "frozen_outcome_rows": 400, "oof_selection_rows": len(output), "cv": {"scheme": "GroupKFold by edit_id", "splits": N_SPLITS},
               "methods": {method: {"overall": describe(group.to_dict("records")),
                                     "by_source_subset": {source.upper(): describe(part.to_dict("records")) for source, part in group.groupby("source_subset")}}
                           for method, group in output.groupby("method", sort=False)}}
    SUMMARY_OUTPUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {ROWS_OUTPUT} and {SUMMARY_OUTPUT}")


if __name__ == "__main__": main()
