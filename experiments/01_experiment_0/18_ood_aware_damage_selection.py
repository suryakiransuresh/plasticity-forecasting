"""Experiment 0.9b: OOD-gated safety for leave-one-family-out selection.

For every held-out task family, fit gain/damage forecasters on the other three
families.  An episode is flagged OOD when its *pre-update fingerprint* has a
shrinkage-Mahalanobis distance above the training-fingerprint 95th percentile.
OOD episodes use ``late_lr_5e-6``; all other episodes use the calibrated
forecast policy (highest predicted gain subject to predicted damage <= 0.008).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from sklearn.compose import ColumnTransformer
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_FILE = Path("results/experiment_0/forecasting_dataset_320.csv")
ROW_OUTPUT_FILE = Path("results/experiment_0/ood_aware_damage_selection.csv")
SUMMARY_OUTPUT_FILE = Path("results/experiment_0/ood_aware_damage_selection_summary.csv")
FAMILIES = ("facts", "symbols", "rules", "arithmetic")
PREDICTED_DAMAGE_THRESHOLD = 0.008
DAMAGE_CAP = 0.010
OOD_QUANTILE = 0.95
PRIMARY_STATIC_BASELINE = "late_lr_1e-5"
STRICT_SAFE_STATIC_BASELINE = "late_lr_5e-6"
RANDOM_STATE = 42

CANDIDATE_FEATURES = ["candidate", "mode", "learning_rate"]
FINGERPRINT_FEATURES = [
    "novel_loss", "total_grad_norm", "early_grad_norm", "middle_grad_norm",
    "late_grad_norm", "early_grad_fraction", "middle_grad_fraction",
    "late_grad_fraction", "peak_layer", "layer_concentration",
    "mean_protected_cosine", "min_protected_cosine", "max_protected_cosine",
    "same_family_cosine", "off_family_cosine", "cosine_to_facts",
    "cosine_to_symbols", "cosine_to_rules", "cosine_to_arithmetic",
]
FEATURES = FINGERPRINT_FEATURES + CANDIDATE_FEATURES


def make_model(X: pd.DataFrame) -> Pipeline:
    """Build the RF forecaster used in Experiment 0.9."""
    categorical = [column for column in FEATURES if is_string_dtype(X[column])]
    numeric = [column for column in FEATURES if column not in categorical]
    return Pipeline([
        ("preprocess", ColumnTransformer([
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", "passthrough", numeric),
        ])),
        ("model", RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ])


def choose_forecast_candidate(episode: pd.DataFrame) -> tuple[pd.Series, str]:
    """Choose highest predicted gain below the calibrated damage threshold."""
    qualifying = episode[episode["predicted_mean_damage"] <= PREDICTED_DAMAGE_THRESHOLD]
    if qualifying.empty:
        return (
            episode.sort_values(
                ["predicted_mean_damage", "predicted_acquisition_gain", "candidate"],
                ascending=[True, False, True],
            ).iloc[0],
            "fallback_lowest_predicted_damage",
        )
    return (
        qualifying.sort_values(
            ["predicted_acquisition_gain", "predicted_mean_damage", "candidate"],
            ascending=[False, True, True],
        ).iloc[0],
        "highest_predicted_gain_among_predicted_safe",
    )


def choose_oracle_candidate(episode: pd.DataFrame) -> tuple[pd.Series, str]:
    """Choose the highest actual gain subject to the actual damage cap."""
    qualifying = episode[episode["mean_damage"] <= DAMAGE_CAP]
    if qualifying.empty:
        return (
            episode.sort_values(["mean_damage", "acquisition_gain", "candidate"],
                                ascending=[True, False, True]).iloc[0],
            "fallback_lowest_actual_damage",
        )
    return (
        qualifying.sort_values(["acquisition_gain", "mean_damage", "candidate"],
                                ascending=[False, True, True]).iloc[0],
        "highest_actual_gain_among_actual_safe",
    )


def mark_selection(frame: pd.DataFrame, chooser, flag: str, reason: str) -> pd.DataFrame:
    """Mark exactly one candidate per episode for a policy."""
    result = frame.copy()
    result[flag] = False
    result[reason] = ""
    for _, episode in result.groupby("episode_id", sort=False):
        chosen, selection_reason = chooser(episode)
        result.loc[chosen.name, flag] = True
        result.loc[chosen.name, reason] = selection_reason
    return result


def add_ood_scores(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Add episode-level shrinkage Mahalanobis OOD scores using fingerprints only."""
    train_episodes = train.drop_duplicates("episode_id").set_index("episode_id")
    test_episodes = test.drop_duplicates("episode_id").set_index("episode_id")
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_episodes[FINGERPRINT_FEATURES])
    test_x = scaler.transform(test_episodes[FINGERPRINT_FEATURES])
    covariance = LedoitWolf().fit(train_x)
    train_scores = covariance.mahalanobis(train_x)
    test_scores = covariance.mahalanobis(test_x)
    threshold = float(np.quantile(train_scores, OOD_QUANTILE))
    score_by_episode = pd.Series(test_scores, index=test_episodes.index)
    result = test.copy()
    result["fingerprint_ood_score"] = result["episode_id"].map(score_by_episode)
    result["fingerprint_ood_threshold"] = threshold
    result["fingerprint_ood_flag"] = result["fingerprint_ood_score"] > threshold
    return result, threshold


def choose_ood_aware_candidate(episode: pd.DataFrame) -> tuple[pd.Series, str]:
    """Use strict-safe fallback for OOD episodes, otherwise forecast selection."""
    if bool(episode["fingerprint_ood_flag"].iloc[0]):
        fallback = episode[episode["candidate"] == STRICT_SAFE_STATIC_BASELINE]
        if fallback.empty:
            raise ValueError(f"Missing OOD fallback candidate {STRICT_SAFE_STATIC_BASELINE}")
        return fallback.iloc[0], "ood_strict_safe_fallback"
    chosen, reason = choose_forecast_candidate(episode)
    return chosen, f"in_distribution_{reason}"


def metric_row(scope: str, family: str, scored: pd.DataFrame, policy: str) -> dict:
    """Summarize realized policy outcomes, OOD coverage, and forecast quality."""
    flag = f"{policy}_selected"
    selected = scored[scored[flag]]
    static = scored[scored["is_primary_static_baseline"]]
    oracle = scored[scored["oracle_selected"]]
    if not (len(selected) == len(static) == len(oracle)):
        raise RuntimeError(f"Expected one selection per episode for {scope}/{policy}.")
    baseline_gain, oracle_gain = static["acquisition_gain"].mean(), oracle["acquisition_gain"].mean()
    selected_gain = selected["acquisition_gain"].mean()
    oracle_gap = oracle_gain - baseline_gain
    violations = int((selected["mean_damage"] > DAMAGE_CAP).sum())
    arithmetic = selected[selected["novel_family"] == "arithmetic"]
    row = {
        "scope": scope, "heldout_family": family, "policy": policy,
        "episodes": len(selected), "mean_realized_gain": selected_gain,
        "mean_realized_damage": selected["mean_damage"].mean(),
        "hard_cap_violations": violations,
        "hard_cap_violation_rate": violations / len(selected),
        "arithmetic_episodes": len(arithmetic),
        "arithmetic_hard_cap_violations": int((arithmetic["mean_damage"] > DAMAGE_CAP).sum()),
        "arithmetic_violation_rate": (
            float((arithmetic["mean_damage"] > DAMAGE_CAP).mean()) if len(arithmetic) else np.nan
        ),
        "oracle_gap_closed_vs_late_lr_1e-5": (
            (selected_gain - baseline_gain) / oracle_gap if not np.isclose(oracle_gap, 0) else np.nan
        ),
        "ood_flag_count": int(selected["fingerprint_ood_flag"].sum()),
        "ood_flag_rate": selected["fingerprint_ood_flag"].mean(),
        "mean_fingerprint_ood_score": selected["fingerprint_ood_score"].mean(),
        "mean_fingerprint_ood_threshold": selected["fingerprint_ood_threshold"].mean(),
        "late_lr_1e-5_mean_realized_gain": baseline_gain,
        "late_lr_1e-5_mean_realized_damage": static["mean_damage"].mean(),
        "oracle_mean_realized_gain": oracle_gain,
        "oracle_mean_realized_damage": oracle["mean_damage"].mean(),
        "gain_mae": mean_absolute_error(scored["acquisition_gain"], scored["predicted_acquisition_gain"]),
        "gain_r2": r2_score(scored["acquisition_gain"], scored["predicted_acquisition_gain"]),
        "damage_mae": mean_absolute_error(scored["mean_damage"], scored["predicted_mean_damage"]),
        "damage_r2": r2_score(scored["mean_damage"], scored["predicted_mean_damage"]),
    }
    for candidate in sorted(scored["candidate"].unique()):
        row[f"selected_count_{candidate}"] = int((selected["candidate"] == candidate).sum())
    return row


def main() -> None:
    df = pd.read_csv(DATA_FILE)
    required = set(FEATURES + ["episode_id", "novel_family", "acquisition_gain", "mean_damage"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if set(df["novel_family"].unique()) != set(FAMILIES):
        raise ValueError(f"Expected families {FAMILIES}; found {sorted(df['novel_family'].unique())}")

    all_scored, summary_rows = [], []
    for heldout_family in FAMILIES:
        train = df[df["novel_family"] != heldout_family]
        test = df[df["novel_family"] == heldout_family].copy()
        gain_model, damage_model = make_model(train[FEATURES]), make_model(train[FEATURES])
        gain_model.fit(train[FEATURES], train["acquisition_gain"])
        damage_model.fit(train[FEATURES], train["mean_damage"])
        test["heldout_family"] = heldout_family
        test["predicted_acquisition_gain"] = gain_model.predict(test[FEATURES])
        test["predicted_mean_damage"] = damage_model.predict(test[FEATURES])
        test, _ = add_ood_scores(train, test)
        test = mark_selection(test, choose_forecast_candidate, "forecast_selected", "forecast_selection_reason")
        test = mark_selection(test, choose_ood_aware_candidate, "ood_aware_selected", "ood_aware_selection_reason")
        test = mark_selection(test, choose_oracle_candidate, "oracle_selected", "oracle_selection_reason")
        test["late_lr_1e-5_selected"] = test["candidate"] == PRIMARY_STATIC_BASELINE
        test["late_lr_5e-6_selected"] = test["candidate"] == STRICT_SAFE_STATIC_BASELINE
        test["is_primary_static_baseline"] = test["late_lr_1e-5_selected"]
        all_scored.append(test)
        for policy in ("forecast", "ood_aware", "late_lr_1e-5", "late_lr_5e-6", "oracle"):
            summary_rows.append(metric_row(heldout_family, heldout_family, test, policy))

    scored = pd.concat(all_scored, ignore_index=True)
    for policy in ("forecast", "ood_aware", "late_lr_1e-5", "late_lr_5e-6", "oracle"):
        summary_rows.append(metric_row("overall", "all", scored, policy))
    summary = pd.DataFrame(summary_rows)
    ROW_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(ROW_OUTPUT_FILE, index=False)
    summary.to_csv(SUMMARY_OUTPUT_FILE, index=False)

    display = ["scope", "policy", "episodes", "mean_realized_gain", "mean_realized_damage",
               "hard_cap_violations", "hard_cap_violation_rate", "arithmetic_violation_rate",
               "oracle_gap_closed_vs_late_lr_1e-5", "ood_flag_count", "ood_flag_rate"]
    print("\n=== EXPERIMENT 0.9b: OOD-AWARE DAMAGE SELECTION ===")
    print("OOD score: Ledoit-Wolf shrinkage Mahalanobis distance on pre-update fingerprints.")
    print(f"OOD threshold: training-distribution {OOD_QUANTILE:.0%} percentile; fallback: {STRICT_SAFE_STATIC_BASELINE}")
    print(f"Forecast policy: predicted damage <= {PREDICTED_DAMAGE_THRESHOLD:.3f}; actual cap: {DAMAGE_CAP:.3f}")
    print("\n=== SUMMARY ===")
    print(summary[display].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\n=== CANDIDATE SELECTION COUNTS ===")
    counts = ["scope", "policy"] + [column for column in summary if column.startswith("selected_count_")]
    print(summary[counts].to_string(index=False))
    print(f"\nSaved row-level results: {ROW_OUTPUT_FILE}")
    print(f"Saved compact summary: {SUMMARY_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
