"""Experiment 0.9: rotating held-out-family forecast-guided selection.

For each novel task family, fit separate gain and damage Random Forest models
on the other three families, then select one of the eight candidate updates for
every held-out episode using the calibrated predicted-damage threshold (0.008).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_FILE = Path("results/experiment_0/forecasting_dataset_320.csv")
ROW_OUTPUT_FILE = Path("results/experiment_0/heldout_family_generalization.csv")
SUMMARY_OUTPUT_FILE = Path(
    "results/experiment_0/heldout_family_generalization_summary.csv"
)
FAMILIES = ("facts", "symbols", "rules", "arithmetic")
PREDICTED_DAMAGE_THRESHOLD = 0.008
DAMAGE_CAP = 0.010
PRIMARY_STATIC_BASELINE = "late_lr_1e-5"
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
    """Build the Experiment 0.8 RF model with robust categorical handling."""
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
    """Choose highest predicted gain below threshold; otherwise safest forecast."""
    qualifying = episode[
        episode["predicted_mean_damage"] <= PREDICTED_DAMAGE_THRESHOLD
    ]
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
    """Choose the best realized gain under the actual hard damage cap."""
    qualifying = episode[episode["mean_damage"] <= DAMAGE_CAP]
    if qualifying.empty:
        return (
            episode.sort_values(
                ["mean_damage", "acquisition_gain", "candidate"],
                ascending=[True, False, True],
            ).iloc[0],
            "fallback_lowest_actual_damage",
        )
    return (
        qualifying.sort_values(
            ["acquisition_gain", "mean_damage", "candidate"],
            ascending=[False, True, True],
        ).iloc[0],
        "highest_actual_gain_among_actual_safe",
    )


def select_one_per_episode(frame: pd.DataFrame, chooser, flag: str, reason: str) -> pd.DataFrame:
    """Attach a policy-selection flag and reason to every candidate row."""
    result = frame.copy()
    result[flag] = False
    result[reason] = ""
    for _, episode in result.groupby("episode_id", sort=False):
        chosen, selection_reason = chooser(episode)
        result.loc[chosen.name, flag] = True
        result.loc[chosen.name, reason] = selection_reason
    return result


def metric_row(scope: str, family: str, scored: pd.DataFrame) -> dict:
    """Summarize selection and held-out regression performance for one scope."""
    forecast = scored[scored["forecast_selected"]]
    static = scored[scored["is_primary_static_baseline"]]
    oracle = scored[scored["oracle_selected"]]
    if not (len(forecast) == len(static) == len(oracle)):
        raise RuntimeError(f"Expected exactly one selection per episode for {scope}.")

    baseline_gain = static["acquisition_gain"].mean()
    oracle_gain = oracle["acquisition_gain"].mean()
    selected_gain = forecast["acquisition_gain"].mean()
    oracle_gap = oracle_gain - baseline_gain
    violations = int((forecast["mean_damage"] > DAMAGE_CAP).sum())
    row = {
        "scope": scope,
        "heldout_family": family,
        "episodes": len(forecast),
        "candidate_rows": len(scored),
        "mean_realized_gain": selected_gain,
        "mean_realized_damage": forecast["mean_damage"].mean(),
        "hard_cap_violations": violations,
        "hard_cap_violation_rate": violations / len(forecast),
        "oracle_gap_closed_vs_late_lr_1e-5": (
            (selected_gain - baseline_gain) / oracle_gap
            if not np.isclose(oracle_gap, 0) else np.nan
        ),
        "late_lr_1e-5_mean_realized_gain": baseline_gain,
        "late_lr_1e-5_mean_realized_damage": static["mean_damage"].mean(),
        "late_lr_1e-5_hard_cap_violations": int((static["mean_damage"] > DAMAGE_CAP).sum()),
        "oracle_mean_realized_gain": oracle_gain,
        "oracle_mean_realized_damage": oracle["mean_damage"].mean(),
        "oracle_hard_cap_violations": int((oracle["mean_damage"] > DAMAGE_CAP).sum()),
        "gain_mae": mean_absolute_error(scored["acquisition_gain"], scored["predicted_acquisition_gain"]),
        "gain_r2": r2_score(scored["acquisition_gain"], scored["predicted_acquisition_gain"]),
        "damage_mae": mean_absolute_error(scored["mean_damage"], scored["predicted_mean_damage"]),
        "damage_r2": r2_score(scored["mean_damage"], scored["predicted_mean_damage"]),
        "fallback_count": int((forecast["forecast_selection_reason"] == "fallback_lowest_predicted_damage").sum()),
    }
    for candidate in sorted(scored["candidate"].unique()):
        row[f"selected_count_{candidate}"] = int((forecast["candidate"] == candidate).sum())
    return row


def main() -> None:
    df = pd.read_csv(DATA_FILE)
    required = set(FEATURES + ["episode_id", "novel_family", "acquisition_gain", "mean_damage"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    observed_families = set(df["novel_family"].unique())
    if set(FAMILIES) != observed_families:
        raise ValueError(f"Expected families {FAMILIES}; found {sorted(observed_families)}")

    all_scored = []
    summary_rows = []
    for heldout_family in FAMILIES:
        train = df[df["novel_family"] != heldout_family]
        test = df[df["novel_family"] == heldout_family].copy()
        gain_model = make_model(train[FEATURES])
        damage_model = make_model(train[FEATURES])
        gain_model.fit(train[FEATURES], train["acquisition_gain"])
        damage_model.fit(train[FEATURES], train["mean_damage"])
        test["heldout_family"] = heldout_family
        test["predicted_acquisition_gain"] = gain_model.predict(test[FEATURES])
        test["predicted_mean_damage"] = damage_model.predict(test[FEATURES])
        test = select_one_per_episode(
            test, choose_forecast_candidate, "forecast_selected", "forecast_selection_reason",
        )
        test = select_one_per_episode(
            test, choose_oracle_candidate, "oracle_selected", "oracle_selection_reason",
        )
        test["is_primary_static_baseline"] = test["candidate"] == PRIMARY_STATIC_BASELINE
        all_scored.append(test)
        summary_rows.append(metric_row(heldout_family, heldout_family, test))

    scored = pd.concat(all_scored, ignore_index=True)
    summary_rows.append(metric_row("overall", "all", scored))
    summary = pd.DataFrame(summary_rows)
    ROW_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(ROW_OUTPUT_FILE, index=False)
    summary.to_csv(SUMMARY_OUTPUT_FILE, index=False)

    display_columns = [
        "scope", "episodes", "mean_realized_gain", "mean_realized_damage",
        "hard_cap_violations", "hard_cap_violation_rate",
        "oracle_gap_closed_vs_late_lr_1e-5", "gain_mae", "gain_r2",
        "damage_mae", "damage_r2", "fallback_count",
    ]
    print("\n=== EXPERIMENT 0.9: HELD-OUT-FAMILY GENERALIZATION ===")
    print("Train on three families; predict and select on the fourth.")
    print(f"Predicted-damage threshold: {PREDICTED_DAMAGE_THRESHOLD:.3f}")
    print(f"Actual hard damage cap: {DAMAGE_CAP:.3f}")
    print("Fallback: lowest predicted damage, then highest predicted gain")
    print("\n=== SUMMARY ===")
    print(summary[display_columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\n=== FORECAST CANDIDATE SELECTION COUNTS ===")
    count_columns = ["scope"] + [column for column in summary if column.startswith("selected_count_")]
    print(summary[count_columns].to_string(index=False))
    print(f"\nSaved row-level predictions/results: {ROW_OUTPUT_FILE}")
    print(f"Saved compact summary: {SUMMARY_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
