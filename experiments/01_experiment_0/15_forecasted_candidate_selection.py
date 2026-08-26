"""Experiment 0.8: OOF forecast-guided candidate selection.

Predictions for an episode are always produced by models that were trained on
other episodes. This keeps policy evaluation separate from outcomes used to
fit the forecasters.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_FILE = Path("results/experiment_0/forecasting_dataset_320.csv")
OUTPUT_FILE = Path("results/experiment_0/forecasted_candidate_selection.csv")
DAMAGE_CAP = 0.01
N_SPLITS = 5
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
PRIMARY_STATIC_BASELINE = "late_lr_1e-5"
STRICT_SAFE_STATIC_BASELINE = "late_lr_5e-6"


def make_model(X: pd.DataFrame) -> Pipeline:
    """Build the same RF preprocessing/model family as Experiment 0.7."""
    categorical = [column for column in FEATURES if is_string_dtype(X[column])]
    numeric = [column for column in FEATURES if column not in categorical]
    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", "passthrough", numeric),
    ])
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ])


def add_oof_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Fit separate gain/damage models and attach held-out-only predictions."""
    result = df.copy()
    result["fold"] = -1
    result["predicted_acquisition_gain"] = np.nan
    result["predicted_mean_damage"] = np.nan
    splitter = GroupKFold(n_splits=N_SPLITS)
    for fold, (train_index, test_index) in enumerate(splitter.split(
        result[FEATURES], result["acquisition_gain"], groups=result["episode_id"],
    )):
        X_train = result.iloc[train_index][FEATURES]
        X_test = result.iloc[test_index][FEATURES]
        gain_model = make_model(X_train)
        damage_model = make_model(X_train)
        gain_model.fit(X_train, result.iloc[train_index]["acquisition_gain"])
        damage_model.fit(X_train, result.iloc[train_index]["mean_damage"])
        result.loc[result.index[test_index], "fold"] = fold
        result.loc[result.index[test_index], "predicted_acquisition_gain"] = gain_model.predict(X_test)
        result.loc[result.index[test_index], "predicted_mean_damage"] = damage_model.predict(X_test)
    if result[["predicted_acquisition_gain", "predicted_mean_damage"]].isna().any().any():
        raise RuntimeError("OOF prediction failed for at least one candidate row.")
    return result


def select_forecast_policy(episode: pd.DataFrame) -> pd.DataFrame:
    """Select max forecast gain under the forecast damage cap.

    With no forecast-safe candidate, fall back to minimum forecast damage;
    forecast gain then candidate name break ties deterministically.
    """
    safe = episode[episode["predicted_mean_damage"] <= DAMAGE_CAP]
    if safe.empty:
        chosen = episode.sort_values(
            ["predicted_mean_damage", "predicted_acquisition_gain", "candidate"],
            ascending=[True, False, True],
        ).iloc[0]
        reason = "fallback_lowest_predicted_damage"
    else:
        chosen = safe.sort_values(
            ["predicted_acquisition_gain", "predicted_mean_damage", "candidate"],
            ascending=[False, True, True],
        ).iloc[0]
        reason = "highest_predicted_gain_among_predicted_safe"
    result = episode.copy()
    result["forecast_selected"] = result.index == chosen.name
    result["forecast_selection_reason"] = ""
    result.loc[chosen.name, "forecast_selection_reason"] = reason
    return result


def select_oracle(episode: pd.DataFrame) -> pd.DataFrame:
    """Select best realized gain among actually cap-compliant candidates."""
    safe = episode[episode["mean_damage"] <= DAMAGE_CAP]
    if safe.empty:
        chosen = episode.sort_values(
            ["mean_damage", "acquisition_gain", "candidate"], ascending=[True, False, True],
        ).iloc[0]
        reason = "fallback_lowest_actual_damage"
    else:
        chosen = safe.sort_values(
            ["acquisition_gain", "mean_damage", "candidate"], ascending=[False, True, True],
        ).iloc[0]
        reason = "highest_actual_gain_among_actual_safe"
    result = episode.copy()
    result["oracle_selected"] = result.index == chosen.name
    result["oracle_selection_reason"] = ""
    result.loc[chosen.name, "oracle_selection_reason"] = reason
    return result


def policy_summary(name: str, selected: pd.DataFrame, oracle_gain: float) -> dict:
    mean_gain = selected["acquisition_gain"].mean()
    violations = int((selected["mean_damage"] > DAMAGE_CAP).sum())
    return {
        "policy": name,
        "episodes": len(selected),
        "mean_actual_gain": mean_gain,
        "mean_actual_damage": selected["mean_damage"].mean(),
        "hard_cap_violations": violations,
        "hard_cap_violation_rate": violations / len(selected),
        "oracle_gap_closed_vs_late_lr_1e-5": np.nan,
        "oracle_mean_actual_gain": oracle_gain,
    }


def main() -> None:
    df = pd.read_csv(DATA_FILE)
    required = set(FEATURES + ["episode_id", "acquisition_gain", "mean_damage"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    scored = add_oof_predictions(df)
    scored = scored.groupby("episode_id", group_keys=False).apply(select_forecast_policy)
    scored = scored.groupby("episode_id", group_keys=False).apply(select_oracle)
    scored["is_primary_static_baseline"] = scored["candidate"] == PRIMARY_STATIC_BASELINE
    scored["is_strict_safe_static_baseline"] = scored["candidate"] == STRICT_SAFE_STATIC_BASELINE

    forecast = scored[scored["forecast_selected"]].copy()
    oracle = scored[scored["oracle_selected"]].copy()
    static = scored[scored["is_primary_static_baseline"]].copy()
    strict_static = scored[scored["is_strict_safe_static_baseline"]].copy()
    episode_count = df["episode_id"].nunique()
    if not all(len(frame) == episode_count for frame in [forecast, oracle, static, strict_static]):
        raise RuntimeError("A policy did not select exactly one candidate per episode.")

    oracle_gain = oracle["acquisition_gain"].mean()
    summaries = [
        policy_summary("forecast_selected", forecast, oracle_gain),
        policy_summary(PRIMARY_STATIC_BASELINE, static, oracle_gain),
        policy_summary(STRICT_SAFE_STATIC_BASELINE, strict_static, oracle_gain),
        policy_summary("constrained_oracle", oracle, oracle_gain),
    ]
    baseline_gain = static["acquisition_gain"].mean()
    denominator = oracle_gain - baseline_gain
    for summary in summaries:
        summary["oracle_gap_closed_vs_late_lr_1e-5"] = (
            (summary["mean_actual_gain"] - baseline_gain) / denominator
            if not np.isclose(denominator, 0) else np.nan
        )
    summary_df = pd.DataFrame(summaries)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(OUTPUT_FILE, index=False)

    print("\n=== EXPERIMENT 0.8: FORECAST-GUIDED CANDIDATE SELECTION ===")
    print("OOF evaluation: 5-fold GroupKFold by episode")
    print(f"Damage cap: {DAMAGE_CAP:.4f}")
    print("Fallback: lowest predicted damage, then highest predicted gain")
    print("\n=== ACTUAL REALIZED POLICY PERFORMANCE ===")
    print(summary_df.drop(columns="oracle_mean_actual_gain").to_string(
        index=False, float_format=lambda value: f"{value:.4f}",
    ))
    print("\n=== FORECAST SELECTION COUNTS ===")
    print(forecast["candidate"].value_counts().sort_index().to_string())
    print("\n=== FORECAST FALLBACKS ===")
    print((forecast["forecast_selection_reason"] == "fallback_lowest_predicted_damage").sum())
    print(f"\nSaved per-candidate OOF predictions and selections: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
