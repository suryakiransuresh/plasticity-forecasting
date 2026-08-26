"""Experiment 0.9c: uncertainty-aware damage selection.

For each leave-one-family-out split, Random Forest gain and damage forecasters
are trained on the other three task families.  Damage risk is estimated from
the distribution of predictions across the fitted damage forest's trees:

    D_upper = mean_tree_damage_prediction + lambda * std_tree_prediction

For every lambda, the policy selects the highest predicted-gain candidate with
``D_upper <= 0.008``.  If none qualifies, it uses the fixed strict-safe
candidate ``late_lr_5e-6``.
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
ROW_OUTPUT_FILE = Path("results/experiment_0/uncertainty_aware_damage_selection.csv")
SUMMARY_OUTPUT_FILE = Path(
    "results/experiment_0/uncertainty_aware_damage_selection_summary.csv"
)
FAMILIES = ("facts", "symbols", "rules", "arithmetic")
LAMBDAS = (0.0, 0.5, 1.0, 1.5, 2.0)
UPPER_DAMAGE_THRESHOLD = 0.008
DAMAGE_CAP = 0.010
PRIMARY_STATIC_BASELINE = "late_lr_1e-5"
STRICT_SAFE_FALLBACK = "late_lr_5e-6"
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


def lambda_label(value: float) -> str:
    """Return a stable, CSV-friendly label for a lambda value."""
    return f"lambda_{value:.1f}".replace(".", "_")


def make_model(X: pd.DataFrame) -> Pipeline:
    """Build the RF forecaster used by the preceding selection experiments."""
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


def tree_prediction_distribution(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return one damage prediction per fitted tree for every candidate row."""
    transformed = model.named_steps["preprocess"].transform(X)
    forest = model.named_steps["model"]
    return np.column_stack([tree.predict(transformed) for tree in forest.estimators_])


def choose_uncertainty_aware_candidate(
    episode: pd.DataFrame, lambda_value: float
) -> tuple[pd.Series, str]:
    """Select max predicted gain subject to the upper-risk constraint."""
    upper_column = f"damage_upper_{lambda_label(lambda_value)}"
    qualifying = episode[episode[upper_column] <= UPPER_DAMAGE_THRESHOLD]
    if qualifying.empty:
        fallback = episode[episode["candidate"] == STRICT_SAFE_FALLBACK]
        if fallback.empty:
            raise ValueError(
                f"Episode {episode['episode_id'].iloc[0]} lacks {STRICT_SAFE_FALLBACK}."
            )
        return fallback.iloc[0], "strict_safe_fallback_no_candidate_meets_upper_risk"
    return (
        qualifying.sort_values(
            ["predicted_acquisition_gain", upper_column, "candidate"],
            ascending=[False, True, True],
        ).iloc[0],
        "highest_predicted_gain_among_upper_risk_safe",
    )


def choose_oracle_candidate(episode: pd.DataFrame) -> tuple[pd.Series, str]:
    """Choose the highest realized gain under the actual hard damage cap."""
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


def mark_selection(
    frame: pd.DataFrame, chooser, flag: str, reason: str
) -> pd.DataFrame:
    """Mark exactly one selected candidate and its selection reason per episode."""
    result = frame.copy()
    result[flag] = False
    result[reason] = ""
    for _, episode in result.groupby("episode_id", sort=False):
        chosen, selection_reason = chooser(episode)
        result.loc[chosen.name, flag] = True
        result.loc[chosen.name, reason] = selection_reason
    return result


def metric_row(scope: str, family: str, scored: pd.DataFrame, lambda_value: float) -> dict:
    """Summarize policy quality, safety, and selection distribution for one scope."""
    label = lambda_label(lambda_value)
    selected = scored[scored[f"{label}_selected"]]
    static = scored[scored["is_primary_static_baseline"]]
    oracle = scored[scored["oracle_selected"]]
    if not (len(selected) == len(static) == len(oracle)):
        raise RuntimeError(f"Expected one selection per episode for {scope}/{label}.")

    baseline_gain = static["acquisition_gain"].mean()
    oracle_gain = oracle["acquisition_gain"].mean()
    selected_gain = selected["acquisition_gain"].mean()
    oracle_gap = oracle_gain - baseline_gain
    arithmetic = selected[selected["novel_family"] == "arithmetic"]
    violations = selected["mean_damage"] > DAMAGE_CAP
    arithmetic_violations = arithmetic["mean_damage"] > DAMAGE_CAP
    row = {
        "scope": scope,
        "heldout_family": family,
        "lambda": lambda_value,
        "episodes": len(selected),
        "mean_realized_gain": selected_gain,
        "mean_realized_damage": selected["mean_damage"].mean(),
        "hard_cap_violations": int(violations.sum()),
        "hard_cap_violation_rate": float(violations.mean()),
        "arithmetic_episodes": len(arithmetic),
        "arithmetic_hard_cap_violations": int(arithmetic_violations.sum()),
        "arithmetic_hard_cap_violation_rate": (
            float(arithmetic_violations.mean()) if len(arithmetic) else np.nan
        ),
        "oracle_gap_closed_vs_late_lr_1e-5": (
            (selected_gain - baseline_gain) / oracle_gap
            if not np.isclose(oracle_gap, 0) else np.nan
        ),
        "fallback_count": int((selected[f"{label}_selection_reason"] ==
                               "strict_safe_fallback_no_candidate_meets_upper_risk").sum()),
        "mean_predicted_damage": selected["predicted_mean_damage"].mean(),
        "mean_tree_damage_std": selected["tree_damage_prediction_std"].mean(),
        "mean_upper_risk": selected[f"damage_upper_{label}"].mean(),
        "late_lr_1e-5_mean_realized_gain": baseline_gain,
        "late_lr_1e-5_mean_realized_damage": static["mean_damage"].mean(),
        "late_lr_1e-5_hard_cap_violations": int((static["mean_damage"] > DAMAGE_CAP).sum()),
        "oracle_mean_realized_gain": oracle_gain,
        "oracle_mean_realized_damage": oracle["mean_damage"].mean(),
        "oracle_hard_cap_violations": int((oracle["mean_damage"] > DAMAGE_CAP).sum()),
        "gain_mae": mean_absolute_error(
            scored["acquisition_gain"], scored["predicted_acquisition_gain"]
        ),
        "gain_r2": r2_score(scored["acquisition_gain"], scored["predicted_acquisition_gain"]),
        "damage_mae": mean_absolute_error(
            scored["mean_damage"], scored["predicted_mean_damage"]
        ),
        "damage_r2": r2_score(scored["mean_damage"], scored["predicted_mean_damage"]),
    }
    for candidate in sorted(scored["candidate"].unique()):
        row[f"selected_count_{candidate}"] = int((selected["candidate"] == candidate).sum())
    return row


def main() -> None:
    df = pd.read_csv(DATA_FILE)
    required = set(FEATURES + [
        "episode_id", "novel_family", "acquisition_gain", "mean_damage",
    ])
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

        tree_predictions = tree_prediction_distribution(damage_model, test[FEATURES])
        test["heldout_family"] = heldout_family
        test["predicted_acquisition_gain"] = gain_model.predict(test[FEATURES])
        test["predicted_mean_damage"] = tree_predictions.mean(axis=1)
        test["tree_damage_prediction_std"] = tree_predictions.std(axis=1)
        for lambda_value in LAMBDAS:
            label = lambda_label(lambda_value)
            test[f"damage_upper_{label}"] = (
                test["predicted_mean_damage"]
                + lambda_value * test["tree_damage_prediction_std"]
            )
            test = mark_selection(
                test,
                lambda episode, value=lambda_value: choose_uncertainty_aware_candidate(episode, value),
                f"{label}_selected",
                f"{label}_selection_reason",
            )
        test = mark_selection(test, choose_oracle_candidate, "oracle_selected", "oracle_selection_reason")
        test["is_primary_static_baseline"] = test["candidate"] == PRIMARY_STATIC_BASELINE
        all_scored.append(test)
        for lambda_value in LAMBDAS:
            summary_rows.append(metric_row(heldout_family, heldout_family, test, lambda_value))

    scored = pd.concat(all_scored, ignore_index=True)
    for lambda_value in LAMBDAS:
        summary_rows.append(metric_row("overall", "all", scored, lambda_value))
    summary = pd.DataFrame(summary_rows)
    ROW_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(ROW_OUTPUT_FILE, index=False)
    summary.to_csv(SUMMARY_OUTPUT_FILE, index=False)

    display = [
        "scope", "lambda", "episodes", "mean_realized_gain", "mean_realized_damage",
        "hard_cap_violations", "hard_cap_violation_rate",
        "arithmetic_hard_cap_violations", "arithmetic_hard_cap_violation_rate",
        "oracle_gap_closed_vs_late_lr_1e-5", "fallback_count",
    ]
    print("\n=== EXPERIMENT 0.9c: UNCERTAINTY-AWARE DAMAGE SELECTION ===")
    print("Train on three families; select candidates on the held-out fourth family.")
    print("Upper-risk rule: predicted damage + lambda * per-tree damage std <= "
          f"{UPPER_DAMAGE_THRESHOLD:.3f}")
    print(f"Fallback when none qualify: {STRICT_SAFE_FALLBACK}")
    print(f"Actual violation cap: mean damage > {DAMAGE_CAP:.3f}")
    print("\n=== SUMMARY ===")
    print(summary[display].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\n=== CANDIDATE SELECTION COUNTS ===")
    count_columns = ["scope", "lambda"] + [
        column for column in summary if column.startswith("selected_count_")
    ]
    print(summary[count_columns].to_string(index=False))
    print(f"\nSaved row-level results: {ROW_OUTPUT_FILE}")
    print(f"Saved compact summary: {SUMMARY_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
