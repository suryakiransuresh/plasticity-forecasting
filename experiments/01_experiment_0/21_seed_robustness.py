"""Experiment 0.11: repeated grouped-CV robustness for fingerprint forecasts.

Each repeat uses a new shuffled GroupKFold assignment of complete episodes and
a new Random Forest seed.  Metrics are computed once from all out-of-fold
predictions in each repeat, then summarized across repeats; candidate rows from
the same episode are never split between train and test data.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_FILE = Path("results/experiment_0/forecasting_dataset_320.csv")
OUTPUT_FILE = Path("results/experiment_0/seed_robustness_summary.csv")
N_REPEATS = 10
N_SPLITS = 5
N_ESTIMATORS = 200
BASE_SPLIT_SEED = 100
BASE_MODEL_SEED = 1_000
TARGETS = ("acquisition_gain", "mean_damage")

CANDIDATE_FEATURES = ["candidate", "mode", "learning_rate"]
MAGNITUDE_PROFILE_FEATURES = [
    "novel_loss",
    "total_grad_norm",
    *[f"layer_{layer}_grad_norm" for layer in range(12)],
    "early_grad_norm",
    "middle_grad_norm",
    "late_grad_norm",
    "early_grad_fraction",
    "middle_grad_fraction",
    "late_grad_fraction",
    "peak_layer",
    "layer_concentration",
]
ALIGNMENT_FEATURES = [
    "mean_protected_cosine",
    "min_protected_cosine",
    "max_protected_cosine",
    "same_family_cosine",
    "off_family_cosine",
    "cosine_to_facts",
    "cosine_to_symbols",
    "cosine_to_rules",
    "cosine_to_arithmetic",
]
FEATURE_GROUPS = {
    "candidate_only": CANDIDATE_FEATURES,
    "magnitude_profile_plus_candidate": MAGNITUDE_PROFILE_FEATURES + CANDIDATE_FEATURES,
    "alignment_plus_candidate": ALIGNMENT_FEATURES + CANDIDATE_FEATURES,
    "full_fingerprint_plus_candidate": (
        MAGNITUDE_PROFILE_FEATURES + ALIGNMENT_FEATURES + CANDIDATE_FEATURES
    ),
}


def make_model(features: list[str], X: pd.DataFrame, random_state: int) -> Pipeline:
    """Build the established RF model with categorical candidate handling."""
    categorical = [column for column in features if is_string_dtype(X[column])]
    numeric = [column for column in features if column not in categorical]
    return Pipeline([
        ("preprocess", ColumnTransformer([
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", "passthrough", numeric),
        ])),
        ("model", RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])


def score_repeat(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    split_seed: int,
    model_seed: int,
) -> tuple[float, float]:
    """Return one complete out-of-fold MAE/R2 score for a grouped-CV repeat."""
    splitter = GroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=split_seed)
    predictions = np.empty(len(df), dtype=float)
    seen = np.zeros(len(df), dtype=bool)

    for fold, (train_index, test_index) in enumerate(
        splitter.split(df[features], df[target], groups=df["episode_id"])
    ):
        train = df.iloc[train_index]
        test = df.iloc[test_index]
        train_episodes = set(train["episode_id"])
        if train_episodes.intersection(test["episode_id"]):
            raise RuntimeError("Episode leakage detected in grouped split.")
        model = make_model(features, train[features], model_seed + fold)
        model.fit(train[features], train[target])
        predictions[test_index] = model.predict(test[features])
        seen[test_index] = True

    if not seen.all():
        raise RuntimeError("Not all rows received an out-of-fold prediction.")
    return (
        mean_absolute_error(df[target], predictions),
        r2_score(df[target], predictions),
    )


def main() -> None:
    df = pd.read_csv(DATA_FILE)
    required = {"episode_id", *TARGETS}
    for features in FEATURE_GROUPS.values():
        required.update(features)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if df["episode_id"].nunique() < N_SPLITS:
        raise ValueError(f"Need at least {N_SPLITS} unique episodes for GroupKFold.")

    repeat_rows = []
    for repeat in range(N_REPEATS):
        split_seed = BASE_SPLIT_SEED + repeat
        model_seed = BASE_MODEL_SEED + repeat
        for group_name, features in FEATURE_GROUPS.items():
            for target in TARGETS:
                mae, r2 = score_repeat(df, features, target, split_seed, model_seed)
                repeat_rows.append({
                    "feature_group": group_name,
                    "target": target,
                    "repeat": repeat + 1,
                    "split_seed": split_seed,
                    "model_seed": model_seed,
                    "mae": mae,
                    "r2": r2,
                })

    repeat_scores = pd.DataFrame(repeat_rows)
    summary = (
        repeat_scores.groupby(["feature_group", "target"], sort=False)
        .agg(
            repeats=("repeat", "count"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
        )
        .reset_index()
    )
    standard_error = np.sqrt(summary["repeats"])
    summary["mae_ci95_low"] = summary["mae_mean"] - 1.96 * summary["mae_std"] / standard_error
    summary["mae_ci95_high"] = summary["mae_mean"] + 1.96 * summary["mae_std"] / standard_error
    summary["r2_ci95_low"] = summary["r2_mean"] - 1.96 * summary["r2_std"] / standard_error
    summary["r2_ci95_high"] = summary["r2_mean"] + 1.96 * summary["r2_std"] / standard_error
    summary.insert(2, "feature_count", summary["feature_group"].map(
        {name: len(features) for name, features in FEATURE_GROUPS.items()}
    ))
    summary.insert(4, "folds_per_repeat", N_SPLITS)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_FILE, index=False)

    display = summary.copy()
    display["MAE (mean +/- std; 95% CI)"] = display.apply(
        lambda row: (
            f"{row.mae_mean:.4f} +/- {row.mae_std:.4f} "
            f"[{row.mae_ci95_low:.4f}, {row.mae_ci95_high:.4f}]"
        ),
        axis=1,
    )
    display["R2 (mean +/- std; 95% CI)"] = display.apply(
        lambda row: (
            f"{row.r2_mean:.4f} +/- {row.r2_std:.4f} "
            f"[{row.r2_ci95_low:.4f}, {row.r2_ci95_high:.4f}]"
        ),
        axis=1,
    )
    print("\n=== EXPERIMENT 0.11: SEED ROBUSTNESS ===")
    print(
        f"{N_REPEATS} shuffled 5-fold GroupKFold repeats by episode; "
        f"RandomForestRegressor ({N_ESTIMATORS} trees)."
    )
    print(display[
        ["feature_group", "target", "feature_count", "MAE (mean +/- std; 95% CI)", "R2 (mean +/- std; 95% CI)"]
    ].to_string(index=False))
    print(f"\nSaved compact summary: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
