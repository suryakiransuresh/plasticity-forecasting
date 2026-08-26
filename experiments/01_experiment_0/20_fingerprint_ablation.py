"""Experiment 0.10: grouped episode-level fingerprint ablation.

Compare which portions of the pre-update fingerprint help Random Forest
forecasts of acquisition gain and mean protected-memory damage.  Every fold
holds out complete episodes, so the eight candidate updates for an episode
never appear in both training and evaluation data.
"""

from pathlib import Path

import pandas as pd
from pandas.api.types import is_string_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_FILE = Path("results/experiment_0/forecasting_dataset_320.csv")
OUTPUT_FILE = Path("results/experiment_0/fingerprint_ablation_summary.csv")
N_SPLITS = 5
RANDOM_STATE = 42
N_ESTIMATORS = 300
TARGETS = ("acquisition_gain", "mean_damage")

# These are the candidate descriptors available before applying an update.
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


def make_model(features: list[str], X: pd.DataFrame) -> Pipeline:
    """Use the established RF setup while one-hot encoding candidate metadata."""
    categorical = [column for column in features if is_string_dtype(X[column])]
    numeric = [column for column in features if column not in categorical]
    return Pipeline([
        ("preprocess", ColumnTransformer([
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", "passthrough", numeric),
        ])),
        ("model", RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def score_feature_group(
    df: pd.DataFrame, group_name: str, features: list[str], target: str
) -> list[dict]:
    """Return one strictly held-out episode-level score per GroupKFold fold."""
    rows = []
    splitter = GroupKFold(n_splits=N_SPLITS)
    for fold, (train_index, test_index) in enumerate(
        splitter.split(df[features], df[target], groups=df["episode_id"]), start=1
    ):
        train = df.iloc[train_index]
        test = df.iloc[test_index]
        model = make_model(features, train[features])
        model.fit(train[features], train[target])
        prediction = model.predict(test[features])
        rows.append({
            "feature_group": group_name,
            "target": target,
            "fold": fold,
            "test_episodes": test["episode_id"].nunique(),
            "test_rows": len(test),
            "mae": mean_absolute_error(test[target], prediction),
            "r2": r2_score(test[target], prediction),
        })
    return rows


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

    fold_rows = []
    for group_name, features in FEATURE_GROUPS.items():
        for target in TARGETS:
            fold_rows.extend(score_feature_group(df, group_name, features, target))

    fold_scores = pd.DataFrame(fold_rows)
    summary = (
        fold_scores.groupby(["feature_group", "target"], sort=False)
        .agg(
            folds=("fold", "count"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
        )
        .reset_index()
    )
    feature_counts = {name: len(features) for name, features in FEATURE_GROUPS.items()}
    summary.insert(2, "feature_count", summary["feature_group"].map(feature_counts))
    summary = summary[
        ["feature_group", "target", "feature_count", "folds", "mae_mean", "mae_std", "r2_mean", "r2_std"]
    ]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_FILE, index=False)

    display = summary.copy()
    display["MAE (mean +/- std)"] = display.apply(
        lambda row: f"{row.mae_mean:.4f} +/- {row.mae_std:.4f}", axis=1
    )
    display["R2 (mean +/- std)"] = display.apply(
        lambda row: f"{row.r2_mean:.4f} +/- {row.r2_std:.4f}", axis=1
    )
    print("\n=== EXPERIMENT 0.10: FINGERPRINT ABLATION ===")
    print("5-fold GroupKFold by episode; RandomForestRegressor (300 trees).")
    print(
        display[
            ["feature_group", "target", "feature_count", "MAE (mean +/- std)", "R2 (mean +/- std)"]
        ].to_string(index=False)
    )
    print(f"\nSaved compact summary: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
