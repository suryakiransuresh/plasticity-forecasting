import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from pandas.api.types import is_string_dtype


# ============================================================
# LOAD DATA
# ============================================================

DATA_FILE = (
    "results/experiment_0/"
    "forecasting_dataset_320.csv"
)

df = pd.read_csv(DATA_FILE)

print("\n=== DATASET ===")
print(f"Rows:     {len(df)}")
print(f"Episodes: {df['episode_id'].nunique()}")
print(f"Candidates per episode: {df.groupby('episode_id')['candidate'].nunique().min()}")


# ============================================================
# TARGETS
# ============================================================

targets = [
    "acquisition_gain",
    "mean_damage",
]


# ============================================================
# FEATURE SETS
# ============================================================

candidate_features = [
    "candidate",
    "mode",
    "learning_rate",
]

fingerprint_features = [
    "novel_loss",
    "total_grad_norm",
    "early_grad_norm",
    "middle_grad_norm",
    "late_grad_norm",
    "early_grad_fraction",
    "middle_grad_fraction",
    "late_grad_fraction",
    "peak_layer",
    "layer_concentration",
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

feature_sets = {
    "candidate_only": candidate_features,
    "fingerprint_only": fingerprint_features,
    "fingerprint_plus_candidate":
        fingerprint_features + candidate_features,
}


# ============================================================
# GROUPED CROSS-VALIDATION
# ============================================================

groups = df["episode_id"]

gkf = GroupKFold(
    n_splits=5
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_feature_set(
    target,
    feature_names,
):

    X = df[feature_names].copy()
    y = df[target].values


    categorical = [
        col
        for col in feature_names
        if is_string_dtype(X[col])
    ]

    numeric = [
        col
        for col in feature_names
        if col not in categorical
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical,
            ),
            (
                "num",
                "passthrough",
                numeric,
            ),
        ],
        remainder="drop",
    )

    fold_mae = []
    fold_r2 = []

    for train_idx, test_idx in gkf.split(
        X,
        y,
        groups=groups,
    ):

        model = Pipeline(
            steps=[
                (
                    "preprocess",
                    preprocessor,
                ),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        model.fit(
            X.iloc[train_idx],
            y[train_idx],
        )

        pred = model.predict(
            X.iloc[test_idx]
        )

        fold_mae.append(
            mean_absolute_error(
                y[test_idx],
                pred,
            )
        )

        fold_r2.append(
            r2_score(
                y[test_idx],
                pred,
            )
        )

    return (
        np.mean(fold_mae),
        np.std(fold_mae),
        np.mean(fold_r2),
        np.std(fold_r2),
    )


# ============================================================
# DUMMY BASELINE
# ============================================================

def evaluate_dummy(target):

    y = df[target].values

    fold_mae = []
    fold_r2 = []

    for train_idx, test_idx in gkf.split(
        df,
        y,
        groups=groups,
    ):

        model = DummyRegressor(
            strategy="mean"
        )

        model.fit(
            np.zeros(
                (len(train_idx), 1)
            ),
            y[train_idx],
        )

        pred = model.predict(
            np.zeros(
                (len(test_idx), 1)
            )
        )

        fold_mae.append(
            mean_absolute_error(
                y[test_idx],
                pred,
            )
        )

        fold_r2.append(
            r2_score(
                y[test_idx],
                pred,
            )
        )

    return (
        np.mean(fold_mae),
        np.std(fold_mae),
        np.mean(fold_r2),
        np.std(fold_r2),
    )


# ============================================================
# RUN
# ============================================================

print("\n=== BASELINE FORECASTING ===")

for target in targets:

    print(
        f"\nTARGET: {target}"
    )

    dummy_result = evaluate_dummy(
        target
    )

    print(
        "dummy_mean"
        f" | MAE={dummy_result[0]:.4f}"
        f" ± {dummy_result[1]:.4f}"
        f" | R2={dummy_result[2]:.4f}"
        f" ± {dummy_result[3]:.4f}"
    )

    for name, features in feature_sets.items():

        result = evaluate_feature_set(
            target,
            features,
        )

        print(
            f"{name:<28}"
            f" | MAE={result[0]:.4f}"
            f" ± {result[1]:.4f}"
            f" | R2={result[2]:.4f}"
            f" ± {result[3]:.4f}"
        )