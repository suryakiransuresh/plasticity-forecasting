import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

RESULTS_DIR = "results/experiment_0"

FINGERPRINT_FILE = os.path.join(
    RESULTS_DIR,
    "preupdate_fingerprints_40.csv",
)

OUTCOME_FILE = os.path.join(
    RESULTS_DIR,
    "constrained_replication_candidates_seedcontrolled.csv",
)

OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "forecasting_dataset_320.csv",
)


# ============================================================
# LOAD
# ============================================================

fingerprints = pd.read_csv(FINGERPRINT_FILE)
outcomes = pd.read_csv(OUTCOME_FILE)

print("\n=== INPUT CHECK ===")
print(f"Fingerprint rows: {len(fingerprints)}")
print(f"Outcome rows:     {len(outcomes)}")


# ============================================================
# MERGE
# ============================================================

merged = outcomes.merge(
    fingerprints,
    on="episode_id",
    how="left",
    validate="many_to_one",
    suffixes=("", "_fp"),
)


# ============================================================
# VALIDATION
# ============================================================

print("\n=== MERGE CHECK ===")
print(f"Merged rows:      {len(merged)}")

missing_fingerprint_rows = (
    merged["novel_loss"].isna().sum()
)

print(
    f"Missing fingerprint rows: "
    f"{missing_fingerprint_rows}"
)

duplicate_pairs = (
    merged.duplicated(
        subset=["episode_id", "candidate"]
    ).sum()
)

print(
    f"Duplicate episode-candidate pairs: "
    f"{duplicate_pairs}"
)

candidate_counts = (
    merged.groupby("episode_id")[
        "candidate"
    ]
    .nunique()
)

print(
    f"Candidates per episode: "
    f"min={candidate_counts.min()}, "
    f"max={candidate_counts.max()}"
)


# ============================================================
# HARD ASSERTIONS
# ============================================================

assert len(fingerprints) == 40
assert len(outcomes) == 320
assert len(merged) == 320

assert missing_fingerprint_rows == 0
assert duplicate_pairs == 0

assert candidate_counts.min() == 8
assert candidate_counts.max() == 8


# ============================================================
# SAVE
# ============================================================

merged.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("\n=== FORECASTING DATASET READY ===")
print(f"Rows:    {len(merged)}")
print(f"Columns: {len(merged.columns)}")
print(f"Saved to:\n{OUTPUT_FILE}")