"""Experiment 0.8b: calibrate a conservative forecast-damage threshold.

Uses the fixed out-of-fold predictions produced by Experiment 0.8; it does not
refit models.  For each threshold, choose the highest predicted-gain candidate
whose predicted damage is at or below the threshold.  If none qualify, choose
the lowest predicted-damage candidate instead.
"""

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path("results/experiment_0/forecasted_candidate_selection.csv")
OUTPUT_FILE = Path("results/experiment_0/conservative_selection_calibration.csv")
DAMAGE_CAP = 0.010
THRESHOLDS = (0.010, 0.008, 0.006, 0.004)
BASELINE = "late_lr_1e-5"


def choose_candidate(episode: pd.DataFrame, threshold: float) -> tuple[pd.Series, str]:
    """Apply one conservative threshold with deterministic fallback."""
    safe = episode[episode["predicted_mean_damage"] <= threshold]
    if safe.empty:
        return (
            episode.sort_values(
                ["predicted_mean_damage", "predicted_acquisition_gain", "candidate"],
                ascending=[True, False, True],
            ).iloc[0],
            "fallback_lowest_predicted_damage",
        )
    return (
        safe.sort_values(
            ["predicted_acquisition_gain", "predicted_mean_damage", "candidate"],
            ascending=[False, True, True],
        ).iloc[0],
        "highest_predicted_gain_among_predicted_safe",
    )


def select_policy(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    selected = []
    for _, episode in df.groupby("episode_id", sort=False):
        chosen, reason = choose_candidate(episode, threshold)
        chosen = chosen.copy()
        chosen["selection_reason"] = reason
        selected.append(chosen)
    return pd.DataFrame(selected)


def main() -> None:
    df = pd.read_csv(INPUT_FILE)
    required = {
        "episode_id", "candidate", "acquisition_gain", "mean_damage",
        "predicted_acquisition_gain", "predicted_mean_damage", "oracle_selected",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    episode_count = df["episode_id"].nunique()
    baseline = df[df["candidate"] == BASELINE]
    oracle = df[df["oracle_selected"]]
    if len(baseline) != episode_count or len(oracle) != episode_count:
        raise RuntimeError("Expected one baseline and one oracle selection per episode.")

    baseline_gain = baseline["acquisition_gain"].mean()
    oracle_gain = oracle["acquisition_gain"].mean()
    oracle_gap = oracle_gain - baseline_gain
    candidates = sorted(df["candidate"].unique())
    summaries = []
    for threshold in THRESHOLDS:
        selected = select_policy(df, threshold)
        mean_gain = selected["acquisition_gain"].mean()
        violations = int((selected["mean_damage"] > DAMAGE_CAP).sum())
        row = {
            "predicted_damage_threshold": threshold,
            "episodes": len(selected),
            "mean_actual_gain": mean_gain,
            "mean_actual_damage": selected["mean_damage"].mean(),
            "hard_cap_violations": violations,
            "hard_cap_violation_rate": violations / len(selected),
            "oracle_gap_closed_vs_late_lr_1e-5": (
                (mean_gain - baseline_gain) / oracle_gap
                if not np.isclose(oracle_gap, 0) else np.nan
            ),
            "baseline_mean_actual_gain": baseline_gain,
            "oracle_mean_actual_gain": oracle_gain,
            "fallback_count": int((selected["selection_reason"] == "fallback_lowest_predicted_damage").sum()),
        }
        row.update({f"selected_count_{candidate}": int((selected["candidate"] == candidate).sum())
                    for candidate in candidates})
        summaries.append(row)

    summary = pd.DataFrame(summaries)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_FILE, index=False)

    print("\n=== EXPERIMENT 0.8b: CONSERVATIVE SELECTION CALIBRATION ===")
    print(f"Input: {INPUT_FILE}")
    print(f"Fallback: lowest predicted damage, then highest predicted gain")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved summary: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
