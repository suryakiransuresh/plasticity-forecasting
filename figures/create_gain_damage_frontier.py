"""Render the paper-facing gain--damage frontier from frozen result tables.

This script only aggregates recorded candidate outcomes; it does not run or
alter any experiment.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
GPT2_RESULTS = ROOT / "results/experiment_0/constrained_replication_candidates_seedcontrolled.csv"
QWEN_RESULTS = ROOT / "results/external_validation/rippleedits_qwen_final_candidate_sweep.csv"
OUTPUT_PDF = ROOT / "figures/gain_damage_frontier.pdf"
OUTPUT_PNG = ROOT / "figures/gain_damage_frontier.png"

BLUE = "#0072B2"
GRAY = "#9AA1A8"
DARK = "#222222"
GRID = "#D8DDE3"


def pareto_frontier(frame: pd.DataFrame) -> pd.Series:
    """Return whether each candidate is nondominated (higher gain, lower damage)."""
    is_frontier = []
    for _, row in frame.iterrows():
        dominates = (
            (frame["gain"] >= row["gain"])
            & (frame["damage"] <= row["damage"])
            & ((frame["gain"] > row["gain"]) | (frame["damage"] < row["damage"]))
        )
        is_frontier.append(not dominates.any())
    return pd.Series(is_frontier, index=frame.index)


def style_axis(ax: plt.Axes) -> None:
    ax.axhline(0, color="#6E747B", linewidth=1.0, zorder=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#555B61")
    ax.tick_params(colors="#3D4248", labelsize=9)


def plot_gpt2(ax: plt.Axes) -> None:
    raw = pd.read_csv(GPT2_RESULTS)
    points = (raw.groupby("candidate", sort=False)
              .agg(gain=("acquisition_gain", "mean"), damage=("mean_damage", "mean"))
              .reset_index())
    points["frontier"] = pareto_frontier(points)
    points["label"] = points["candidate"].map({
        "full_lr_1e-6": "Full 1e−6", "full_lr_1e-5": "Full 1e−5",
        "early_lr_5e-6": "Early 5e−6", "early_lr_1e-5": "Early 1e−5",
        "early_lr_2e-5": "Early 2e−5", "late_lr_5e-6": "Late 5e−6",
        "late_lr_1e-5": "Late 1e−5", "late_lr_2e-5": "Late 2e−5",
    })
    frontier = points[points["frontier"]].sort_values("gain")
    ax.plot(frontier["gain"], frontier["damage"], color=BLUE, linewidth=1.6, zorder=1)
    ax.scatter(points.loc[~points["frontier"], "gain"], points.loc[~points["frontier"], "damage"],
               s=38, color=GRAY, edgecolor="white", linewidth=0.7, zorder=3)
    ax.scatter(frontier["gain"], frontier["damage"], s=48, color=BLUE,
               edgecolor="white", linewidth=0.7, zorder=4)
    offsets = {
        "Full 1e−6": (6, -13), "Early 5e−6": (-12, 10), "Late 5e−6": (7, -16),
        "Early 1e−5": (-55, 12), "Late 1e−5": (4, 11), "Early 2e−5": (-43, 8),
        "Late 2e−5": (7, -14), "Full 1e−5": (-40, 10),
    }
    for _, row in points.iterrows():
        ax.annotate(row["label"], (row["gain"], row["damage"]), xytext=offsets[row["label"]],
                    textcoords="offset points", fontsize=6.6, color=DARK)
    style_axis(ax)
    ax.set_title("Controlled GPT-2 (40 episodes)", loc="left", fontsize=11, fontweight="bold", pad=9)
    ax.set_xlabel("Acquisition gain, $G$ (higher is better)", fontsize=9.5)
    ax.set_ylabel("Mean protected damage, $D$ (lower is better)", fontsize=9.5)
    ax.set_xlim(1.65, 9.65)
    ax.set_ylim(-0.0045, 0.075)
    ax.set_yticks([0.00, 0.02, 0.04, 0.06])


def plot_qwen(ax: plt.Axes) -> None:
    raw = pd.read_csv(QWEN_RESULTS)
    points = (raw.groupby("candidate", sort=False)
              .agg(gain=("acquisition_gain", "mean"), damage=("D_known", "mean"),
                   stable=("numerical_stability", lambda status: int((status == "stable").sum())))
              .reset_index())
    points["frontier"] = pareto_frontier(points)
    points["label"] = points["candidate"].map({
        "full_lr_1e-5_steps1": "Full ×1", "early_lr_1e-5_steps10": "Early ×10",
        "late_lr_1e-5_steps10": "Late ×10", "full_lr_2e-5_steps5": "Full 2e−5 ×5†",
    })
    aggressive = points[points["candidate"] == "full_lr_2e-5_steps5"].iloc[0]
    frontier = points[points["frontier"]].sort_values("gain")
    stable_frontier = frontier[frontier["candidate"] != "full_lr_2e-5_steps5"]
    ax.plot(frontier["gain"], frontier["damage"], color=BLUE, linewidth=1.6, zorder=1)
    ax.scatter(points.loc[~points["frontier"], "gain"], points.loc[~points["frontier"], "damage"],
               s=44, color=GRAY, edgecolor="white", linewidth=0.7, zorder=3)
    ax.scatter(stable_frontier["gain"], stable_frontier["damage"], s=52, color=BLUE,
               edgecolor="white", linewidth=0.7, zorder=4)
    ax.scatter(aggressive["gain"], aggressive["damage"], s=72, marker="X", color=BLUE,
               edgecolor="white", linewidth=0.7, zorder=5)
    offsets = {"Full ×1": (-1, 10), "Early ×10": (8, 9), "Late ×10": (8, -14),
               "Full 2e−5 ×5†": (8, 8)}
    for _, row in points.iterrows():
        ax.annotate(row["label"], (row["gain"], row["damage"]), xytext=offsets[row["label"]],
                    textcoords="offset points", fontsize=8, color=DARK)
    ax.annotate("† 8/100 nonfinite\n(92/100 stable)", (aggressive["gain"], aggressive["damage"]),
                xytext=(9, -29), textcoords="offset points", fontsize=7.4, color=DARK,
                ha="left", va="top")
    style_axis(ax)
    ax.set_title("Qwen2.5-0.5B / RippleEdits (100 edits)", loc="left", fontsize=11, fontweight="bold", pad=9)
    ax.set_xlabel("Acquisition gain, $G$ (higher is better)", fontsize=9.5)
    ax.set_ylabel("Mean known-protected damage, $D_{known}$ (lower is better)", fontsize=9.5)
    ax.set_xlim(2.05, 6.35)
    ax.set_ylim(-0.31, 0.57)
    ax.set_yticks([-0.2, 0.0, 0.2, 0.4])


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(10.6, 4.4), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.55, 0.85]},
    )
    plot_gpt2(left)
    plot_qwen(right)
    figure.legend(handles=[
        Line2D([0], [0], color=BLUE, marker="o", markersize=5.5, linewidth=1.6, label="Pareto frontier"),
        Line2D([0], [0], color=GRAY, marker="o", markersize=5.5, linewidth=0, label="Dominated candidate"),
        Line2D([0], [0], color=BLUE, marker="X", markersize=6.5, linewidth=0, label="Candidate with nonfinite runs"),
    ], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12), frameon=False,
        fontsize=8.7, handletextpad=0.5, columnspacing=1.8)
    figure.savefig(OUTPUT_PDF, bbox_inches="tight")
    figure.savefig(OUTPUT_PNG, dpi=360, bbox_inches="tight")


if __name__ == "__main__":
    main()
