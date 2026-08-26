#!/usr/bin/env python3
"""Render the forecasting ablation from frozen grouped-CV values only.

This is a paper-figure reproduction script.  It neither runs experiments nor
loads experimental result files, and the values below are the frozen grouped
cross-validation R² summaries used in the figure.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PDF = ROOT / "figures/forecasting_ablation.pdf"
OUTPUT_PNG = ROOT / "figures/forecasting_ablation.png"

FEATURE_GROUPS = [
    "Candidate\nonly",
    "Magnitude /\nprofile",
    "Alignment",
    "Full\nfingerprint",
]
COLORS = ["#9AA0A6", "#0072B2", "#D55E00", "#4E79A7"]

FROZEN_R2 = {
    "Controlled GPT-2 · acquisition": [0.46, 0.76, 0.55, 0.76],
    "Qwen/RippleEdits · acquisition": [0.22, 0.87, 0.03, 0.87],
    "Controlled GPT-2 · protected damage": [0.12, 0.14, 0.46, 0.43],
    "Qwen/RippleEdits · known-protected damage": [0.02, -0.28, -0.33, -0.23],
}

TITLE = "Pre-update forecasting transfers for acquisition but not protected damage"
YLIM = (-0.48, 1.02)


def plot_panel(ax: plt.Axes, title: str, values: list[float]) -> None:
    """Draw one frozen grouped-CV comparison panel."""
    x = np.arange(len(FEATURE_GROUPS))
    bars = ax.bar(x, values, width=0.68, color=COLORS, edgecolor="none", zorder=3)
    ax.axhline(0, color="#5F6368", linewidth=0.8, zorder=2)

    for bar, value in zip(bars, values):
        position = value + 0.045 if value >= 0 else value - 0.06
        vertical_alignment = "bottom" if value >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            position,
            f"{value:.2f}",
            ha="center",
            va=vertical_alignment,
            fontsize=7.2,
        )

    ax.set_title(title, loc="left", fontsize=9.2, fontweight="bold", pad=8)
    ax.set_xticks(x, FEATURE_GROUPS, fontsize=7.1)
    ax.set_ylim(*YLIM)
    ax.grid(axis="y", color="#DADCE0", linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#5F6368")
    ax.tick_params(axis="y", labelsize=7.5, color="#5F6368")


def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.labelcolor": "#202124",
        "text.color": "#202124",
    })

    figure, axes = plt.subplots(2, 2, figsize=(7.15, 4.35), sharey="row")
    for axis, (panel_title, values) in zip(axes.flat, FROZEN_R2.items()):
        plot_panel(axis, panel_title, values)

    axes[0, 0].set_ylabel("Out-of-fold $R^2$", fontsize=8.5)
    axes[1, 0].set_ylabel("Out-of-fold $R^2$", fontsize=8.5)
    figure.text(
        0.5,
        0.025,
        "Pre-update feature set (each includes candidate metadata)",
        ha="center",
        fontsize=8.4,
    )
    figure.suptitle(TITLE, fontsize=10.5, fontweight="bold", y=0.985)
    figure.subplots_adjust(
        left=0.095, right=0.99, top=0.89, bottom=0.13, wspace=0.04, hspace=0.43
    )
    figure.savefig(OUTPUT_PDF, bbox_inches="tight")
    figure.savefig(OUTPUT_PNG, dpi=450, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
