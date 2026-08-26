"""Render the GPT-2 policy-selection comparison from frozen summary values.

This script contains only the paper-facing frozen aggregate values.  It does
not load, run, or alter experiments or result files.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PDF = ROOT / "figures/policy_selection_comparison.pdf"
OUTPUT_PNG = ROOT / "figures/policy_selection_comparison.png"

POLICIES = ["Static", "Forecast", "Oracle"]
GAIN = [5.331, 7.723, 8.261]
DAMAGE = [0.00200, 0.00535, 0.00356]
VIOLATION = 12.5  # Only the forecast rate was frozen/reported.

BLUE = "#0072B2"
NAVY = "#1B4F72"
GRAY = "#9AA1A8"
DARK = "#222222"
GRID = "#D8DDE3"
COLORS = [GRAY, BLUE, NAVY]


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#555B61")
    ax.tick_params(colors="#3D4248", labelsize=9)
    ax.set_axisbelow(True)


def value_labels(ax: plt.Axes, values: list[float], fmt: str, pad: float) -> None:
    for index, value in enumerate(values):
        ax.text(index, value + pad, fmt.format(value), ha="center", va="bottom",
                color=DARK, fontsize=9, fontweight="semibold")


def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure, axes = plt.subplots(1, 3, figsize=(10.6, 3.35), constrained_layout=True)

    ax = axes[0]
    ax.bar(POLICIES, GAIN, color=COLORS, width=0.62, zorder=3)
    value_labels(ax, GAIN, "{:.3f}", 0.17)
    style_axis(ax)
    ax.set_title("Realized acquisition gain", loc="left", fontsize=11, fontweight="bold", pad=9)
    ax.set_ylabel("$G$ (higher is better)", fontsize=9.5)
    ax.set_ylim(0, 10.4)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.text(0.5, 0.96, "Forecast: +44.9% vs static\n81.7% of oracle gap closed",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.1, color=DARK)

    ax = axes[1]
    ax.bar(POLICIES, DAMAGE, color=COLORS, width=0.62, zorder=3)
    value_labels(ax, DAMAGE, "{:.5f}", 0.00022)
    style_axis(ax)
    ax.set_title("Mean protected damage", loc="left", fontsize=11, fontweight="bold", pad=9)
    ax.set_ylabel("$D$ (lower is better)", fontsize=9.5)
    ax.set_ylim(0, 0.0065)
    ax.set_yticks([0.000, 0.002, 0.004, 0.006])

    ax = axes[2]
    ax.bar(0, VIOLATION, color=BLUE, width=0.62, zorder=3)
    ax.text(0, VIOLATION + 0.8, "5/40 (12.5%)", ha="center", va="bottom",
            color=DARK, fontsize=8.8, fontweight="semibold")
    style_axis(ax)
    ax.set_title("Constraint-violation rate", loc="left", fontsize=11, fontweight="bold", pad=9)
    ax.set_ylabel("Episodes (%)", fontsize=9.5)
    ax.set_xticks([0], ["Forecast"])
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(0, 20.0)
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.text(0.5, 0.96, "Forecast selection is not\nsafety guaranteed", transform=ax.transAxes,
            ha="center", va="top", fontsize=8.4, color=DARK)

    figure.savefig(OUTPUT_PDF, bbox_inches="tight")
    figure.savefig(OUTPUT_PNG, dpi=360, bbox_inches="tight")


if __name__ == "__main__":
    main()
