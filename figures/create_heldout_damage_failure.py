"""Render a held-out-family limitation figure from frozen result summaries.

This script only reads recorded summary CSVs. It neither runs experiments nor
changes any experimental outcome.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
HELDOUT = ROOT / "results/experiment_0/heldout_family_generalization_summary.csv"
OOD = ROOT / "results/experiment_0/ood_aware_damage_selection_summary.csv"
OUTPUT_PDF = ROOT / "figures/heldout_damage_failure.pdf"
OUTPUT_PNG = ROOT / "figures/heldout_damage_failure.png"

BLUE = "#0072B2"
ORANGE = "#D55E00"
TEAL = "#009E73"
GRAY = "#9AA1A8"
DARK = "#222222"
GRID = "#D8DDE3"


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#555B61")
    ax.tick_params(colors="#3D4248", labelsize=8.5)
    ax.set_axisbelow(True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    heldout = pd.read_csv(HELDOUT)
    ood = pd.read_csv(OOD)
    expected_families = {"facts", "symbols", "rules", "arithmetic", "all"}
    if set(heldout["heldout_family"]) != expected_families:
        raise ValueError("Unexpected held-out-family summary rows.")
    required_policies = {"forecast", "ood_aware", "late_lr_1e-5"}
    overall = ood[(ood["scope"] == "overall") & (ood["policy"].isin(required_policies))]
    if set(overall["policy"]) != required_policies:
        raise ValueError("Missing frozen overall OOD-aware policy rows.")
    return heldout, overall.set_index("policy")


def plot_r2(ax: plt.Axes, heldout: pd.DataFrame) -> None:
    order = ["facts", "symbols", "rules", "arithmetic", "all"]
    labels = ["Facts", "Symbols", "Rules", "Arithmetic", "Overall"]
    data = heldout.set_index("heldout_family").loc[order]
    ax.set_xlim(-0.25, 2.02)
    ax.set_ylim(5.05, -0.85)
    for row, (label, (_, values)) in enumerate(zip(labels, data.iterrows())):
        fontweight = "semibold" if label == "Overall" else "normal"
        ax.text(-0.05, row, label, ha="right", va="center", fontsize=8.8,
                color=DARK, fontweight=fontweight)
        for col, (value, facecolor) in enumerate(
            [(values["gain_r2"], "#DCECF7"), (values["damage_r2"], "#F8E1D2")]
        ):
            ax.add_patch(Rectangle((col, row - 0.36), 0.92, 0.72, facecolor=facecolor,
                                   edgecolor="white", linewidth=1.2))
            ax.text(col + 0.46, row, f"{value:.3f}", ha="center", va="center", fontsize=8.9,
                    color=BLUE if col == 0 else ORANGE, fontweight=fontweight)
    ax.text(0.46, -0.55, "Acquisition\nforecast", ha="center", va="center", fontsize=8.5,
            color=BLUE, fontweight="semibold")
    ax.text(1.46, -0.55, "Damage\nforecast", ha="center", va="center", fontsize=8.5,
            color=ORANGE, fontweight="semibold")
    ax.set_axis_off()
    ax.set_title("A. Held-out-family forecast quality ($R^2$)", loc="left", fontsize=10.4,
                 fontweight="bold", pad=10)


def plot_arithmetic(ax: plt.Axes, heldout: pd.DataFrame) -> None:
    arithmetic = heldout.set_index("heldout_family").loc["arithmetic"]
    predicted = arithmetic["mean_predicted_damage"] if "mean_predicted_damage" in arithmetic else None
    # The held-out summary intentionally contains only frozen forecast metrics.
    # The corresponding per-policy summary is the source for the predicted value.
    uncertainty = pd.read_csv(ROOT / "results/experiment_0/uncertainty_aware_damage_selection_summary.csv")
    arithmetic_row = uncertainty[(uncertainty["scope"] == "arithmetic") & (uncertainty["lambda"] == 0.0)].iloc[0]
    predicted = float(arithmetic_row["mean_predicted_damage"])
    realized = float(arithmetic["mean_realized_damage"])
    values = [predicted, realized]
    bars = ax.bar([0, 1], values, color=[GRAY, ORANGE], width=0.6, zorder=3)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.0025, f"{value:.5f}",
                ha="center", va="bottom", fontsize=8.7, fontweight="semibold", color=DARK)
    style_axis(ax)
    ax.set_xticks([0, 1], ["Predicted\ndamage", "Realized\ndamage"])
    ax.set_ylabel("Mean damage, $D$", fontsize=9.3)
    ax.set_ylim(0, 0.105)
    ax.set_yticks([0, 0.025, 0.050, 0.075, 0.100])
    ax.set_title("B. Arithmetic failure", loc="left", fontsize=10.4,
                 fontweight="bold", pad=9)
    ratio = realized / predicted
    violations = int(arithmetic["hard_cap_violations"])
    episodes = int(arithmetic["episodes"])
    ax.text(0.5, 0.94, f"{ratio:.1f}× underprediction\n{violations}/{episodes} budget violations",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.6, color=DARK,
            fontweight="semibold")


def plot_ood_tradeoff(ax: plt.Axes, policies: pd.DataFrame) -> None:
    display = [
        ("Forecast", "forecast", BLUE),
        ("Static\n(late 1e−5)", "late_lr_1e-5", GRAY),
        ("OOD-aware\nfallback", "ood_aware", TEAL),
    ]
    for label, key, color in display:
        row = policies.loc[key]
        ax.scatter(row["mean_realized_gain"], row["mean_realized_damage"], s=90,
                   color=color, edgecolor="white", linewidth=0.9, zorder=4)
        xoff, yoff = {"forecast": (5, 7), "late_lr_1e-5": (5, -14), "ood_aware": (5, 8)}[key]
        ax.annotate(label, (row["mean_realized_gain"], row["mean_realized_damage"]),
                    xytext=(xoff, yoff), textcoords="offset points", fontsize=7.6, color=DARK)
    style_axis(ax)
    ax.set_xlim(3.45, 7.15)
    ax.set_ylim(-0.001, 0.024)
    ax.set_xlabel("Realized acquisition gain, $G$", fontsize=9.3)
    ax.set_ylabel("Mean damage, $D$", fontsize=9.3)
    ax.set_title("C. OOD-aware fallback", loc="left", fontsize=10.4,
                 fontweight="bold", pad=9)
    fallback = policies.loc["ood_aware"]
    ax.text(0.02, 0.96,
            f"Fallback: $G$={fallback['mean_realized_gain']:.3f}, $D$={fallback['mean_realized_damage']:.5f}\n"
            f"0/40 violations; flags {int(fallback['ood_flag_count'])}/40 (85%)",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.9, color=DARK)
    ax.text(0.5, -0.24, "Safety recovery comes with substantial utility loss.", transform=ax.transAxes,
            ha="center", va="top", fontsize=8.0, color=DARK)


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    heldout, policies = load_data()
    figure, axes = plt.subplots(1, 3, figsize=(11.0, 3.75), constrained_layout=True,
                                gridspec_kw={"width_ratios": [1.8, 0.9, 1.15]})
    plot_r2(axes[0], heldout)
    plot_arithmetic(axes[1], heldout)
    plot_ood_tradeoff(axes[2], policies)
    figure.savefig(OUTPUT_PDF, bbox_inches="tight")
    figure.savefig(OUTPUT_PNG, dpi=360, bbox_inches="tight")


if __name__ == "__main__":
    main()
