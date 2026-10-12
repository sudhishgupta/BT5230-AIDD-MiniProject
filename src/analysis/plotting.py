"""
Shared figure style.

The categorical palette is a validated colourblind-safe set: the three slots
used here clear the all-pairs CVD separation gate (worst deutan dE 9.2) and the
normal-vision floor (worst dE 24.0) against a light surface.  Three is a cap,
not an accident -- beyond three slots no ordering of the parent palette clears
the all-pairs floors, and scatter plots need all-pairs separation because any
two series can end up adjacent on screen.

Because one slot (aqua) sits below 3:1 contrast against the surface, every
figure that uses it carries a legend, and the numbers behind every figure are
also written to results/tables/ as CSV.  That is the documented relief for a
contrast warning.

Sequential encodings use a single blue hue, light to dark.  Diverging
encodings (correlation matrices) use blue-to-red through a neutral grey, never
a rainbow: a rainbow implies ordering where there is none and hides the sign
change at zero.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src import config

# -- validated categorical slots (light mode) ------------------------------
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
VIOLET = "#4a3aa7"
RED = "#e34948"

CATEGORICAL = [BLUE, ORANGE, AQUA]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8980"
GRID = "#e3e2dd"

# cohort -> colour, fixed so a cohort keeps its colour across every figure
COHORT_COLOURS = {
    "jain2017_hic": BLUE, "jain2024_hic": ORANGE, "jetha2019_hic": AQUA,
    "jain2017_acsins": BLUE, "jain2024_acsins": ORANGE,
    "jain2017_tm": BLUE, "jain2024_tm": ORANGE,
    "jain2017_psr": BLUE, "jain2024_psr": ORANGE,
}

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "abdev_seq", ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
)
DIVERGING = LinearSegmentedColormap.from_list(
    "abdev_div", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f0a3a2", "#e34948", "#8f2322"]
)


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.labelcolor": INK_SECONDARY,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "font.size": 9.5,
        "figure.dpi": 110,
        "savefig.dpi": config.FIG_DPI,
        "savefig.bbox": "tight",
    })


def despine(ax, keep=("left", "bottom")) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def save(fig, name: str) -> str:
    """Save a figure into results/figures and return its path."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / f"{name}.{config.FIG_FORMAT}"
    fig.savefig(path)
    plt.close(fig)
    print(f"    figure -> {path.name}")
    return str(path)
