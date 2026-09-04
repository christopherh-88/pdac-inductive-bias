"""Shared figure style: one palette, one set of rules, every figure in the paper.

Colour assignments are fixed by *entity*, never by rank or by how many series a given
figure happens to draw, so the CNN arm is the same blue in figure 1 and figure 6 and the
reader never has to re-learn the key.

  categorical (identity)  arms: CNN = blue, transformer = orange, identity control = aqua.
                          Validated as a 3-slot all-pairs set: worst CVD dE 9.2, worst
                          normal-vision dE 24.0 on the #fcfcfb surface. Aqua sits below 3:1
                          contrast, so every figure using it also carries direct labels.
  sequential (magnitude)  Dice heatmaps: one hue, light -> dark. Never a rainbow.
  diverging (polarity)    delta-Dice heatmaps: blue <-> red with a NEUTRAL GRAY midpoint, so
                          "no difference" reads as nothing rather than as a third colour.

These are print figures for a paper: a single light surface is a deliberate commitment, not
an oversight, and the light palette is validated against that surface.
"""
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

ARM_COLORS = {
    "cnn": "#2a78d6",              # categorical slot 1 (blue)
    "tf": "#eb6834",               # slot 2 (orange)
    "transformer": "#eb6834",
    "identity_control": "#1baf7a",  # slot 3 (aqua) — sub-3:1, always direct-labelled
    "identity": "#1baf7a",
}
ARM_LABELS = {
    "cnn": "CNN (ResEnc)", "tf": "Transformer (PrimusV2)", "transformer": "Transformer (PrimusV2)",
    "identity_control": "Identity control", "identity": "Identity control",
}

# Sequential blue ramp, steps 100 -> 700, light means "near zero".
SEQ_STEPS = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQUENTIAL = LinearSegmentedColormap.from_list("pdac_seq", SEQ_STEPS)

# Diverging blue <-> red through a neutral gray midpoint (equal steps per arm).
DIV_STEPS = ["#0d366b", "#1c5cab", "#2a78d6", "#86b6ef", "#cde2fb",
             "#f0efec",
             "#f6c9c9", "#ec9a9a", "#e34948", "#c22f2e", "#8f1f1e"]
DIVERGING = LinearSegmentedColormap.from_list("pdac_div", DIV_STEPS)


def arm_color(arm: str) -> str:
    return ARM_COLORS.get(str(arm).lower(), MUTED)


def arm_label(arm: str) -> str:
    return ARM_LABELS.get(str(arm).lower(), str(arm))


def apply_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2.0,
        "text.color": INK,
    })


def despine(ax, keep=("left", "bottom")):
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)
