from itertools import pairwise

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .config import COLORS, STAGES


def set_style():
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#D3D8DE",
            "axes.labelcolor": "#26313D",
            "text.color": "#26313D",
            "xtick.color": "#52606D",
            "ytick.color": "#52606D",
            "grid.color": "#E8EBEF",
            "grid.linewidth": 0.65,
            "axes.axisbelow": True,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.pad_inches": 0.3,
            "svg.fonttype": "none",
            "svg.hashsalt": "nanoreason",
            "pdf.fonttype": 42,
            "axes.titleweight": "bold",
            "axes.unicode_minus": True,
        }
    )


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for extension in ["png", "svg"]:
        fig.savefig(
            path.with_suffix(f".{extension}"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.show()
    plt.close(fig)


def footer(fig, text):
    fig.text(
        0.035,
        0.027,
        text,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#63717F",
        linespacing=1.5,
    )


def stage_legend(fig, y=0.84):
    fig.legend(
        handles=[Patch(facecolor=COLORS[s], label=s) for s in STAGES],
        loc="upper left",
        bbox_to_anchor=(0.03, y),
        ncol=4,
        frameon=False,
        columnspacing=2,
        handlelength=1.3,
        fontsize=10,
    )


def stage_arrows(ax, points):
    for start, end in pairwise(points):
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#9AA4AE",
                "lw": 1.3,
                "mutation_scale": 11,
                "shrinkA": 7,
                "shrinkB": 7,
            },
            zorder=2,
        )
