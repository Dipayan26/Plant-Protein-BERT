"""Generate publication-quality bar charts of UniRef50 superkingdom distribution.

Produces:
  outputs/figures/uniref50_distribution.pdf   — vector, for paper
  outputs/figures/uniref50_distribution.png   — raster, 300 DPI

Usage:
    python scripts/plot_uniref50_distribution.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

matplotlib.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":        9,
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       150,
    "svg.fonttype":     "none",
    "pdf.fonttype":     42,   # embeds fonts in PDF (required by most journals)
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

with open("outputs/uniref50_distribution_v2.json") as f:
    data = json.load(f)

total_clusters = data["total_clusters"]
total_members  = data["total_members"]

# Display order: sorted by cluster count descending, plants highlighted
ORDER = [
    "Bacteria",
    "Metazoa",
    "Fungi",
    "Plants (Viridiplantae)",
    "Other Eukaryota",
    "Unclassified / Root",
    "Archaea",
    "Viruses",
]

LABELS = {
    "Bacteria":               "Bacteria",
    "Metazoa":                "Metazoa",
    "Fungi":                  "Fungi",
    "Plants (Viridiplantae)": "Plants\n(Viridiplantae)",
    "Other Eukaryota":        "Other\nEukaryota",
    "Unclassified / Root":    "Unclassified",
    "Archaea":                "Archaea",
    "Viruses":                "Viruses",
}

# Per-kingdom colors (inspired by HTML palette)
COLORS = {
    "Bacteria":               "#3266AD",
    "Metazoa":                "#73726C",
    "Fungi":                  "#8B6914",
    "Plants (Viridiplantae)": "#4A8C2A",
    "Other Eukaryota":        "#9C9A92",
    "Unclassified / Root":    "#B4B2A9",
    "Archaea":                "#7F77DD",
    "Viruses":                "#D4537E",
}

# Lighter tint for member bars (65% opacity equivalent via hex blend with white)
def lighten(hex_color: str, factor: float = 0.45) -> str:
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    r2 = int(r + (255 - r) * factor)
    g2 = int(g + (255 - g) * factor)
    b2 = int(b + (255 - b) * factor)
    return f"#{r2:02X}{g2:02X}{b2:02X}"

cluster_pcts = [100 * data["superkingdom_clusters"][k] / total_clusters for k in ORDER]
member_pcts  = [100 * data["superkingdom_members"][k]  / total_members  for k in ORDER]

bar_colors_c = [COLORS[k]          for k in ORDER]
bar_colors_m = [lighten(COLORS[k]) for k in ORDER]

x      = np.arange(len(ORDER))
width  = 0.38
labels = [LABELS[k] for k in ORDER]
plant_idx = ORDER.index("Plants (Viridiplantae)")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9.0, 5.0))

bars_c = ax.bar(x - width / 2, cluster_pcts, width, label="Clusters",
                color=bar_colors_c, edgecolor="white", linewidth=0.6, zorder=3)
bars_m = ax.bar(x + width / 2, member_pcts,  width, label="Member sequences",
                color=bar_colors_m, edgecolor="white", linewidth=0.6, zorder=3,
                hatch="///")

# Value labels — all above bars in black
for bars, pcts in [(bars_c, cluster_pcts), (bars_m, member_pcts)]:
    for i, (b, pct) in enumerate(zip(bars, pcts)):
        if pct < 0.5:
            continue
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.7,
            f"{pct:.1f}%",
            ha="center", va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#111111",
        )

# Axis formatting
ax.set_xticks(x)
ax.set_xticklabels(labels, ha="center")
ax.set_ylabel("Percentage of UniRef50 (%)")
ax.set_title(
    "Superkingdom composition of UniRef50",
    pad=10, fontsize=11, fontweight="bold",
)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.set_ylim(0, max(max(cluster_pcts), max(member_pcts)) * 1.25)
ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
ax.set_axisbelow(True)

# Highlight plant column
ax.axvspan(plant_idx - 0.5, plant_idx + 0.5, color="#EAF4E5", zorder=0, linewidth=0)

# Legend
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor="#888888", label="Clusters (representative sequences)"),
    Patch(facecolor="#BBBBBB", hatch="///", label="Member sequences (total in clusters)"),
]
ax.legend(handles=legend_handles, loc="upper right", frameon=False,
          handlelength=1.4, handletextpad=0.6)

fig.tight_layout()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

out_dir = Path("outputs/figures")
out_dir.mkdir(parents=True, exist_ok=True)

fig.savefig(out_dir / "uniref50_distribution.pdf", bbox_inches="tight")
fig.savefig(out_dir / "uniref50_distribution.png", bbox_inches="tight", dpi=300)

print(f"Saved:")
print(f"  {out_dir / 'uniref50_distribution.pdf'}")
print(f"  {out_dir / 'uniref50_distribution.png'}")

plt.show()
