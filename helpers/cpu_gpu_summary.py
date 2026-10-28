import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# how many times the GPU (with copying) is faster than the CPU on all 11 cores, for the largest image of every algorithm
project = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Bar chart: CPU on all cores divided by GPU total time.")
parser.add_argument("--output", type=Path, default=project / "results/cpu_gpu_summary.png")
parser.add_argument("--threads", type=int, default=11)
args = parser.parse_args()

series = [
    ("Progowanie", "threshold", {}),
    ("Korekcja dystorsji", "distortion", {"k1": 0.15, "k2": 0.0}),
    ("Otsu", "otsu", {}),
    ("Konwolucja 3×3", "convolution", {"filter": "box_blur"}),
    ("Bernsen (okno 15×15)", "bernsen", {}),
    ("Filtr medianowy 3×3", "median_filter", {"sort": "standard"}),
]
names, ratios = [], []
for name, file, filters in series:
    data = pd.read_csv(project / f"results/{file}_scaling.csv")
    data = data[data["threads"] == args.threads]
    for column, value in filters.items():
        data = data[data[column] == value]
    largest = data.loc[(data["width"] * data["height"]).idxmax()]
    names.append(f"{name}\n{largest['width']}×{largest['height']}")
    ratios.append(largest["cpu_nt_us"] / largest["gpu_total_us"])

pg_blue, pg_red, ink, muted = "#003865", "#C10A27", "#1f1f1f", "#6b6b6b"
figure, axis = plt.subplots(figsize=(8, 4.6))
colors = [pg_blue if ratio >= 1 else pg_red for ratio in ratios]
positions = range(len(names))
# bars start at 1 (a draw), so the CPU-faster case points left and the GPU-faster cases point right
axis.barh(positions, [ratio - 1 for ratio in ratios], left=1, color=colors, height=0.62)
axis.axvline(1, color=ink, linewidth=1)
for position, ratio in zip(positions, ratios):
    text = f"{ratio:.1f}×".replace(".", ",")
    if ratio < 1:  # the bar is short, so the label goes on the empty side of the draw line
        axis.text(1.04, position, f"CPU {1 / ratio:.0f}× szybsze", va="center", ha="left", color=pg_red, fontsize=11, fontweight="bold")
    else:
        axis.text(ratio * 1.04, position, text, va="center", ha="left", color=ink, fontsize=11, fontweight="bold")
axis.set_xscale("log")
axis.set_xticks([0.5, 1, 2, 5, 10, 20], ["0,5×", "1×", "2×", "5×", "10×", "20×"])
axis.set_xlim(0.4, 30)
axis.minorticks_off()
axis.set_yticks(list(positions), names, fontsize=10)
axis.invert_yaxis()
axis.set_xlabel(f"Ile razy GPU (z kopiowaniem danych) jest szybsze od CPU na {args.threads} rdzeniach", color=muted)
axis.text(1.04, -0.75, "GPU szybsze →", color=pg_blue, fontsize=10, ha="left")
axis.text(0.96, -0.75, "← CPU szybsze", color=pg_red, fontsize=10, ha="right")
for side in ("top", "right", "left"):
    axis.spines[side].set_visible(False)
axis.tick_params(axis="y", length=0)
axis.grid(True, axis="x", alpha=0.25)
figure.tight_layout()
figure.savefig(args.output, dpi=200)
print(f"Zapisano: {args.output}")
