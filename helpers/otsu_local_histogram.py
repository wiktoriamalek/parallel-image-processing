import argparse
from pathlib import Path

import matplotlib.pyplot as plt

resolutions = ["1024×1024", "4096×4096", "8192×8192"]
variants = {
    "globalne atomiki": [4.06, 66.25, 285.20],
    "lokalny histogram, 2 grupy": [13.04, 237.36, 950.31],
    "lokalny histogram, 256 grup": [1.70, 26.39, 108.10],
}
groups = [2, 14, 56, 224, 1024]  # 8192×8192, measured in the same run as the global atomics value below
groupTimes = [950.31, 209.65, 131.34, 107.44, 105.17]
globalAtomics8192 = 285.20

project = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Otsu histogram on GPU: global atomics vs local histogram per work-group.")
parser.add_argument("--output", type=Path, default=project / "results/otsu_plots/local_histogram.png")
args = parser.parse_args()

colors = ["#2a78d6", "#eb6834", "#1baf7a"]
markers = ["o", "s", "^"]
figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5.5))

for (label, times), color, marker in zip(variants.items(), colors, markers):
    left.plot(resolutions, times, marker=marker, color=color, linewidth=2, markersize=8, label=label)
    left.annotate(f"{times[-1]:.0f} µs", (len(resolutions) - 1, times[-1]), xytext=(8, 0),
                  textcoords="offset points", va="center")
left.set_yscale("log")
left.set_xlabel("Rozdzielczość obrazu")
left.set_ylabel("Mediana czasu kerneli GPU [µs]")
left.set_title("Histogram + progowanie na GPU")
left.legend()

right.plot(groups, groupTimes, marker="^", color=colors[2], linewidth=2, markersize=8, label="lokalny histogram")
right.axhline(globalAtomics8192, color="#52514e", linestyle="--", linewidth=1.5, label="globalne atomiki")
right.set_xscale("log", base=2)
right.set_xticks(groups, [str(count) for count in groups])
right.set_xlabel("Liczba grup roboczych (po 256 wątków)")
right.set_ylabel("Mediana czasu kerneli GPU [µs]")
right.set_title("Obraz 8192×8192: wpływ liczby grup roboczych")
right.set_ylim(bottom=0)
right.legend()

for axis in (left, right):
    axis.grid(True, alpha=0.3)
figure.tight_layout()
args.output.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(args.output, dpi=200)
plt.show()
