import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# speedup of the CPU with more and more threads; the GPU is far above this scale, so it is given in the title
project = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Plot CPU speedup against the number of threads.")
parser.add_argument("--input", type=Path, default=project / "results/thread_scaling")
parser.add_argument("--performance-cores", type=int, default=5)
parser.add_argument("--cores", type=int, default=11, help="all physical CPU cores")
args = parser.parse_args()

titles = {"bernsen": "Bernsen (radius = 7), CMS 2048×2048", "median_filter": "Filtr medianowy 3×3, 3543×2362"}
files = [path for path in sorted(args.input.glob("*.csv")) if path.stem in titles]
figure, axes = plt.subplots(1, len(files), figsize=(6.5 * len(files), 5), squeeze=False)

for axis, path in zip(axes[0], files):
    data = pd.read_csv(path).sort_values("threads").drop_duplicates("threads", keep="last")
    cpu1 = data["cpu_1t_us"].median()  # every run measures one thread again, the median of them is the reference
    threads = [1, *data["threads"]]
    speedup = [1.0, *(cpu1 / data["cpu_nt_us"])]
    ideal = [min(count, args.cores) for count in threads]
    axis.plot(threads, ideal, ":", color="#52514e", linewidth=1.5, label="idealne przyspieszenie (liczba rdzeni)")
    axis.plot(threads, speedup, "o-", color="#2a78d6", linewidth=2, markersize=7, label="CPU: zmierzone przyspieszenie")
    axis.axvspan(0.5, args.performance_cores + 0.5, color="#2a78d6", alpha=0.07, label=f"{args.performance_cores} rdzeni wydajnych")
    axis.axvline(args.cores, color="#52514e", linestyle="--", linewidth=1, label=f"wszystkie {args.cores} rdzeni")
    gpu = cpu1 / data["gpu_total_us"].median()
    axis.set_title(f"{titles[path.stem]}\nGPU z kopiowaniem: {gpu:.0f}× szybciej niż 1 wątek")
    axis.set_xticks(threads)
    axis.set_xlim(0.5, max(threads) + 0.5)
    axis.set_ylim(0, args.cores + 1)
    axis.set_xlabel("Liczba wątków CPU")
    axis.set_ylabel("Przyspieszenie względem 1 wątku CPU")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="lower right")

figure.tight_layout()
output = args.input / "thread_scaling.png"
figure.savefig(output, dpi=200)
print(f"Zapisano: {output}")
