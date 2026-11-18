import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd

project = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Plot work scheduling on asymmetric CPU cores.")
parser.add_argument("--scheduling", type=Path, default=project / "results/core_scheduling.csv")
parser.add_argument("--throughput", type=Path, default=project / "results/core_throughput.csv")
parser.add_argument("--output", type=Path, default=project / "results/core_scheduling_plots")
parser.add_argument("--performance-cores", type=int, default=5)
parser.add_argument("--cores", type=int, default=11, help="all physical CPU cores")
parser.add_argument("--columns", type=int, default=2, help="plots per row")
parser.add_argument("--efficiency-ratio", type=float,
                    help="speed of an efficiency core relative to a performance core; by default measured from the "
                         "median filter copies in the throughput file")
args = parser.parse_args()

titles = {"bernsen": "Bernsen (r = 7)", "niblack": "Niblack (r = 7)", "median_filter": "Filtr medianowy 3×3",
          "contrast": "Kontrast"}
schedules = {"static": ("równy podział", "#52514e"), "dynamic": ("dynamiczny, stałe kawałki", "#2a78d6"),
             "guided": ("dynamiczny, malejące kawałki", "#e07b39")}

scheduling = pd.read_csv(args.scheduling).groupby(
    ["algorithm", "image", "threads", "schedule", "chunks_per_thread"], as_index=False).median(numeric_only=True)
throughput = pd.read_csv(args.throughput).groupby(["algorithm", "image", "instances"], as_index=False).median(
    numeric_only=True)
algorithms = [name for name in titles if name in set(scheduling["algorithm"])]
chunks = int(scheduling.loc[scheduling["threads"] == 1, "chunks_per_thread"].mode()[0])
max_threads = int(scheduling["threads"].max())
args.output.mkdir(parents=True, exist_ok=True)

efficiency = args.efficiency_ratio
if efficiency is None:
    median = throughput[throughput["algorithm"] == "median_filter"].set_index("instances")["time_us"]
    if len(median) == args.cores:
        work = median.index * median[1] / median
        efficiency = (work[args.cores] - work[args.performance_cores]) / (args.cores - args.performance_cores)


def hardware_limit(threads):
    return min(threads, args.performance_cores) + efficiency * max(0, threads - args.performance_cores)


def figure_row():
    columns = min(args.columns, len(algorithms))
    rows = -(-len(algorithms) // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4.1 * columns, 3.5 * rows), squeeze=False)
    for axis in axes.flat[len(algorithms):]:
        axis.set_visible(False)
    for axis in axes.flat:
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}".replace(".", ",")))
    return figure, axes.flat


def shade_performance_cores(axis):
    axis.axvspan(0.5, args.performance_cores + 0.5, color="#2a78d6", alpha=0.07,
                 label=f"{args.performance_cores} rdzeni wydajnych")


def save(figure, name, legend_axis):
    handles, labels = legend_axis.get_legend_handles_labels()
    columns = 3
    rows = -(-len(labels) // columns)
    figure.legend(handles, labels, loc="lower center", ncol=columns, fontsize=9, frameon=False)
    figure.tight_layout(rect=(0, 0.04 * rows + 0.005, 1, 1))
    figure.savefig(args.output / name, dpi=200)
    plt.close(figure)
    print(f"Saved {args.output / name}")


figure, axes = figure_row()
for axis, algorithm in zip(axes, algorithms):
    data = scheduling[(scheduling["algorithm"] == algorithm) & (scheduling["chunks_per_thread"] == chunks)]
    single = data[(data["schedule"] == "static") & (data["threads"] == 1)]["time_us"].iloc[0]
    threads = sorted(data["threads"].unique())
    axis.plot(threads, [min(count, args.cores) for count in threads], ":", color="#52514e", linewidth=1.2,
              label="idealne przyspieszenie")
    if efficiency is not None:
        axis.plot(threads, [hardware_limit(count) for count in threads], "--", color="#8a1c7c", linewidth=1.2,
                  label=f"limit: rdzeń E = {efficiency:.2f} rdzenia P".replace(".", ","))
    for schedule, (label, color) in schedules.items():
        rows = data[data["schedule"] == schedule].sort_values("threads")
        axis.plot(rows["threads"], single / rows["time_us"], "o-", color=color, linewidth=2, markersize=5, label=label)
    shade_performance_cores(axis)
    axis.set_title(titles[algorithm])
    axis.set_xticks(threads)
    axis.set_xlim(0.5, max(threads) + 0.5)
    axis.set_ylim(0, args.cores + 1)
    axis.set_xlabel("Liczba wątków CPU")
    axis.set_ylabel("Przyspieszenie względem 1 wątku")
    axis.grid(True, alpha=0.3)
save(figure, "scheduling_speedup.png", axes[0])

figure, axes = figure_row()
for axis, algorithm in zip(axes, algorithms):
    data = scheduling[(scheduling["algorithm"] == algorithm) & (scheduling["threads"] == max_threads) &
                      (scheduling["chunks_per_thread"] == chunks)].set_index("schedule")
    names = [schedule for schedule in schedules if schedule in data.index]
    positions = range(len(names))
    axis.bar([p - 0.2 for p in positions], data.loc[names, "busy_min_us"] / 1000, width=0.4, color="#9cc3ee",
             label="najszybszy wątek")
    axis.bar([p + 0.2 for p in positions], data.loc[names, "busy_max_us"] / 1000, width=0.4, color="#2a78d6",
             label="najwolniejszy wątek")
    axis.plot(list(positions), data.loc[names, "time_us"] / 1000, "D", color="#e07b39", markersize=8,
              label="czas całego algorytmu")
    axis.set_xticks(list(positions))
    axis.set_xticklabels([schedules[name][0].replace(", ", ",\n") for name in names], fontsize=8)
    axis.set_title(f"{titles[algorithm]}, {max_threads} wątków")
    axis.set_ylabel("Czas pracy [ms]")
    axis.grid(True, axis="y", alpha=0.3)
save(figure, "thread_busy_times.png", axes[0])

figure, axes = figure_row()
for axis, algorithm in zip(axes, algorithms):
    data = scheduling[(scheduling["algorithm"] == algorithm) & (scheduling["threads"] == max_threads)]
    dynamic = data[data["schedule"] == "dynamic"].sort_values("chunks_per_thread")
    static = data[data["schedule"] == "static"]["time_us"].iloc[0]
    axis.plot(dynamic["chunks_per_thread"], dynamic["time_us"] / 1000, "o-", color="#2a78d6", linewidth=2,
              label="podział dynamiczny")
    axis.axhline(static / 1000, color="#52514e", linestyle="--", linewidth=1.2, label="równy podział")
    axis.set_xscale("log", base=2)
    axis.set_xticks(dynamic["chunks_per_thread"])
    axis.set_xticklabels(dynamic["chunks_per_thread"].astype(int))
    axis.set_ylim(bottom=0)
    axis.set_title(f"{titles[algorithm]}, {max_threads} wątków")
    axis.set_xlabel("Liczba kawałków na wątek")
    axis.set_ylabel("Czas [ms]")
    axis.grid(True, alpha=0.3)
save(figure, "chunk_size.png", axes[0])

figure, axes = figure_row()
for axis, algorithm in zip(axes, algorithms):
    data = throughput[throughput["algorithm"] == algorithm].sort_values("instances")
    single = data[data["instances"] == 1]["time_us"].iloc[0]
    work = data["instances"] * single / data["time_us"]
    axis.plot(data["instances"], data["instances"], ":", color="#52514e", linewidth=1.2,
              label="gdyby wszystkie rdzenie były wydajne")
    axis.plot(data["instances"], work, "o-", color="#2a78d6", linewidth=2, markersize=5, label="zmierzone")
    shade_performance_cores(axis)
    axis.set_title(titles[algorithm])
    axis.set_xticks(data["instances"])
    axis.set_xlim(0.5, data["instances"].max() + 0.5)
    axis.set_ylim(0, args.cores + 1)
    axis.set_xlabel("Liczba kopii algorytmu uruchomionych naraz")
    axis.set_ylabel("Wykonana praca\n(w kopiach na 1 rdzeniu)")
    axis.grid(True, alpha=0.3)
save(figure, "core_throughput.png", axes[0])
