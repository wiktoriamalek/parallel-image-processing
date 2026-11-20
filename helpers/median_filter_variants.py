import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

project = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Compare median filter variants: sorting method, window size and strip length.")
parser.add_argument("csv", nargs="?", type=Path, default=project / "results/median_filter_variants.csv")
parser.add_argument("--strips", type=Path, default=project / "results/median_filter_strips.csv")
args = parser.parse_args()

VARIANTS = {
    "standard": ("Sortowanie przez wstawianie (CPU: nth_element)", "#2a78d6", "o"),
    "network": ("Sieć selekcji, jedno okno", "#eb6834", "s"),
    "bitonic": ("Sieć bitoniczna (dopełnienie do 2^n)", "#1baf7a", "^"),
    "shared": ("Sieć współdzielona przez pas okien", "#eda100", "D"),
}
WINDOW_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}
TEXT = "#52514e"
keys = ["image", "sort", "radius", "strip", "threads", "repeats"]


def load(path):
    data = pd.read_csv(path).drop_duplicates(keys, keep="last").copy()
    data["pixels"] = data["width"] * data["height"]
    data["window"] = (2 * data["radius"] + 1).map(lambda size: f"{size}×{size}")
    return data


def style(ax, ylabel):
    ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def label_end(ax, x, y, text):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=TEXT)


def by_window(data, column, ylabel, title, output):
    largest = data[data["pixels"] == data["pixels"].max()]
    fig, ax = plt.subplots(figsize=(10, 6))
    for sort, (label, color, marker) in VARIANTS.items():
        rows = largest[largest["sort"] == sort].sort_values("radius")
        if rows.empty:
            continue
        ax.plot(rows["window"], rows[column], marker=marker, color=color, linewidth=2, markersize=8, label=label)
        last = rows.iloc[-1]
        label_end(ax, last["window"], last[column], f"{last[column] / 1000:.2f} ms")
    style(ax, ylabel)
    ax.set_xlabel("Rozmiar okna")
    width, height = largest.iloc[0][["width", "height"]]
    ax.set_title(f"{title}\nobraz {width}×{height}, mediana z {largest.iloc[0]['repeats']} powtórzeń")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def by_resolution(data, column, ylabel, title, output):
    radii = sorted(data["radius"].unique())
    fig, axes = plt.subplots(1, len(radii), figsize=(5 * len(radii), 5), sharey=True)
    for ax, radius in zip(axes, radii):
        for sort, (label, color, marker) in VARIANTS.items():
            rows = data[(data["sort"] == sort) & (data["radius"] == radius)].sort_values("pixels")
            if not rows.empty:
                ax.plot(rows["pixels"], rows[column], marker=marker, color=color, linewidth=2, markersize=7, label=label)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Liczba pikseli")
        size = 2 * radius + 1
        ax.set_title(f"okno {size}×{size}")
        style(ax, ylabel if radius == radii[0] else "")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def by_strip(strips, output):
    fig, (time_ax, work_ax) = plt.subplots(1, 2, figsize=(12, 5))
    for radius, rows in strips.groupby("radius"):
        rows = rows.sort_values("strip")
        size = 2 * radius + 1
        color = WINDOW_COLORS.get(radius, TEXT)
        time_ax.plot(rows["strip"], rows["gpu_kernel_us"], marker="o", color=color, linewidth=2, markersize=7, label=f"okno {size}×{size}")
        work_ax.plot(rows["strip"], rows["min_max_per_pixel"], marker="o", color=color, linewidth=2, markersize=7, label=f"okno {size}×{size}")
    for ax in (time_ax, work_ax):
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(strips["strip"].unique()), [str(s) for s in sorted(strips["strip"].unique())])
        ax.set_xlabel("Liczba okien liczonych przez jeden work-item (długość pasa)")
    style(time_ax, "Czas kernela [µs]")
    style(work_ax, "Operacje min/max na piksel")
    time_ax.set_title("GPU: sam kernel")
    work_ax.set_title("Praca na piksel po przycięciu sieci")
    time_ax.legend(frameon=False)
    width, height = strips.iloc[0][["width", "height"]]
    fig.suptitle(f"Sieć współdzielona: wpływ długości pasa, obraz {width}×{height}")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


data = load(args.csv)
output_dir = args.csv.parent / f"{args.csv.stem}_plots"
output_dir.mkdir(parents=True, exist_ok=True)
by_window(data, "gpu_kernel_us", "Mediana czasu kernela [µs]", "Filtr medianowy na GPU: sam kernel", output_dir / "gpu_kernel_by_window.png")
by_window(data, "gpu_total_us", "Mediana czasu [µs]", "Filtr medianowy na GPU: zapis, obliczenia i odczyt", output_dir / "gpu_total_by_window.png")
by_window(data, "cpu_1t_us", "Mediana czasu [µs]", "Filtr medianowy na CPU: 1 wątek", output_dir / "cpu_1t_by_window.png")
by_window(data, "cpu_nt_us", "Mediana czasu [µs]", f"Filtr medianowy na CPU: {data['threads'].max()} wątków", output_dir / "cpu_nt_by_window.png")
by_resolution(data, "gpu_kernel_us", "Mediana czasu kernela [µs]", "Filtr medianowy na GPU: sam kernel", output_dir / "gpu_kernel_by_resolution.png")

largest = data[data["pixels"] == data["pixels"].max()]
table = largest.pivot_table(index="window", columns="sort", values=["gpu_kernel_us", "gpu_total_us", "cpu_1t_us", "cpu_nt_us"])
speedup = pd.DataFrame(index=table.index)
for column in ["gpu_kernel_us", "gpu_total_us", "cpu_1t_us", "cpu_nt_us"]:
    for sort in VARIANTS:
        if (column, sort) in table.columns:
            speedup[f"{column}:{sort}"] = table[(column, "network")] / table[(column, sort)]
work = largest[largest["sort"] != "standard"].pivot_table(index="window", columns="sort", values=["comparators_per_pixel", "min_max_per_pixel"])
print("Times [us] on the largest image:\n", table.round(1).to_string())
print("\nSpeedup relative to the single-window network (above 1 = faster than network):\n", speedup.round(2).to_string())
print("\nComparators before pruning and min/max operations after pruning, per pixel:\n", work.round(2).to_string())
speedup.to_csv(output_dir / "speedup_vs_network.csv")
work.to_csv(output_dir / "network_work.csv")

if args.strips.exists():
    by_strip(load(args.strips), output_dir / "shared_strip_length.png")
