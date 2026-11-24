import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator
import pandas as pd

project = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Plot the OpenCL and Metal comparison.")
parser.add_argument("--main", type=Path, default=project / "results/opencl_metal.csv")
parser.add_argument("--sizes", type=Path, default=project / "results/opencl_metal_sizes.csv")
parser.add_argument("--variants", type=Path, default=project / "results/opencl_metal_variants.csv")
parser.add_argument("--overhead", type=Path, default=project / "results/opencl_metal_overhead.csv")
parser.add_argument("--workgroups", type=Path, default=project / "results/opencl_metal_workgroups.csv")
parser.add_argument("--output", type=Path, default=project / "results/opencl_metal_plots")
args = parser.parse_args()

names = {"threshold": "Progowanie", "negative": "Negatyw", "contrast": "Kontrast", "otsu": "Otsu",
         "multi_otsu": "Multi-Otsu (3 klasy)", "peaks_valleys": "Doliny histogramu", "bernsen": "Bernsen (r = 7)",
         "niblack": "Niblack (r = 7)", "sauvola": "Sauvola (r = 7)", "convolution": "Konwolucja 3×3",
         "median": "Filtr medianowy 3×3", "distortion": "Korekcja dystorsji", "resize": "Zmiana rozmiaru ×2",
         "region_growing": "Rozrost obszaru", "canny": "Canny"}
opencl_color, metal_color, cpu_color = "#2a78d6", "#eb6834", "#1baf7a"
total_color, kernel_color = "#6250d6", "#1baf7a"
ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted,
                     "ytick.color": muted, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight"})
args.output.mkdir(parents=True, exist_ok=True)


def comma(value, digits=None):
    if digits is None and value >= 10000 and float(value).is_integer():
        return f"{int(value):,}".replace(",", "\u2009")
    text = f"{value:g}" if digits is None else f"{value:.{digits}f}"
    return text.replace(".", ",")


def ratio_axis(axis, low, high):
    axis.set_yscale("log")
    ticks = [t for t in (0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8) if low <= t <= high]
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.yaxis.set_major_locator(FixedLocator(ticks))
    axis.yaxis.set_minor_locator(NullLocator())
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: comma(value)))
    axis.set_ylim(low, high)
    axis.axhline(1, color=muted, linewidth=1)
    axis.grid(axis="y", color=grid, linewidth=0.6)
    axis.set_axisbelow(True)


main = pd.read_csv(args.main)
order = [name for name in names if name in set(main["algorithm"])]
summary = main.groupby("algorithm").median(numeric_only=True).loc[order]

figure, axis = plt.subplots(figsize=(7.2, 4.6))
positions = range(len(order))
axis.hlines(positions, summary["opencl_over_metal_kernel"].clip(upper=summary["opencl_over_metal_total"]),
            summary["opencl_over_metal_kernel"].clip(lower=summary["opencl_over_metal_total"]), color=grid, linewidth=2)
axis.scatter(summary["opencl_over_metal_total"], positions, s=46, color=total_color, zorder=3,
             edgecolor="white", linewidth=1.5, label="cały przebieg (kopiowanie + kernele)")
axis.scatter(summary["opencl_over_metal_kernel"], positions, s=46, marker="D", facecolor="white",
             edgecolor=kernel_color, linewidth=1.8, zorder=3, label="same kernele (znaczniki czasu GPU)")
axis.set_xscale("log")
axis.xaxis.set_major_locator(FixedLocator([0.75, 1, 1.25, 1.5, 2]))
axis.xaxis.set_minor_locator(NullLocator())
axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: comma(value) + "×"))
axis.set_xlim(0.7, 2.3)
axis.axvline(1, color=muted, linewidth=1)
axis.grid(axis="x", color=grid, linewidth=0.6)
axis.set_axisbelow(True)
axis.set_yticks(list(positions), [names[name] for name in order])
axis.invert_yaxis()
axis.set_xlabel("czas OpenCL ÷ czas Metal (więcej niż 1: szybszy Metal)")
axis.text(0.97, len(order) - 0.35, "szybszy OpenCL", color=muted, fontsize=8, va="bottom", ha="right")
axis.text(2.25, len(order) - 0.35, "szybszy Metal", color=muted, fontsize=8, va="bottom", ha="right")
axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8, handletextpad=0.3)
figure.savefig(args.output / "ratio_2048.png")
plt.close(figure)

sizes = pd.concat([main, pd.read_csv(args.sizes)]).groupby(["algorithm", "width"]).median(numeric_only=True)
columns = 3
rows = -(-len(order) // columns)
figure, axes = plt.subplots(rows, columns, figsize=(7.2, 1.75 * rows), sharex=True, sharey=True)
for axis, algorithm in zip(axes.flat, order):
    data = sizes.loc[algorithm].sort_index()
    axis.plot(data.index, data["opencl_over_metal_total"], color=total_color, linewidth=2, marker="o", markersize=4)
    axis.plot(data.index, data["opencl_over_metal_kernel"], color=kernel_color, linewidth=2, marker="D", markersize=4,
              markerfacecolor="white", linestyle=(0, (4, 2)))
    axis.set_xscale("log", base=2)
    axis.xaxis.set_major_locator(FixedLocator([256, 512, 1024, 2048, 4096, 8192]))
    axis.xaxis.set_minor_locator(NullLocator())
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value)}"))
    axis.tick_params(axis="x", labelsize=6.5)
    ratio_axis(axis, 0.5, 4)
    axis.set_title(names[algorithm], fontsize=8.5, color=ink, pad=3)
for axis in axes.flat[len(order):]:
    axis.set_visible(False)
for axis in axes[-1]:
    axis.set_xlabel("bok obrazu [px]")
for axis in axes[:, 0]:
    axis.set_ylabel("OpenCL ÷ Metal")
figure.legend(handles=[Line2D([], [], color=total_color, linewidth=2, marker="o", markersize=4, label="cały przebieg"),
                       Line2D([], [], color=kernel_color, linewidth=2, marker="D", markersize=4, markerfacecolor="white",
                              linestyle=(0, (4, 2)), label="same kernele")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8)
figure.tight_layout(h_pad=0.8, w_pad=0.6)
figure.savefig(args.output / "ratio_sizes.png")
plt.close(figure)

if args.overhead.exists():
    overhead = pd.read_csv(args.overhead).groupby("bytes", as_index=False).median(numeric_only=True).sort_values("bytes")
    megabytes = overhead["bytes"] / 2**20
    figure, (left, right) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for axis in (left, right):
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.xaxis.set_major_locator(FixedLocator(list(megabytes)))
        axis.xaxis.set_minor_locator(NullLocator())
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: comma(value)))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: comma(value)))
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.grid(color=grid, linewidth=0.6)
        axis.set_axisbelow(True)
        axis.set_xlabel("rozmiar obrazu [MB]")
    left.plot(megabytes, overhead["opencl_write_us"], color=opencl_color, linewidth=2, marker="o", markersize=4,
              label="OpenCL: clEnqueueWriteBuffer")
    left.plot(megabytes, overhead["opencl_read_us"], color=opencl_color, linewidth=2, marker="s", markersize=4,
              linestyle=(0, (4, 2)), label="OpenCL: clEnqueueReadBuffer")
    left.plot(megabytes, overhead["metal_write_us"], color=metal_color, linewidth=2, marker="o", markersize=4,
              label="Metal: memcpy do bufora wspólnego")
    left.plot(megabytes, overhead["metal_read_us"], color=metal_color, linewidth=2, marker="s", markersize=4,
              linestyle=(0, (4, 2)), label="Metal: memcpy z bufora wspólnego")
    left.set_ylabel("czas [µs]")
    left.set_title("Jedno kopiowanie obrazu", fontsize=9, color=ink)
    left.legend(fontsize=7, loc="upper left")
    right.plot(megabytes, overhead["opencl_negative_total_us"], color=opencl_color, linewidth=2, marker="o", markersize=4,
               label="OpenCL z kopiowaniem")
    right.plot(megabytes, overhead["opencl_hostptr_negative_us"], color=opencl_color, linewidth=2, marker="^",
               markersize=5, linestyle=(0, (4, 2)), label="OpenCL, CL_MEM_USE_HOST_PTR")
    right.plot(megabytes, overhead["metal_negative_total_us"], color=metal_color, linewidth=2, marker="o", markersize=4,
               label="Metal z kopiowaniem")
    right.plot(megabytes, overhead["metal_nocopy_negative_us"], color=metal_color, linewidth=2, marker="^",
               markersize=5, linestyle=(0, (4, 2)), label="Metal bez kopiowania (bytesNoCopy)")
    right.set_ylim(bottom=100)
    right.set_title("Negatyw: cały przebieg", fontsize=9, color=ink)
    right.legend(fontsize=7, loc="upper left")
    figure.tight_layout(w_pad=1.5)
    figure.savefig(args.output / "overhead.png")
    plt.close(figure)

if args.workgroups.exists():
    groups = pd.read_csv(args.workgroups).groupby(["kernel", "local_x", "local_y"], as_index=False).median(numeric_only=True)
    kernels = [("threshold", "Progowanie (1D)"), ("convolution", "Konwolucja 3×3 (2D)"), ("median", "Filtr medianowy 3×3 (2D)")]
    figure, axes = plt.subplots(len(kernels), 1, figsize=(7.2, 2.15 * len(kernels)))
    for axis, (kernel, title) in zip(axes, kernels):
        data = groups[groups["kernel"] == kernel]
        automatic = data[data["local_x"] == 0]["opencl_kernel_us"].iloc[0]
        data = data[data["local_x"] > 0].copy()
        data["threads"] = data["local_x"] * data["local_y"]
        data = data.sort_values(["threads", "local_x"])
        labels = [f"{x}" if kernel == "threshold" else f"{x}×{y}" for x, y in zip(data["local_x"], data["local_y"])]
        positions = range(len(data))
        width = 0.38
        opencl = data["opencl_kernel_us"].where(data["opencl_kernel_us"] > 0)
        axis.bar([p - width / 2 - 0.01 for p in positions], opencl, width, color=opencl_color, label="OpenCL", zorder=2)
        for position, value in zip(positions, opencl):
            if pd.isna(value):
                axis.text(position - width / 2, 0, "brak\n(>256)", ha="center", va="bottom", fontsize=6, color=muted)
        axis.bar([p + width / 2 + 0.01 for p in positions], data["metal_kernel_us"], width, color=metal_color,
                 label="Metal", zorder=2)
        axis.axhline(automatic, color=ink, linewidth=1.2, linestyle=(0, (5, 3)), zorder=3,
                     label="OpenCL, rozmiar grupy wybrany przez sterownik")
        axis.set_xticks(list(positions), labels, fontsize=7.5)
        axis.set_xlim(-0.6, len(data) - 0.4)
        axis.set_ylabel("czas kernela [µs]")
        axis.set_title(f"{title}; sterownik OpenCL: {comma(automatic, 0)} µs", fontsize=9, color=ink, loc="left")
        axis.grid(axis="y", color=grid, linewidth=0.6)
        axis.set_axisbelow(True)
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: comma(value)))
    axes[0].legend(loc="lower right", ncol=3, fontsize=8, bbox_to_anchor=(1.0, 1.12))
    axes[-1].set_xlabel("rozmiar grupy roboczej (wątki w osi x × w osi y)")
    figure.tight_layout(h_pad=1.0)
    figure.savefig(args.output / "workgroups.png")
    plt.close(figure)

if args.variants.exists():
    variants = pd.read_csv(args.variants).groupby("algorithm").median(numeric_only=True)

    def family(name):
        if name.startswith("median"):
            return "filtr medianowy (4 sortowania, r = 1–7)"
        if name.startswith(("bernsen", "niblack", "sauvola")):
            return "progowanie lokalne (r = 0–127)"
        return "pozostałe algorytmy i warianty"

    families = {"filtr medianowy (4 sortowania, r = 1–7)": opencl_color, "progowanie lokalne (r = 0–127)": metal_color,
                "pozostałe algorytmy i warianty": cpu_color}
    markers = dict(zip(families, ["o", "s", "^"]))
    variants["family"] = [family(name) for name in variants.index]
    figure, axis = plt.subplots(figsize=(5.4, 5.0))
    low = min(variants["opencl_kernel_us"].min(), variants["metal_kernel_us"].min()) / 1.5
    high = max(variants["opencl_kernel_us"].max(), variants["metal_kernel_us"].max()) * 1.5
    axis.plot([low, high], [low, high], color=muted, linewidth=1, zorder=1)
    axis.plot([low, high], [low / 2, high / 2], color=grid, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    axis.plot([low, high], [low * 2, high * 2], color=grid, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    for name, color in families.items():
        data = variants[variants["family"] == name]
        axis.scatter(data["opencl_kernel_us"], data["metal_kernel_us"], s=30, color=color, marker=markers[name],
                     edgecolor="white", linewidth=0.8, label=f"{name}: {len(data)}", zorder=3)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_aspect("equal")
    for setter in (axis.xaxis, axis.yaxis):
        setter.set_major_formatter(FuncFormatter(lambda value, _: comma(value)))
    axis.grid(color=grid, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.set_xlabel("czas kerneli OpenCL [µs]")
    axis.set_ylabel("czas kerneli Metal [µs]")
    axis.text(high / 1.6, high / 1.6 / 2.4, "Metal 2× szybszy", color=muted, fontsize=7.5, ha="right", rotation=45,
              rotation_mode="anchor")
    axis.text(high / 1.6 / 2.4, high / 1.6, "OpenCL 2× szybszy", color=muted, fontsize=7.5, ha="right", rotation=45,
              rotation_mode="anchor")
    axis.legend(loc="upper left", fontsize=7.5, handletextpad=0.2)
    figure.savefig(args.output / "variants_kernels.png")
    plt.close(figure)

print(f"Saved plots to {args.output}")
