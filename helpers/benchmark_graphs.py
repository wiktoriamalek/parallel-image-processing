import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_benchmarks(default_file, title, parameters=(), numeric_parameters=(),
                    kernel_label="GPU: sam kernel", overhead_batches=1):
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results" / default_file)
    source = parser.parse_args().csv
    data = pd.read_csv(source)
    parameters = [*parameters, *numeric_parameters]
    integer_columns = ["width", "height", "threads", "repeats"]
    time_columns = ["cpu_1t_us", "cpu_nt_us", "gpu_kernel_us", "gpu_total_us"]
    overhead_columns = ["overhead_nt_us"] if "overhead_nt_us" in data.columns else []
    no_copy_columns = ["gpu_no_copy_us"] if "gpu_no_copy_us" in data.columns else []
    positive_columns = [*integer_columns, *time_columns, *no_copy_columns]
    numeric_columns = [*positive_columns, *overhead_columns, *numeric_parameters]
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric)
    if data.empty or data[["image", *parameters]].isna().any().any():
        raise ValueError("Missing measurements, image names or parameters")
    if not np.isfinite(data[numeric_columns].to_numpy()).all():
        raise ValueError("Parameters, resolutions and times must be finite numbers")
    if (data[positive_columns] <= 0).any().any():
        raise ValueError("Resolutions, thread counts, repeats and times must be positive")
    if (data[overhead_columns] < 0).any().any():
        raise ValueError("Thread overhead must not be negative")
    if (data[integer_columns] % 1 != 0).any().any():
        raise ValueError("Resolutions, thread counts and repeats must be integers")
    data[integer_columns] = data[integer_columns].astype(int)
    data = data.drop_duplicates(["image", *integer_columns, *parameters], keep="last").copy()
    if overhead_columns:
        data["cpu_nt_corrected_us"] = data["cpu_nt_us"] - overhead_batches * data["overhead_nt_us"]
    data["pixels"] = data["width"] * data["height"]
    data["series"] = data["image"].map(image_series)
    output_dir = source.parent / f"{source.stem}_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    group_columns = ["series", "repeats", *parameters]

    for key, group in data.groupby(group_columns, sort=True):
        group = group.sort_values(["pixels", "width", "height", "threads"], kind="stable")
        baseline = group.drop_duplicates(["width", "height"], keep="first")
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(baseline["pixels"], baseline["cpu_1t_us"], "o-", label="CPU: 1T")
        for threads, timings in group.groupby("threads"):
            if threads > 1:
                line, = ax.plot(timings["pixels"], timings["cpu_nt_us"], "s-", label=f"CPU: {threads}T")
                if overhead_columns:
                    # for small images the overhead overlaps with the work and the estimate drops to zero or below,
                    # such points say nothing about the computation, so they are left out instead of bending the axis
                    corrected = timings["cpu_nt_corrected_us"].where(timings["cpu_nt_corrected_us"] > 0)
                    ax.plot(timings["pixels"], corrected, "s--", color=line.get_color(),
                            label=f"CPU: {threads}T − narzut (szacunek)")
        ax.plot(baseline["pixels"], baseline["gpu_total_us"], "^-",
                label="GPU: zapis, obliczenia i odczyt")
        if no_copy_columns:
            ax.plot(baseline["pixels"], baseline["gpu_no_copy_us"], "v-",
                    label="GPU: bez kopiowania (obraz na karcie)")
        ax.plot(baseline["pixels"], baseline["gpu_kernel_us"], "D--", label=kernel_label)
        labels = baseline.groupby("pixels", sort=True).apply(
            lambda rows: " / ".join(f"{w}×{h}" for w, h in zip(rows["width"], rows["height"])))
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(labels.index, labels)
        ax.set_xlabel("Rozdzielczość obrazu (pozycja według liczby pikseli)")
        ax.set_ylabel("Mediana czasu [µs]")
        configuration = ", ".join(f"{name}={value}" for name, value in zip(group_columns[1:], key[1:]))
        ax.set_title(f"{title} — {key[0]}\n{configuration}")
        ax.grid(True, which="major", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        name = "_".join([Path(key[0]).stem, *(f"{column}{value}" for column, value in zip(group_columns[1:], key[1:]))])
        fig.savefig(output_dir / (re.sub(r"[^\w.-]+", "_", name) + ".png"), dpi=200)
        plt.close(fig)


def image_series(image):
    path = Path(image)
    stem = re.sub(r"_\d+(?:[xX]\d+)?$", "", path.stem)
    return str(path.with_name(stem + path.suffix))
