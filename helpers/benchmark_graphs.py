import argparse
import hashlib
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_benchmarks(default_file, title, parameters=(), numeric_parameters=(),
                    kernel_label="GPU: sam kernel"):
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results" / default_file)
    source = parser.parse_args().csv
    data = pd.read_csv(source)
    parameters = [*parameters, *numeric_parameters]
    integer_columns = ["width", "height", "threads", "repeats"]
    time_columns = ["cpu_1t_us", "cpu_nt_us", "gpu_kernel_us", "gpu_total_us"]
    positive_columns = [*integer_columns, *time_columns]
    numeric_columns = [*positive_columns, *numeric_parameters]
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric)
    if data.empty or data[["image", *parameters]].isna().any().any():
        raise ValueError("Brak pomiarów, nazwy obrazu lub parametrów")
    if not np.isfinite(data[numeric_columns].to_numpy()).all():
        raise ValueError("Parametry, rozdzielczości i czasy muszą być skończonymi liczbami")
    if (data[positive_columns] <= 0).any().any():
        raise ValueError("Rozdzielczości, liczby wątków, powtórzeń i czasy muszą być dodatnie")
    if (data[integer_columns] % 1 != 0).any().any():
        raise ValueError("Rozdzielczości, liczby wątków i powtórzeń muszą być całkowite")
    data[integer_columns] = data[integer_columns].astype(int)
    data = data.drop_duplicates(["image", *integer_columns, *parameters], keep="last").copy()
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
                ax.plot(timings["pixels"], timings["cpu_nt_us"], "s-", label=f"CPU: {threads}T")
        ax.plot(baseline["pixels"], baseline["gpu_total_us"], "^-",
                label="GPU: zapis, obliczenia i odczyt")
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
        name = re.sub(r"[^\w.-]+", "_", Path(key[0]).stem)
        identifier = hashlib.sha256(repr(key).encode()).hexdigest()[:10]
        fig.savefig(output_dir / f"{name}_{identifier}.png", dpi=200)
        plt.close(fig)


def image_series(image):
    path = Path(image)
    stem = re.sub(r"_\d+(?:[xX]\d+)?$", "", path.stem)
    return str(path.with_name(stem + path.suffix))
