from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

project = Path(__file__).resolve().parents[1]
df = pd.read_csv(project / "results/distortion_benchmarks.csv")

time_columns = ["cpu_1t_us", "cpu_2t_us", "gpu_total_us", "gpu_kernel_us"]
if df.empty:
    raise ValueError("Plik distortion_benchmarks.csv nie zawiera pomiarów")

values = df[["width", "height", "k1", "k2", *time_columns]]
if not np.isfinite(values.to_numpy()).all():
    raise ValueError("Parametry, rozdzielczości i czasy muszą być skończonymi liczbami")
if (df[["width", "height", *time_columns]] <= 0).any().any():
    raise ValueError("Rozdzielczości i czasy muszą być dodatnie dla skali logarytmicznej")
if not df["variant"].isin(["original", "resized"]).all():
    raise ValueError("Wariant obrazu musi mieć wartość original lub resized")
if df.duplicated(["variant", "width", "height", "k1", "k2"]).any():
    raise ValueError("CSV zawiera powtórzone pomiary dla tych samych parametrów")

df["pixels"] = df["width"] * df["height"]
output_dir = project / "results/distortion_plots"
output_dir.mkdir(parents=True, exist_ok=True)


def plot_times(data, positions, labels, title, filename, log_x=False):
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(positions, data["cpu_1t_us"], "o-", label="CPU: 1 wątek")
    ax.plot(positions, data["cpu_2t_us"], "s-", label="CPU: 2 wątki")
    ax.plot(positions, data["gpu_total_us"], "^-", label="GPU: zapis, obliczenia i odczyt")
    ax.plot(positions, data["gpu_kernel_us"], "D--", label="GPU: sam kernel")

    if log_x:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Rozdzielczość obrazu")
    else:
        ax.set_xlabel("Współczynniki dystorsji radialnej")
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mediana czasu [µs]")
    ax.set_title(title)
    ax.grid(True, which="major", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    output = output_dir / filename
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Zapisano: {output}")


for (k1, k2), data in df[df["variant"] == "resized"].groupby(["k1", "k2"]):
    data = data.sort_values("pixels")
    labels = [f"{width}×{height}" for width, height in zip(data["width"], data["height"])]
    parameters = f"k1_{k1:g}_k2_{k2:g}".replace(".", "p")
    plot_times(
        data,
        data["pixels"],
        labels,
        f"Korekcja dystorsji radialnej \nk1 = {k1:g}, k2 = {k2:g}",
        f"mitsubishi_{parameters}.png",
        log_x=True,
    )

for (width, height), data in df[df["variant"] == "original"].groupby(["width", "height"]):
    data = data.sort_values(["k1", "k2"])
    labels = [f"k1 = {k1:g}\nk2 = {k2:g}" for k1, k2 in zip(data["k1"], data["k2"])]
    plot_times(
        data,
        range(len(data)),
        labels,
        f"Korekcja dystorsji radialnej — oryginalny obraz {width}×{height}",
        f"mitsubishi_original_{width}x{height}.png",
    )
