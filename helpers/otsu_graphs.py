from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

project = Path(__file__).resolve().parents[1]
df = pd.read_csv(project / "results/otsu_benchmarks.csv")

if df.empty:
    raise ValueError("Plik otsu_benchmarks.csv nie zawiera pomiarów")

df = df.drop_duplicates(subset=["image", "width", "height"], keep="last")
df["pixels"] = df["width"] * df["height"]
df["series"] = df["image"].str.replace(r"_\d+(?:[xX]\d+)?$", "", regex=True)

time_columns = ["cpu_1t_us", "cpu_2t_us", "gpu_total_us", "gpu_kernel_us"]
values = df[["width", "height", *time_columns]]
if values.isna().any().any() or (values <= 0).any().any():
    raise ValueError("Rozdzielczości i czasy muszą być dodatnie dla skali logarytmicznej")

output_dir = project / "results/otsu_plots"
output_dir.mkdir(parents=True, exist_ok=True)

for series_name, data in df.groupby("series"):
    data = data.sort_values("pixels")
    pixels = data["pixels"]

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(pixels, data["cpu_1t_us"], "o-", label="CPU: 1 wątek")
    ax.plot(pixels, data["cpu_2t_us"], "s-", label="CPU: 2 wątki")
    ax.plot(pixels, data["gpu_total_us"], "^-", label="GPU: total")
    ax.plot(pixels, data["gpu_kernel_us"], "D--", label="GPU: kernel")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")

    labels = [
        f"{width}×{height}"
        for width, height in zip(data["width"], data["height"])
    ]
    ax.set_xticks(pixels)
    ax.set_xticklabels(labels)

    ax.set_xlabel("Rozdzielczość obrazu (pozycja według liczby pikseli)")
    ax.set_ylabel("Mediana czasu [µs] — skala logarytmiczna")
    ax.set_title(f"Otsu - porównanie czasów CPU i GPU")
    ax.grid(True, which="major", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    output = output_dir / f"{series_name}.png"
    fig.savefig(output, dpi=200)
    plt.close(fig)

    print(f"Zapisano: {output}")
