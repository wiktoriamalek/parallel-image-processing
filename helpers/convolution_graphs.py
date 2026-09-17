from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

project = Path(__file__).resolve().parents[1]
df = pd.read_csv(project / "results/convolution_benchmarks.csv")
df["pixels"] = df["width"] * df["height"]

output_dir = project / "results/convolution_plots"
output_dir.mkdir(parents=True, exist_ok=True)

for filter_name, data in df.groupby("filter"):
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
    ax.set_title(f"{filter_name} — porównanie czasów CPU i GPU")
    ax.grid(True, which="major", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    output = output_dir / f"{filter_name}.png"
    fig.savefig(output, dpi=200)
    plt.close(fig)

    print(f"Zapisano: {output}")