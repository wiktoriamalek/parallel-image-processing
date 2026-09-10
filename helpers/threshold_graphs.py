from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

project = Path(__file__).resolve().parents[1]

columns = [
    "resolution", "pixels",
    "cpu_1thread_us", "cpu_2thread_us",
    "gpu_submit_wait_us", "gpu_total_us"
]

tables = []
for file in (project / "results/threshold_csv_results").glob("threshold_test_*.csv"):
    table = pd.read_csv(file, header=None, names=columns)
    tables.append(table)

if not tables:
    raise FileNotFoundError("Nie znaleziono plików threshold_test_*.csv")

df = pd.concat(tables, ignore_index=True)
df = df.sort_values("pixels")

print(df)

pixels = df["pixels"] / 1_000_000

plt.figure(figsize=(11, 6))
plt.plot(pixels, df["cpu_1thread_us"] / 1000, "o-", label="CPU: 1 wątek")
plt.plot(pixels, df["cpu_2thread_us"] / 1000, "s-", label="CPU: 2 wątki")
plt.plot(pixels, df["gpu_total_us"] / 1000, "^-", label="GPU: zapis, obliczenia i odczyt")
plt.plot(pixels, df["gpu_submit_wait_us"] / 1000, "D--", label="GPU: uruchomienie kernela i oczekiwanie")

plt.xscale("log", base=2)
plt.yscale("log")
plt.xticks(pixels, [f"{value:.3g}" for value in pixels])
plt.xlabel("Liczba pikseli [mln] — skala logarytmiczna")
plt.ylabel("Mediana czasu [ms] — skala logarytmiczna")
plt.title("Progowanie — porównanie median czasów CPU i GPU")
plt.grid(True, which="major", alpha=0.3)
plt.legend()
plt.tight_layout()

output = project / "results/threshold_comparison.png"
plt.savefig(output, dpi=200)
print(f"Zapisano wykres: {output}")
plt.show()
