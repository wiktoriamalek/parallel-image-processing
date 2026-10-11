import argparse
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser(description="Speedup of the sorting network: standard time / network time (above 1 = network is faster).")
parser.add_argument("csv", nargs="?", type=Path,
                    default=Path(__file__).resolve().parents[1] / "results/median_filter_scaling.csv")
source = parser.parse_args().csv

data = pd.read_csv(source).drop_duplicates(["image", "sort", "threads", "repeats"], keep="last")
times = ["cpu_1t_us", "cpu_nt_us", "gpu_kernel_us", "gpu_total_us"]
table = data.pivot_table(index=["image", "width", "height", "threads", "repeats"], columns="sort", values=times)
speedup = table.xs("standard", axis=1, level="sort") / table.xs("network", axis=1, level="sort")
print(speedup.round(2).to_string())
speedup.to_csv(source.with_name(source.stem + "_speedup.csv"))
