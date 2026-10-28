from benchmark_graphs import plot_benchmarks

plot_benchmarks("otsu_scaling.csv", "Otsu", kernel_label="GPU: suma czasów trzech kerneli",
                overhead_batches=2)
