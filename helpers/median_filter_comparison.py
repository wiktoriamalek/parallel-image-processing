import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

project = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Show the original, noisy and median filtered image side by side.")
parser.add_argument("--original", type=Path, default=project / "images/nuclear.png")
parser.add_argument("--noisy", type=Path, default=project / "images/nuclear_noisy.png")
parser.add_argument("--filtered", type=Path, default=project / "results/median_filter/nuclear_noisy_median.png")
parser.add_argument("--output", type=Path, default=project / "results/median_filter/comparison.png")
parser.add_argument("--crop", type=int, default=256, help="size of the zoomed fragment from the image centre")
args = parser.parse_args()

original, noisy, filtered = (np.asarray(Image.open(path).convert("L")) for path in (args.original, args.noisy, args.filtered))
if not original.shape == noisy.shape == filtered.shape:
    raise ValueError("All three images must have the same size")

noise = np.mean(original != noisy)  # fraction of pixels changed by the noise
height, width = original.shape
top, left = max(0, (height - args.crop) // 2), max(0, (width - args.crop) // 2)
crop = (slice(top, top + args.crop), slice(left, left + args.crop))

titles = ["Oryginał", f"Szum impulsowy ({noise:.0%} pikseli)", "Po filtrze medianowym 3×3"]
figure, axes = plt.subplots(2, 3, figsize=(13, 7))
for column, (pixels, title) in enumerate(zip((original, noisy, filtered), titles)):
    for row, view in enumerate((pixels, pixels[crop])):
        axes[row, column].imshow(view, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axes[row, column].set_title(title + (" — fragment" if row else ""))
        axes[row, column].axis("off")
figure.tight_layout()
args.output.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(args.output, dpi=160)
