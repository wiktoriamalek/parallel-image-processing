import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

project = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Quality of the plain and switching median filter for several window sizes.")
parser.add_argument("--original", type=Path, default=project / "images/nuclear.png")
parser.add_argument("--noisy", type=Path, default=project / "images/nuclear_noisy.png")
parser.add_argument("--results", type=Path, default=project / "results/median_filter")
parser.add_argument("--crop", type=int, default=160, help="size of the zoomed fragment from the image centre")
args = parser.parse_args()

MODES = {"plain": ("Mediana zwykła", "#2a78d6", "o"), "switching": ("Mediana przełączająca (tylko piksele 0 i 255)", "#eb6834", "s")}
TEXT = "#52514e"


def load(path):
    return np.asarray(Image.open(path).convert("L")).astype(np.int16)


original, noisy = load(args.original), load(args.noisy)
border = 3
inner = (slice(border, -border), slice(border, -border))
impulses = ((noisy == 0) | (noisy == 255)) & (noisy != original)


def metrics(image):
    error = (image[inner] - original[inner]).astype(np.float64)
    clean = (noisy == original)[inner]
    return {
        "psnr_db": 10 * np.log10(255 ** 2 / np.mean(error ** 2)),
        "mae": np.mean(np.abs(error)),
        "changed_clean_percent": 100 * np.mean((image != noisy)[inner][clean]),
        "impulses_left_percent": 100 * np.mean((image == noisy)[inner][impulses[inner]]),
    }


rows = [{"mode": "noisy", "window": "-", **metrics(noisy)}]
filtered = {}
for mode in MODES:
    for size in (3, 5, 7):
        name = args.noisy.stem + "_median" + ("_switching" if mode == "switching" else "") + ("" if size == 3 else f"_{size}x{size}") + ".png"
        path = args.results / name
        if not path.exists():
            continue
        filtered[mode, size] = load(path)
        rows.append({"mode": mode, "window": f"{size}×{size}", **metrics(filtered[mode, size])})
table = pd.DataFrame(rows)
print(f"Metrics without the {border}-pixel border (borders are not filtered):")
print(table.round(2).to_string(index=False))
table.to_csv(args.results / "quality_modes.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 5.5))
for mode, (label, color, marker) in MODES.items():
    part = table[table["mode"] == mode]
    if part.empty:
        continue
    ax.plot(part["window"], part["psnr_db"], marker=marker, color=color, linewidth=2, markersize=8, label=label)
    for window, psnr in zip(part["window"], part["psnr_db"]):
        ax.annotate(f"{psnr:.1f} dB", (window, psnr), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8, color=TEXT)
noisy_psnr = table.loc[table["mode"] == "noisy", "psnr_db"].iloc[0]
ax.axhline(noisy_psnr, color=TEXT, linewidth=1, linestyle="--")
ax.annotate(f"obraz zaszumiony: {noisy_psnr:.1f} dB", (0, noisy_psnr), xytext=(0, 5), textcoords="offset points", fontsize=8, color=TEXT)
ax.set_xlabel("Rozmiar okna")
ax.set_ylabel("PSNR względem oryginału [dB]")
noise = 100 * np.mean(original != noisy)
ax.set_title(f"Jakość filtru medianowego, szum impulsowy {noise:.0f}% pikseli")
ax.grid(True, axis="y", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="center right")
fig.tight_layout()
fig.savefig(args.results / "quality_psnr_by_window.png", dpi=200)
plt.close(fig)

if ("plain", 3) in filtered and ("switching", 3) in filtered:
    height, width = original.shape
    top, left = (height - args.crop) // 2, (width - args.crop) // 2
    crop = (slice(top, top + args.crop), slice(left, left + args.crop))
    panels = [(original, "Oryginał"), (noisy, f"Szum impulsowy, PSNR {noisy_psnr:.1f} dB")]
    for mode in MODES:
        psnr = table.loc[(table["mode"] == mode) & (table["window"] == "3×3"), "psnr_db"].iloc[0]
        panels.append((filtered[mode, 3], f"{MODES[mode][0].split(' (')[0]} 3×3, PSNR {psnr:.1f} dB"))
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.4))
    for ax, (pixels, title) in zip(axes, panels):
        ax.imshow(pixels[crop], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"Fragment {args.crop}×{args.crop} pikseli ze środka obrazu")
    fig.tight_layout()
    fig.savefig(args.results / "quality_switching_comparison.png", dpi=160)
    plt.close(fig)
