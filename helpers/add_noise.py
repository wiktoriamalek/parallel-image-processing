import argparse
from pathlib import Path

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description="Add impulse (salt-and-pepper) noise to a grayscale image.")
parser.add_argument("input", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--amount", type=float, default=0.05, help="fraction of pixels replaced by noise")
parser.add_argument("--width", type=int, help="resize to this width first, keeping the aspect ratio")
parser.add_argument("--seed", type=int, default=20261011, help="the same seed gives the same noise")
args = parser.parse_args()

image = Image.open(args.input).convert("L")
if args.width:
    height = round(args.width * image.height / image.width)
    image = image.resize((args.width, height), Image.Resampling.LANCZOS)

pixels = np.asarray(image).copy()
count = round(pixels.size * args.amount)
selected = np.random.default_rng(args.seed).choice(pixels.size, count, replace=False)
pixels.flat[selected[:count // 2]] = 0      # half of the chosen pixels become black
pixels.flat[selected[count // 2:]] = 255    # the other half become white
Image.fromarray(pixels).save(args.output)
