import argparse
from pathlib import Path

from PIL import Image, ImageDraw

parser = argparse.ArgumentParser(description="Draw a grayscale grid image for showing geometric distortion.")
parser.add_argument("output", type=Path)
parser.add_argument("--width", type=int, default=801)
parser.add_argument("--height", type=int, default=601)
parser.add_argument("--spacing", type=int, default=50, help="distance between grid lines in pixels")
parser.add_argument("--line-width", type=int, default=3)
parser.add_argument("--background", type=int, default=235, help="brightness of the background, 0-255")
parser.add_argument("--line", type=int, default=40, help="brightness of the lines, 0-255")
args = parser.parse_args()

image = Image.new("L", (args.width, args.height), args.background)
draw = ImageDraw.Draw(image)
for x in range(0, args.width, args.spacing):
    draw.line([(x, 0), (x, args.height - 1)], fill=args.line, width=args.line_width)
for y in range(0, args.height, args.spacing):
    draw.line([(0, y), (args.width - 1, y)], fill=args.line, width=args.line_width)
image.save(args.output)
