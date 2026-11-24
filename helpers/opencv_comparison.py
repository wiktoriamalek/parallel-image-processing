import argparse
import csv
import os
import statistics
import time

import cv2
import numpy as np

IMAGES = [
    "images/cms_resolution_change/cms_512.png",
    "images/cms_resolution_change/cms_2048.png",
    "images/cms_resolution_change/cms_8192.png",
]
HEADER = ["image", "width", "height", "operation", "implementation", "threads", "repeats", "total_us",
          "compared_pixels", "different_pixels", "max_abs", "note"]
SHARPEN = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)


def load_gray(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.uint32)
    gray = (bgr[:, :, 2] * 77 + bgr[:, :, 1] * 150 + bgr[:, :, 0] * 29) >> 8
    return np.ascontiguousarray(gray.astype(np.uint8))


def operations():
    replicate = cv2.BORDER_REPLICATE
    return [
        ("threshold", 0, lambda image: cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]),
        ("otsu", 0, lambda image: cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]),
        ("convolution_sharpen", 1, lambda image: cv2.filter2D(image, -1, SHARPEN, borderType=replicate)),
        ("convolution_box_blur", 1, lambda image: cv2.blur(image, (3, 3), borderType=replicate)),
        ("median_3x3", 1, lambda image: cv2.medianBlur(image, 3)),
        ("median_5x5", 2, lambda image: cv2.medianBlur(image, 5)),
        ("median_7x7", 3, lambda image: cv2.medianBlur(image, 7)),
        ("canny", 2, lambda image: cv2.Canny(cv2.GaussianBlur(image, (3, 3), 0, borderType=replicate), 50, 100,
                                             L2gradient=True)),
    ]


def measure(run, repeats):
    warm_up_end = time.perf_counter() + 0.2
    count = 0
    while count < 5 or time.perf_counter() < warm_up_end:
        run()
        count += 1
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = run()
        samples.append((time.perf_counter_ns() - start) / 1000.0)
    return statistics.median(samples), result


def compare(expected, actual, border):
    height, width = expected.shape
    a = expected[border:height - border, border:width - border].astype(np.int16)
    b = actual[border:height - border, border:width - border].astype(np.int16)
    difference = np.abs(a - b)
    return a.size, int(np.count_nonzero(difference)), int(difference.max()) if difference.size else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--output", default="results/opencv_comparison.csv")
    parser.add_argument("--repeats", type=int, default=20)
    arguments = parser.parse_args()

    cv2.ocl.setUseOpenCL(True)
    device = cv2.ocl.Device.getDefault()
    print(f"OpenCV {cv2.__version__}, CPU threads {cv2.getNumberOfCPUs()}, OpenCL device: {device.name()}")
    write_header = not os.path.exists(arguments.output)
    with open(arguments.output, "a", newline="") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(HEADER)
        for path in IMAGES:
            image = load_gray(path)
            height, width = image.shape
            print(f"\n{path} ({width}x{height})")
            for name, border, run in operations():
                expected = cv2.imread(os.path.join(arguments.outputs, f"{name}_{width}.png"), cv2.IMREAD_GRAYSCALE)
                variants = []
                cv2.setNumThreads(0)
                variants.append(("opencv_cpu1", 1, lambda: run(image)))
                cv2.setNumThreads(-1)
                variants.append(("opencv_cpu", cv2.getNumThreads(), lambda: run(image)))
                variants.append(("opencv_opencl", 0, lambda: run(cv2.UMat(image)).get()))
                for implementation, threads, call in variants:
                    cv2.setNumThreads(0 if implementation == "opencv_cpu1" else -1)
                    total, output = measure(call, arguments.repeats)
                    compared, different, maximum = compare(expected, output, border)
                    note = ""
                    if name == "otsu":
                        note = f"threshold={int(cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0])}"
                    writer.writerow([path, width, height, name, implementation, threads, arguments.repeats, f"{total:.3f}",
                                     compared, different, maximum, note])
                    print(f"{name:22}{implementation:16}{total:12.1f} us   diff {different} (max {maximum}) {note}")
    cv2.setNumThreads(-1)


if __name__ == "__main__":
    main()
