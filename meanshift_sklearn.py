#!/usr/bin/env python3
"""
meanshift_sklearn.py — Mean shift image segmentation using scikit-learn.

Usage:
    python meanshift_sklearn.py <image> [bandwidth] [max_iter]

Arguments:
    image      : path to input image
    bandwidth  : float, default 150
    max_iter   : int,   default 100 (passed to sklearn MeanShift)

Output:
    Saves result as <stem>_result_py.png next to the input image.
    Prints cluster count to stdout.
"""

import sys
import os
import numpy as np
from PIL import Image
from sklearn.cluster import MeanShift


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <image> [bandwidth] [max_iter]", file=sys.stderr)
        sys.exit(1)

    image_path = sys.argv[1]
    bandwidth = float(sys.argv[2]) if len(sys.argv) >= 3 else 150.0
    max_iter = int(sys.argv[3]) if len(sys.argv) >= 4 else 100

    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    h, w, _ = arr.shape

    # Build 5D feature matrix: [x, y, R, G, B] per pixel
    xs = np.tile(np.arange(w, dtype=np.float32), h)
    ys = np.repeat(np.arange(h, dtype=np.float32), w)
    rgb = arr.reshape(-1, 3)
    features = np.column_stack([xs, ys, rgb])

    ms = MeanShift(bandwidth=bandwidth, max_iter=max_iter, bin_seeding=False, n_jobs=-1)
    ms.fit(features)

    labels = ms.labels_
    centers = ms.cluster_centers_

    n_clusters = len(centers)
    print(f"Clusters: {n_clusters}")

    # Reconstruct image from cluster center RGB values (columns 2,3,4 of centers)
    result_rgb = centers[labels, 2:5].reshape(h, w, 3)
    result_rgb = np.clip(result_rgb, 0, 255).astype(np.uint8)

    result_img = Image.fromarray(result_rgb, "RGB")
    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.dirname(os.path.abspath(image_path))
    out_path = os.path.join(out_dir, f"{stem}_result_py.png")
    result_img.save(out_path)


if __name__ == "__main__":
    main()
