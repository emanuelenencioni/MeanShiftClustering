#!/usr/bin/env python3
"""
Scikit-learn Mean Shift comparison script for image segmentation.

Uses spatial + color features (5D: x, y, R, G, B) as per sklearn default.
Outputs segmented image to <input_name>_result_py.png
"""

import sys
import os
import numpy as np
import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth


def load_image(path):
    """Load image using OpenCV (BGR format)."""
    img = cv2.imread(path)
    if img is None:
        print(f"Error: Could not load image '{path}'")
        sys.exit(1)
    return img


def image_to_features(img):
    """
    Convert image to feature array for mean shift clustering.
    Returns: (n_pixels, 5) array with [x, y, R, G, B] features
    """
    height, width = img.shape[:2]
    
    # Create coordinate grid
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    
    # Flatten spatial coordinates
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    
    # Flatten color channels (BGR -> RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels = img_rgb.reshape(-1, 3)
    
    # Combine: [x, y, R, G, B]
    features = np.column_stack([x_flat, y_flat, pixels])
    
    return features


def features_to_image(labels, shape):
    """
    Convert cluster labels back to image format.
    Each pixel gets the mean color of its cluster.
    """
    height, width = shape[:2]
    n_clusters = len(np.unique(labels))
    
    # Reshape labels to image dimensions
    labels_img = labels.reshape(height, width)
    
    return labels_img, n_clusters


def apply_cluster_colors(img, labels_img, cluster_centers):
    """
    Create segmented output image by assigning each pixel
    the mean color of its cluster.
    """
    height, width = img.shape[:2]
    output = np.zeros_like(img)
    
    # Extract RGB components from cluster centers (ignore x, y)
    cluster_colors = cluster_centers[:, 2:5]  # [R, G, B]
    
    # Assign mean color to each cluster
    for cluster_id in range(len(cluster_colors)):
        mask = (labels_img == cluster_id)
        # Convert RGB back to BGR for OpenCV
        output[mask] = cluster_colors[cluster_id][::-1]
    
    return output


def run_meanshift(img, bandwidth=150, max_iter=100):
    """
    Run scikit-learn Mean Shift clustering on image.
    Uses spatial + color features (5D).
    """
    print(f"Running Mean Shift (bandwidth={bandwidth}, max_iter={max_iter})...")
    
    # Convert image to feature array
    features = image_to_features(img)
    
    # Run Mean Shift
    ms = MeanShift(bandwidth=bandwidth, max_iter=max_iter, bin_seeding=True)
    ms.fit(features)
    
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    
    # Convert back to image
    labels_img, n_clusters = features_to_image(labels, img.shape)
    output = apply_cluster_colors(img, labels_img, cluster_centers)
    
    return output, n_clusters


def main():
    if len(sys.argv) < 2:
        print("Usage: python meanshift_sklearn.py <image> [bandwidth] [max_iter]")
        print("\nExamples:")
        print("  python meanshift_sklearn.py Images/2.png")
        print("  python meanshift_sklearn.py Images/2.png 30")
        print("  python meanshift_sklearn.py Images/2.png 30 50")
        print("\nDefaults: bandwidth=150, max_iter=100")
        sys.exit(1)
    
    # Parse arguments
    image_path = sys.argv[1]
    bandwidth = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    max_iter = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    
    # Load image
    print(f"Loading image: {image_path}")
    img = load_image(image_path)
    height, width = img.shape[:2]
    print(f"Image size: {width}x{height} ({width * height} pixels)")
    
    # Run mean shift
    result, n_clusters = run_meanshift(img, bandwidth, max_iter)
    
    # Generate output path
    base_path = os.path.splitext(image_path)[0]
    output_path = f"{base_path}_result_py.png"
    
    # Save result
    cv2.imwrite(output_path, result)
    print(f"Segmentation complete: {n_clusters} clusters found")
    print(f"Result saved to: {output_path}")


if __name__ == "__main__":
    main()
