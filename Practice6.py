#!/usr/bin/env python3
"""
sobel_vs_scharr_demo.py

Compares the gradient magnitude output of Sobel and Scharr filters
on one or more grayscale images and displays original, Sobel, and Scharr.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

def gradient_magnitude_filter(image_gray: np.ndarray, filter_type='sobel', ksize=3):
    """
    Compute gradient magnitude using Sobel or Scharr.

    Parameters
    ----------
    image_gray : np.ndarray
        Input grayscale image.
    filter_type : str
        'sobel' or 'scharr'.
    ksize : int
        Kernel size for Sobel (ignored for Scharr).

    Returns
    -------
    np.ndarray
        8-bit gradient magnitude.
    """
    if filter_type == 'sobel':
        dx = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=ksize)
        dy = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=ksize)
    else:  # scharr
        dx = cv2.Scharr(image_gray, cv2.CV_64F, 1, 0)
        dy = cv2.Scharr(image_gray, cv2.CV_64F, 0, 1)

    mag = np.sqrt(dx**2 + dy**2)
    mag = np.clip((mag / mag.max()) * 255, 0, 255).astype(np.uint8)
    return mag

def main(image_paths, ksize):
    n = len(image_paths)
    fig, axes = plt.subplots(n, 3, figsize=(15, 5*n))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    for i, path in enumerate(image_paths):
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"Error: Could not load '{path}'", file=sys.stderr)
            continue

        sobel_mag = gradient_magnitude_filter(gray, 'sobel', ksize)
        scharr_mag = gradient_magnitude_filter(gray, 'scharr')

        axes[i, 0].imshow(gray, cmap='gray')
        axes[i, 0].set_title(f"{path}\nOriginal")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(sobel_mag, cmap='gray')
        axes[i, 1].set_title(f"Sobel (ksize={ksize})")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(scharr_mag, cmap='gray')
        axes[i, 2].set_title("Scharr")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Sobel vs. Scharr gradient magnitudes on images."
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="Paths to one or more grayscale images (e.g. Test.jpg Test2.jpg)"
    )
    parser.add_argument(
        "--ksize", "-k",
        type=int,
        default=3,
        help="Sobel kernel size (odd integer, default=3)"
    )
    args = parser.parse_args()
    main(args.images, args.ksize)
