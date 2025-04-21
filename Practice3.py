#!/usr/bin/env python3
"""
gradient_magnitude_demo.py

Defines a function that computes the gradient magnitude of a grayscale image
using the Sobel operator, and applies it to two example images.
Displays each original and its gradient magnitude side by side.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys


def gradient_magnitude(image_gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Compute the gradient magnitude of a grayscale image using the Sobel operator.

    Parameters
    ----------
    image_gray : np.ndarray
        Input image in grayscale (single channel).
    ksize : int
        Size of the Sobel kernel (must be odd), default is 3.

    Returns
    -------
    np.ndarray
        8‑bit image of gradient magnitudes scaled to [0,255].
    """
    # 1. Compute Sobel derivatives in X and Y
    sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=ksize)

    # 2. Compute gradient magnitude: sqrt(dx^2 + dy^2)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # 3. Normalize to 0–255 and convert to uint8
    magnitude = np.clip((magnitude / magnitude.max()) * 255, 0, 255)
    return magnitude.astype(np.uint8)


def main(image_paths, ksize):
    n = len(image_paths)
    # Prepare a figure with n rows and 2 columns
    fig, axes = plt.subplots(n, 2, figsize=(10, 5 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    for idx, path in enumerate(image_paths):
        # Load as grayscale
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"Error: Could not load image '{path}'", file=sys.stderr)
            continue

        # Compute gradient magnitude
        grad_mag = gradient_magnitude(gray, ksize=ksize)

        # Show original
        ax_orig = axes[idx, 0]
        ax_orig.imshow(gray, cmap='gray')
        ax_orig.set_title(f"{path} — Original")
        ax_orig.axis('off')

        # Show gradient magnitude
        ax_grad = axes[idx, 1]
        ax_grad.imshow(grad_mag, cmap='gray')
        ax_grad.set_title(f"{path} — Gradient Magnitude (ksize={ksize})")
        ax_grad.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and display Sobel gradient magnitude for two images."
    )
    parser.add_argument(
        "images",
        nargs=2,
        help="Two image file paths to process (e.g. Test.jpg Test2.jpg)"
    )
    parser.add_argument(
        "--ksize", "-k",
        type=int,
        default=3,
        help="Sobel kernel size (odd integer, default=3)"
    )
    args = parser.parse_args()

    main(args.images, args.ksize)
