#!/usr/bin/env python3
"""
sobel_xy_combined.py

Loads one or more images in grayscale, applies the Sobel operator in both
x and y directions, combines the results, and displays original, Sobel X,
Sobel Y, and combined edge maps side by side for each image.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

def sobel_combined(image_gray: np.ndarray, ksize: int = 3):
    """
    Compute Sobel X, Sobel Y, and their combined edge map.

    Parameters
    ----------
    image_gray : np.ndarray
        Grayscale input image.
    ksize : int
        Sobel kernel size (must be odd).

    Returns
    -------
    sobelx : np.ndarray
        Absolute Sobel X response.
    sobely : np.ndarray
        Absolute Sobel Y response.
    combined : np.ndarray
        Weighted combination of sobelx and sobely.
    """
    # Sobel in X and Y
    sobelx = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=ksize)

    # Convert to 8‑bit absolute values
    abs_sobelx = np.uint8(np.absolute(sobelx))
    abs_sobely = np.uint8(np.absolute(sobely))

    # Combine with equal weighting
    combined = cv2.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)

    return abs_sobelx, abs_sobely, combined

def main(image_paths, ksize):
    # Number of images
    n = len(image_paths)

    # Create a figure with n rows and 4 columns
    fig, axes = plt.subplots(n, 4, figsize=(4*4, 4*n))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    for row, img_path in enumerate(image_paths):
        # Load grayscale
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"Error: could not load '{img_path}'", file=sys.stderr)
            continue

        # Compute Sobel responses
        sobelx, sobely, combined = sobel_combined(gray, ksize)

        # Titles for this row
        titles = [
            f"{img_path}\nOriginal",
            "Sobel X",
            "Sobel Y",
            "Combined"
        ]
        images = [gray, sobelx, sobely, combined]

        # Plot each of the 4
        for col in range(4):
            ax = axes[row, col]
            ax.imshow(images[col], cmap='gray')
            ax.set_title(titles[col])
            ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply Sobel X, Sobel Y and combined edge detection on one or more images."
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="Paths to input images (e.g. test.jpg Test2.jpg)"
    )
    parser.add_argument(
        "--ksize", "-k",
        type=int,
        default=3,
        help="Sobel kernel size (odd integer, default=3)"
    )

    args = parser.parse_args()
    main(args.images, args.ksize)
