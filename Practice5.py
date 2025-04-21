#!/usr/bin/env python3
"""
sobel_line_extraction.py

Extract vertical or horizontal lines from one or more grayscale images
using the Sobel operator plus thresholding, and display results.

Usage:
    python sobel_line_extraction.py image1.jpg image2.jpg [--ksize KSIZE] [--thresh THRESH]

Example:
    python sobel_line_extraction.py /mnt/data/Test.jpg /mnt/data/Test2.jpg --ksize 3 --thresh 100
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

def extract_sobel_lines(
    image_gray: np.ndarray,
    direction: str = 'vertical',
    ksize: int = 3,
    thresh: int = 100
) -> np.ndarray:
    """
    Extract vertical or horizontal lines using Sobel + threshold.

    Parameters
    ----------
    image_gray : np.ndarray
        Grayscale input image.
    direction : str
        'vertical' or 'horizontal'.
    ksize : int
        Sobel kernel size (odd integer).
    thresh : int
        Threshold for binary line mask.

    Returns
    -------
    np.ndarray
        Binary mask of extracted lines (0 or 255).
    """
    if direction == 'vertical':
        sobel = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=ksize)
    else:
        sobel = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=ksize)

    abs_sobel = np.uint8(np.absolute(sobel))
    _, binary = cv2.threshold(abs_sobel, thresh, 255, cv2.THRESH_BINARY)
    return binary

def demo_line_extraction(
    image_paths: list[str],
    ksize: int,
    thresh: int
):
    """
    Apply vertical and horizontal Sobel line extraction to each image
    and display side by side.

    Parameters
    ----------
    image_paths : list of str
        Paths to the input images.
    ksize : int
        Sobel kernel size.
    thresh : int
        Threshold value for binary mask.
    """
    n = len(image_paths)
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    for i, path in enumerate(image_paths):
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"Error: cannot load '{path}'", file=sys.stderr)
            continue

        vert = extract_sobel_lines(gray, 'vertical', ksize, thresh)
        hori = extract_sobel_lines(gray, 'horizontal', ksize, thresh)

        # Original
        ax = axes[i, 0]
        ax.imshow(gray, cmap='gray')
        ax.set_title(f"{path}\nOriginal")
        ax.axis('off')

        # Vertical lines
        ax = axes[i, 1]
        ax.imshow(vert, cmap='gray')
        ax.set_title("Vertical Lines")
        ax.axis('off')

        # Horizontal lines
        ax = axes[i, 2]
        ax.imshow(hori, cmap='gray')
        ax.set_title("Horizontal Lines")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract vertical/horizontal lines from images using Sobel."
    )
    parser.add_argument(
        "images",
        nargs=2,
        help="Two image paths to process (e.g. Test.jpg Test2.jpg)"
    )
    parser.add_argument(
        "--ksize", "-k",
        type=int,
        default=3,
        help="Sobel kernel size (odd integer, default=3)"
    )
    parser.add_argument(
        "--thresh", "-t",
        type=int,
        default=100,
        help="Threshold for binary mask (default=100)"
    )

    args = parser.parse_args()
    demo_line_extraction(args.images, args.ksize, args.thresh)
