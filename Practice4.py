#!/usr/bin/env python3
"""
sobel_edge_direction.py

Loads one or more images in grayscale, computes the edge direction
using the arctangent of the Sobel derivatives, and visualizes the
direction map alongside the original images.

Usage:
    python sobel_edge_direction.py image1.jpg image2.jpg [--ksize KSIZE]

Example:
    python sobel_edge_direction.py /mnt/data/Test.jpg /mnt/data/Test2.jpg --ksize 5
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

def edge_direction(image_gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Compute edge direction map of a grayscale image via Sobel derivatives.

    Parameters
    ----------
    image_gray : np.ndarray
        Single‑channel grayscale image.
    ksize : int
        Sobel kernel size (must be odd), default 3.

    Returns
    -------
    np.ndarray
        2D array of edge directions (degrees in [0,360)).
    """
    # Compute Sobel derivatives
    dx = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=ksize)
    dy = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=ksize)

    # Compute angle in radians and convert to degrees
    angle = np.arctan2(dy, dx)            # range [-π, π]
    angle_deg = np.degrees(angle)         # range [-180, 180]
    angle_deg = (angle_deg + 360) % 360   # map to [0, 360)
    return angle_deg

def visualize_edge_direction(image_paths, ksize):
    """
    Display each original grayscale image next to its edge-direction map.

    Parameters
    ----------
    image_paths : list of str
        Paths to the images to process.
    ksize : int
        Sobel kernel size for derivative calculation.
    """
    n = len(image_paths)
    fig, axes = plt.subplots(n, 2, figsize=(10, 5 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    for idx, path in enumerate(image_paths):
        # Load as grayscale
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"Error: could not load '{path}'", file=sys.stderr)
            continue

        # Compute edge direction
        dir_map = edge_direction(gray, ksize=ksize)

        # Show original
        ax0 = axes[idx, 0]
        ax0.imshow(gray, cmap='gray')
        ax0.set_title(f"{path}\nOriginal")
        ax0.axis('off')

        # Show direction map
        ax1 = axes[idx, 1]
        im = ax1.imshow(dir_map, cmap='hsv')
        ax1.set_title(f"{path}\nEdge Direction (ksize={ksize})")
        ax1.axis('off')
        fig.colorbar(im, ax=ax1, orientation='vertical',
                     fraction=0.046, pad=0.04, label='Direction (°)')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize edge direction via Sobel arctangent on multiple images."
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="Paths to input images (e.g. Test.jpg Test2.jpg)"
    )
    parser.add_argument(
        "--ksize", "-k",
        type=int,
        default=3,
        help="Sobel kernel size (odd integer, default=3)"
    )
    args = parser.parse_args()

    visualize_edge_direction(args.images, args.ksize)
