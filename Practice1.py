import cv2
import matplotlib.pyplot as plt

def sobel_x_edge_detection(image_path: str, ksize: int = 3):
    """
    Load an image in grayscale, apply the Sobel operator in the x-direction,
    and display the original and edge-detected images side by side.

    Parameters
    ----------
    image_path : str
        Path to the input image file.
    ksize : int, optional
        Size of the Sobel kernel (must be odd), by default 3.
    """
    # 1. Load image in grayscale
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not load image at '{image_path}'")

    # 2. Apply Sobel operator (x-direction)
    #    cv2.CV_64F ensures we capture negative gradients before scaling
    sobelx = cv2.Sobel(gray, cv2.CV_64F, dx=1, dy=0, ksize=ksize)
    sobelx_abs = cv2.convertScaleAbs(sobelx)

    # 3. Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(gray, cmap='gray')
    axes[0].set_title("Original (Grayscale)")
    axes[0].axis("off")

    axes[1].imshow(sobelx_abs, cmap='gray')
    axes[1].set_title(f"Sobel X (ksize={ksize})")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Apply Sobel edge detection in the x-direction."
    )
    parser.add_argument(
        "image_path",
        help="Path to the input image (e.g. 'Test.jpg')"
    )
    parser.add_argument(
        "--ksize", "-k",
        type=int,
        default=3,
        help="Sobel kernel size (odd integer, default=3)"
    )
    args = parser.parse_args()

    sobel_x_edge_detection(args.image_path, ksize=args.ksize)
