# Sobel-Projects
Sobel Projects
# Sobel and Scharr Edge Detection Projects  

This repository contains **six Python scripts** demonstrating various edge detection techniques using OpenCV’s Sobel and Scharr filters. From simple directional gradients to edge magnitude, orientation, line extraction, and filter comparisons, these tools help you explore and visualize image gradients.

---

## Projects Overview

| Project | Script | Description | Usage Example |
|---------|--------|-------------|---------------|
| 1. Sobel X | `sobel_x_edge_detection.py` | Applies the Sobel operator in the x-direction to highlight vertical edges. | `python sobel_x_edge_detection.py Test.jpg --ksize 3` |
| 2. Sobel X+Y | `sobel_xy_combined.py` | Applies Sobel in both x & y, then combines the two results into a single edge map. | `python sobel_xy_combined.py Test.jpg Test2.jpg --ksize 3` |
| 3. Gradient Magnitude | `gradient_magnitude_demo.py` | Defines `gradient_magnitude()` to compute √(dx²+dy²) using Sobel, displays original vs. magnitude. | `python gradient_magnitude_demo.py Test.jpg Test2.jpg --ksize 3` |
| 4. Edge Direction | `sobel_edge_direction.py` | Computes pixel-wise edge orientation (in degrees) via `atan2(dy,dx)` and visualizes it with HSV colormap. | `python sobel_edge_direction.py Test.jpg Test2.jpg --ksize 3` |
| 5. Line Extraction | `sobel_line_extraction.py` | Extracts vertical or horizontal lines by thresholding the Sobel derivative in a single direction. | `python sobel_line_extraction.py Test.jpg Test2.jpg --ksize 3 --thresh 100` |
| 6. Sobel vs. Scharr | `sobel_vs_scharr_demo.py` | Compares gradient magnitudes computed by Sobel vs. Scharr filters side-by-side. | `python sobel_vs_scharr_demo.py Test.jpg Test2.jpg --ksize 3` |

---

## Prerequisites

- Python 3.7 or higher
- OpenCV: `pip install opencv-python`
- NumPy: `pip install numpy`
- Matplotlib: `pip install matplotlib`

## Installation

```bash
git clone <repository_url>
cd <repository_folder>
python -m venv venv
# Activate the virtual environment
# Windows PowerShell: .\venv\Scripts\activate
# Linux/macOS: source venv/bin/activate
pip install opencv-python numpy matplotlib
```

## Usage Tips

- **Kernel Size (`--ksize`)**: Must be an odd integer (e.g., 3, 5, 7). Larger kernels detect broader gradients but are slower.
- **Threshold (`--thresh`)**: For line extraction, tune this value to include or exclude weaker edges.
- **Display**: Scripts use Matplotlib to show results inline or in a pop-up window; close windows to proceed.

## Contributing

Feel free to fork and extend these scripts. Suggestions for additional edge filters, custom thresholding, or performance optimizations are welcome!

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

