from __future__ import annotations

import cv2
import numpy as np

from .image_loader import to_gray_u8
from .utils import ensure_odd_kernel, ensure_uint8_image, normalize_to_uint8


def sobel(gray_or_rgb_u8: np.ndarray, *, direction: str, ksize: int = 3) -> np.ndarray:
    """
    Sobel edge filter.

    Args:
        direction: 'x', 'y', or 'mag'
        ksize: odd kernel size (>=3). OpenCV supports 1,3,5,7...
    """
    ensure_uint8_image(gray_or_rgb_u8, name="image")
    k = ensure_odd_kernel(int(ksize), min_size=3, name="Sobel ksize")
    gray = gray_or_rgb_u8 if gray_or_rgb_u8.ndim == 2 else to_gray_u8(gray_or_rgb_u8)

    dx = 1 if direction.lower() in ("x", "mag") else 0
    dy = 1 if direction.lower() in ("y", "mag") else 0
    if direction.lower() not in ("x", "y", "mag"):
        raise ValueError("direction must be 'x', 'y', or 'mag'.")

    if direction.lower() == "x":
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=k)
        return normalize_to_uint8(np.abs(gx))
    if direction.lower() == "y":
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=k)
        return normalize_to_uint8(np.abs(gy))

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=k)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=k)
    mag = cv2.magnitude(gx, gy)
    return normalize_to_uint8(mag)


def laplacian(
    gray_or_rgb_u8: np.ndarray, *, ksize: int = 3, scale: float = 1.0, delta: float = 0.0
) -> np.ndarray:
    """Laplacian filter with adjustable ksize/scale/delta."""
    ensure_uint8_image(gray_or_rgb_u8, name="image")
    k = ensure_odd_kernel(int(ksize), min_size=1, name="Laplacian ksize")
    gray = gray_or_rgb_u8 if gray_or_rgb_u8.ndim == 2 else to_gray_u8(gray_or_rgb_u8)
    out = cv2.Laplacian(gray, cv2.CV_32F, ksize=k, scale=float(scale), delta=float(delta))
    return normalize_to_uint8(np.abs(out))

