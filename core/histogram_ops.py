from __future__ import annotations

from typing import Dict, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .image_loader import to_gray_u8
from .utils import ImageValidationError, ensure_uint8_image


def compute_hist_gray(gray_u8: np.ndarray, bins: int = 256) -> np.ndarray:
    """Compute grayscale histogram (uint8 gray). Returns counts length=bins."""
    ensure_uint8_image(gray_u8, name="gray")
    if gray_u8.ndim != 2:
        raise ImageValidationError("compute_hist_gray expects a 2D grayscale image.")
    hist, _ = np.histogram(gray_u8.ravel(), bins=bins, range=(0, 256))
    return hist.astype(np.int64)


def compute_hist_rgb(rgb_u8: np.ndarray, bins: int = 256) -> Dict[str, np.ndarray]:
    """Compute per-channel histogram for uint8 RGB. Returns dict with keys r,g,b."""
    ensure_uint8_image(rgb_u8, name="rgb")
    if rgb_u8.ndim != 3 or rgb_u8.shape[2] < 3:
        raise ImageValidationError("compute_hist_rgb expects an RGB image.")
    out: Dict[str, np.ndarray] = {}
    for name, idx in (("r", 0), ("g", 1), ("b", 2)):
        h, _ = np.histogram(rgb_u8[..., idx].ravel(), bins=bins, range=(0, 256))
        out[name] = h.astype(np.int64)
    return out


def fig_hist_gray(gray_u8: np.ndarray, title: str) -> plt.Figure:
    """Matplotlib figure for grayscale histogram."""
    hist = compute_hist_gray(gray_u8)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(hist, color="black", linewidth=1.5)
    ax.set_xlim([0, 255])
    ax.set_title(title)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def fig_hist_rgb(rgb_u8: np.ndarray, title: str) -> plt.Figure:
    """Matplotlib figure for RGB histograms."""
    h = compute_hist_rgb(rgb_u8)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(h["r"], color="red", linewidth=1.2, label="R")
    ax.plot(h["g"], color="green", linewidth=1.2, label="G")
    ax.plot(h["b"], color="blue", linewidth=1.2, label="B")
    ax.set_xlim([0, 255])
    ax.set_title(title)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def equalize_histogram(gray_or_rgb_u8: np.ndarray, *, mode: str = "ycrcb") -> np.ndarray:
    """
    Histogram equalization.

    - If grayscale: OpenCV equalizeHist.
    - If color: equalize luminance/value channel only.

    Args:
        gray_or_rgb_u8: uint8 image, 2D or 3D RGB.
        mode: 'ycrcb' or 'hsv'
    """
    ensure_uint8_image(gray_or_rgb_u8, name="image")
    if gray_or_rgb_u8.ndim == 2:
        return cv2.equalizeHist(gray_or_rgb_u8)

    rgb = gray_or_rgb_u8[..., :3]
    if mode.lower() == "hsv":
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        v_eq = cv2.equalizeHist(v)
        hsv_eq = cv2.merge([h, s, v_eq])
        out = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2RGB)
        return out.astype(np.uint8)

    # default: YCrCb luminance equalization
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    ycrcb_eq = cv2.merge([y_eq, cr, cb])
    out = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2RGB)
    return out.astype(np.uint8)


def get_gray_and_hist(rgb_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience for UI: grayscale + histogram."""
    gray = to_gray_u8(rgb_u8)
    hist = compute_hist_gray(gray)
    return gray, hist

