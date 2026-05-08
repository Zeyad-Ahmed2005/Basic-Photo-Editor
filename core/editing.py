from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .utils import ImageValidationError, crop_bounds, ensure_uint8_image


def crop(rgb_u8: np.ndarray, *, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Crop RGB image using (x1,y1) inclusive and (x2,y2) exclusive bounds."""
    ensure_uint8_image(rgb_u8, name="image")
    if rgb_u8.ndim != 3 or rgb_u8.shape[2] < 3:
        raise ImageValidationError("Crop expects an RGB image.")
    h, w = rgb_u8.shape[:2]
    x1c, y1c, x2c, y2c = crop_bounds(x1, y1, x2, y2, width=w, height=h)
    return rgb_u8[y1c:y2c, x1c:x2c, :3].copy()


def rotate(rgb_u8: np.ndarray, *, angle_deg: float, keep_size: bool = True) -> np.ndarray:
    """
    Rotate image by angle degrees around center.

    If keep_size=False, expands canvas to avoid clipping.
    """
    ensure_uint8_image(rgb_u8, name="image")
    if rgb_u8.ndim != 3 or rgb_u8.shape[2] < 3:
        raise ImageValidationError("Rotate expects an RGB image.")
    h, w = rgb_u8.shape[:2]
    center = (w / 2.0, h / 2.0)
    m = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)

    if keep_size:
        out = cv2.warpAffine(
            rgb_u8,
            m,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT,
        )
        return out.astype(np.uint8)

    cos = abs(m[0, 0])
    sin = abs(m[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    m[0, 2] += (new_w / 2.0) - center[0]
    m[1, 2] += (new_h / 2.0) - center[1]
    out = cv2.warpAffine(
        rgb_u8,
        m,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT,
    )
    return out.astype(np.uint8)


def resize(
    rgb_u8: np.ndarray, *, width: int, height: int, keep_aspect: bool = True
) -> np.ndarray:
    """Resize with optional aspect preservation."""
    ensure_uint8_image(rgb_u8, name="image")
    if rgb_u8.ndim != 3 or rgb_u8.shape[2] < 3:
        raise ImageValidationError("Resize expects an RGB image.")
    h0, w0 = rgb_u8.shape[:2]
    w = int(width)
    h = int(height)
    if w <= 0 or h <= 0:
        raise ImageValidationError("Width and height must be positive.")
    if keep_aspect:
        aspect = w0 / float(h0)
        # fit into the requested box (w,h) while preserving aspect
        if w / float(h) > aspect:
            w = int(round(h * aspect))
        else:
            h = int(round(w / aspect))
        w = max(1, w)
        h = max(1, h)
    out = cv2.resize(rgb_u8, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return out.astype(np.uint8)


def brightness_contrast(
    rgb_u8: np.ndarray, *, brightness: int = 0, contrast: float = 1.0
) -> np.ndarray:
    """
    Adjust brightness and contrast using y = contrast*x + brightness.
    brightness: [-255, 255]
    contrast: [0.0, 3.0] typical
    """
    ensure_uint8_image(rgb_u8, name="image")
    if rgb_u8.ndim != 3 or rgb_u8.shape[2] < 3:
        raise ImageValidationError("Brightness/contrast expects an RGB image.")
    b = int(brightness)
    c = float(contrast)
    out = rgb_u8.astype(np.float32) * c + b
    return np.clip(out, 0, 255).astype(np.uint8)

