from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


class ImageValidationError(ValueError):
    """Raised when an operation is requested on an invalid or empty image."""


def ensure_uint8_image(img: np.ndarray, *, name: str = "image") -> np.ndarray:
    """Validate image is a non-empty uint8 numpy array and return it."""
    if img is None:
        raise ImageValidationError(f"{name} is None.")
    if not isinstance(img, np.ndarray):
        raise ImageValidationError(f"{name} must be a numpy array.")
    if img.size == 0:
        raise ImageValidationError(f"{name} is empty.")
    if img.dtype != np.uint8:
        raise ImageValidationError(f"{name} must be uint8, got {img.dtype}.")
    if img.ndim not in (2, 3):
        raise ImageValidationError(f"{name} must be 2D or 3D, got {img.ndim}D.")
    if img.ndim == 3 and img.shape[2] not in (3, 4):
        raise ImageValidationError(
            f"{name} must have 3 (RGB) or 4 (RGBA) channels, got {img.shape[2]}."
        )
    return img


def as_float32(img: np.ndarray) -> np.ndarray:
    """Convert uint8 image to float32 in [0, 1]."""
    ensure_uint8_image(img)
    return img.astype(np.float32) / 255.0


def float_to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert float image (any range) to uint8 with clipping to [0, 255]."""
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        raise ImageValidationError("Cannot convert empty image.")
    out = np.clip(img, 0.0, 255.0)
    if out.dtype != np.uint8:
        out = out.astype(np.uint8)
    return out


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """Normalize an array to uint8 for visualization (min-max)."""
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        raise ImageValidationError("Cannot normalize empty image.")
    x = img.astype(np.float32)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < 1e-12:
        return np.zeros_like(x, dtype=np.uint8)
    x = (x - mn) / (mx - mn)
    return (x * 255.0).clip(0, 255).astype(np.uint8)


def ensure_odd_kernel(ksize: int, *, min_size: int = 3, name: str = "kernel") -> int:
    """Ensure ksize is an odd integer >= min_size."""
    if not isinstance(ksize, int):
        raise ImageValidationError(f"{name} size must be int.")
    if ksize < min_size:
        raise ImageValidationError(f"{name} size must be >= {min_size}.")
    if ksize % 2 == 0:
        raise ImageValidationError(f"{name} size must be odd.")
    return ksize


def clamp_int(v: int, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(v))))


def image_hash_uint8(img: np.ndarray) -> str:
    """Stable hash for caching based on bytes + shape."""
    ensure_uint8_image(img)
    h = hashlib.sha256()
    h.update(str(img.shape).encode("utf-8"))
    h.update(img.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class ImageInfo:
    height: int
    width: int
    channels: int


def get_image_info(img: np.ndarray) -> ImageInfo:
    ensure_uint8_image(img)
    if img.ndim == 2:
        return ImageInfo(height=int(img.shape[0]), width=int(img.shape[1]), channels=1)
    return ImageInfo(
        height=int(img.shape[0]), width=int(img.shape[1]), channels=int(img.shape[2])
    )


def crop_bounds(
    x1: int, y1: int, x2: int, y2: int, *, width: int, height: int
) -> Tuple[int, int, int, int]:
    """Validate and clamp crop rectangle. Returns (x1,y1,x2,y2) with x2>x1, y2>y1."""
    x1c = clamp_int(x1, 0, width - 1)
    y1c = clamp_int(y1, 0, height - 1)
    x2c = clamp_int(x2, 1, width)
    y2c = clamp_int(y2, 1, height)
    if x2c <= x1c or y2c <= y1c:
        raise ImageValidationError("Invalid crop rectangle (must have positive area).")
    return x1c, y1c, x2c, y2c

