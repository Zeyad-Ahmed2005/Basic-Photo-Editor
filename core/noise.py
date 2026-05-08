from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .utils import ImageValidationError, ensure_uint8_image


def add_salt_and_pepper(
    rgb_u8: np.ndarray, *, amount: float = 0.03, salt_vs_pepper: float = 0.5, seed: int | None = None
) -> np.ndarray:
    """
    Add salt-and-pepper noise to an RGB uint8 image.

    Args:
        amount: fraction of pixels to corrupt in [0,1]
        salt_vs_pepper: fraction of corrupted pixels that become salt (white)
    """
    ensure_uint8_image(rgb_u8, name="image")
    if rgb_u8.ndim != 3 or rgb_u8.shape[2] < 3:
        raise ImageValidationError("Salt-and-pepper noise expects an RGB image.")
    a = float(amount)
    if not (0.0 <= a <= 1.0):
        raise ImageValidationError("amount must be in [0,1].")
    sp = float(salt_vs_pepper)
    if not (0.0 <= sp <= 1.0):
        raise ImageValidationError("salt_vs_pepper must be in [0,1].")

    rng = np.random.default_rng(seed)
    out = rgb_u8.copy()
    h, w, _ = out.shape
    n = int(a * h * w)
    if n <= 0:
        return out

    ys = rng.integers(0, h, size=n, endpoint=False)
    xs = rng.integers(0, w, size=n, endpoint=False)
    salt_mask = rng.random(size=n) < sp
    out[ys[salt_mask], xs[salt_mask], :3] = 255
    out[ys[~salt_mask], xs[~salt_mask], :3] = 0
    return out


def add_periodic_noise(
    rgb_u8: np.ndarray,
    *,
    frequency: float = 8.0,
    amplitude: float = 35.0,
    orientation_deg: float = 45.0,
) -> np.ndarray:
    """
    Add periodic (sinusoidal) noise to an RGB image.

    The noise pattern is: A * sin(2*pi*f*(x*cos + y*sin)/max_dim)
    """
    ensure_uint8_image(rgb_u8, name="image")
    if rgb_u8.ndim != 3 or rgb_u8.shape[2] < 3:
        raise ImageValidationError("Periodic noise expects an RGB image.")

    f = max(0.0, float(frequency))
    a = float(amplitude)
    theta = np.deg2rad(float(orientation_deg))
    h, w, _ = rgb_u8.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # project coordinate along orientation
    proj = xx * np.cos(theta) + yy * np.sin(theta)
    denom = float(max(h, w))
    noise = a * np.sin(2.0 * np.pi * f * proj / denom)
    noise3 = np.repeat(noise[..., None], 3, axis=2)
    out = rgb_u8.astype(np.float32) + noise3
    return np.clip(out, 0, 255).astype(np.uint8)

