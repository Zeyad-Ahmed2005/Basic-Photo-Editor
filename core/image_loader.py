from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

from .utils import ImageInfo, ImageValidationError, ensure_uint8_image, get_image_info


@dataclass(frozen=True)
class LoadedImage:
    """Holds an uploaded image in a normalized format."""

    rgb: np.ndarray  # uint8, HxWx3, RGB
    info: ImageInfo


def load_image_from_bytes(data: bytes) -> LoadedImage:
    """
    Load image bytes (JPG/PNG/BMP) using Pillow and return a normalized uint8 RGB image.

    Raises:
        ImageValidationError: if the bytes cannot be decoded to an image.
    """
    if not data:
        raise ImageValidationError("No image data provided.")
    try:
        img = Image.open(BytesIO(data))
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        rgb = np.array(img, dtype=np.uint8)
        ensure_uint8_image(rgb, name="uploaded_rgb")
        info = get_image_info(rgb)
        return LoadedImage(rgb=rgb, info=info)
    except Exception as e:  # noqa: BLE001
        raise ImageValidationError(f"Failed to load image: {e}") from e


def to_gray_u8(rgb: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB image to uint8 grayscale (luma)."""
    ensure_uint8_image(rgb, name="rgb")
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ImageValidationError("Expected RGB image for grayscale conversion.")
    # ITU-R BT.601 luma
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(y, 0, 255).astype(np.uint8)

