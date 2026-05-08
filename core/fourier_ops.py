from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .image_loader import to_gray_u8
from .utils import ImageValidationError, ensure_uint8_image, normalize_to_uint8


@dataclass(frozen=True)
class FourierData:
    gray_u8: np.ndarray
    f_complex: np.ndarray  # FFT (unshifted)
    f_shift: np.ndarray  # FFT shifted
    magnitude: np.ndarray  # float32 magnitude of shifted
    spectrum_u8: np.ndarray  # uint8 log-magnitude for display


def fft2_gray(gray_u8: np.ndarray) -> FourierData:
    """Compute FFT2 for grayscale uint8 image and return spectrum for visualization."""
    ensure_uint8_image(gray_u8, name="gray")
    if gray_u8.ndim != 2:
        raise ImageValidationError("fft2_gray expects a 2D grayscale image.")
    x = gray_u8.astype(np.float32)
    f = np.fft.fft2(x)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift).astype(np.float32)
    # log visualization: log(1+mag)
    log_mag = np.log1p(mag)
    spectrum_u8 = normalize_to_uint8(log_mag)
    return FourierData(
        gray_u8=gray_u8,
        f_complex=f,
        f_shift=fshift,
        magnitude=mag,
        spectrum_u8=spectrum_u8,
    )


def fft2_from_image(gray_or_rgb_u8: np.ndarray) -> FourierData:
    ensure_uint8_image(gray_or_rgb_u8, name="image")
    gray = gray_or_rgb_u8 if gray_or_rgb_u8.ndim == 2 else to_gray_u8(gray_or_rgb_u8)
    return fft2_gray(gray)


def apply_frequency_filter(
    gray_u8: np.ndarray, filter_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a frequency-domain filter to a grayscale image.

    Args:
        gray_u8: uint8 grayscale image
        filter_mask: float mask with shape (H,W) applied on shifted FFT

    Returns:
        restored_u8: uint8 spatial-domain reconstruction
        filtered_spectrum_u8: uint8 log-magnitude spectrum after filtering
    """
    ensure_uint8_image(gray_u8, name="gray")
    if gray_u8.ndim != 2:
        raise ImageValidationError("apply_frequency_filter expects grayscale image.")
    if filter_mask is None or not isinstance(filter_mask, np.ndarray):
        raise ImageValidationError("filter_mask must be a numpy array.")
    if filter_mask.shape != gray_u8.shape:
        raise ImageValidationError("filter_mask shape must match image shape.")

    f = np.fft.fft2(gray_u8.astype(np.float32))
    fshift = np.fft.fftshift(f)
    fshift_f = fshift * filter_mask.astype(np.float32)
    mag = np.abs(fshift_f).astype(np.float32)
    spec = normalize_to_uint8(np.log1p(mag))
    ishift = np.fft.ifftshift(fshift_f)
    img_back = np.fft.ifft2(ishift)
    img_back = np.real(img_back)
    restored = np.clip(img_back, 0, 255).astype(np.uint8)
    return restored, spec


def center_coords(shape: Tuple[int, int]) -> Tuple[int, int]:
    h, w = int(shape[0]), int(shape[1])
    return h // 2, w // 2

