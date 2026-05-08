from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from skimage.feature import peak_local_max

from .fourier_ops import apply_frequency_filter, center_coords, fft2_gray
from .image_loader import to_gray_u8
from .utils import ImageValidationError, ensure_odd_kernel, ensure_uint8_image, normalize_to_uint8


@dataclass(frozen=True)
class PeakDetectionResult:
    peaks: np.ndarray  # (N,2) rows of (y,x) in shifted FFT coordinates
    paired_peaks: List[Tuple[Tuple[int, int], Tuple[int, int]]]  # ((y,x),(ys,xs))
    spectrum_u8: np.ndarray  # display spectrum used for detection


def median_filter(rgb_u8: np.ndarray, *, ksize: int = 3) -> np.ndarray:
    """Median filter for salt-and-pepper noise removal (applies per-channel)."""
    ensure_uint8_image(rgb_u8, name="image")
    k = ensure_odd_kernel(int(ksize), min_size=3, name="Median ksize")
    if rgb_u8.ndim != 3 or rgb_u8.shape[2] < 3:
        raise ImageValidationError("Median filter expects an RGB image.")
    out = np.empty_like(rgb_u8[..., :3])
    for c in range(3):
        out[..., c] = cv2.medianBlur(rgb_u8[..., c], k)
    return out.astype(np.uint8)


def _suppress_center(mag: np.ndarray, *, radius: int) -> np.ndarray:
    h, w = mag.shape
    cy, cx = center_coords((h, w))
    yy, xx = np.ogrid[:h, :w]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
    out = mag.copy()
    out[mask] = 0.0
    return out


def detect_periodic_peaks(
    gray_or_rgb_u8: np.ndarray,
    *,
    min_distance: int = 12,
    threshold_rel: float = 0.35,
    max_peaks: int = 16,
    center_suppression_radius: int | None = None,
) -> PeakDetectionResult:
    """
    Detect periodic noise peaks from the FFT magnitude spectrum.

    Strategy:
    - Compute shifted FFT magnitude.
    - Suppress center low frequencies with a circular mask.
    - Run local maxima detection on log-magnitude image.
    - Pair peaks with their symmetric counterparts around the center.
    """
    ensure_uint8_image(gray_or_rgb_u8, name="image")
    gray = gray_or_rgb_u8 if gray_or_rgb_u8.ndim == 2 else to_gray_u8(gray_or_rgb_u8)
    fd = fft2_gray(gray)
    mag = fd.magnitude
    h, w = mag.shape
    if center_suppression_radius is None:
        center_suppression_radius = int(0.06 * min(h, w))  # ~6% of min dim
    mag2 = _suppress_center(mag, radius=int(center_suppression_radius))
    log_mag = np.log1p(mag2).astype(np.float32)
    # Normalize for robust thresholding
    log_u8 = normalize_to_uint8(log_mag)

    coords = peak_local_max(
        log_mag,
        min_distance=int(min_distance),
        threshold_rel=float(threshold_rel),
        num_peaks=int(max_peaks),
        exclude_border=False,
    )
    coords = coords.astype(int)

    cy, cx = center_coords((h, w))
    coord_set = {(int(y), int(x)) for y, x in coords}
    paired: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    used: set[tuple[int, int]] = set()

    for (y, x) in sorted(coord_set, key=lambda p: float(log_mag[p[0], p[1]]), reverse=True):
        if (y, x) in used:
            continue
        ys = int(2 * cy - y)
        xs = int(2 * cx - x)
        if (ys, xs) in coord_set and (ys, xs) not in used and (ys, xs) != (y, x):
            paired.append(((y, x), (ys, xs)))
            used.add((y, x))
            used.add((ys, xs))

    return PeakDetectionResult(peaks=coords, paired_peaks=paired, spectrum_u8=log_u8)


def gaussian_notch_reject_mask(
    shape: Tuple[int, int],
    peak_pairs: Sequence[Tuple[Tuple[int, int], Tuple[int, int]]],
    *,
    sigma: float = 6.0,
) -> np.ndarray:
    """
    Build a multiplicative Gaussian notch reject mask in shifted FFT coordinates.
    Mask values near 0 at peaks and ~1 elsewhere.
    """
    h, w = int(shape[0]), int(shape[1])
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    mask = np.ones((h, w), dtype=np.float32)
    s2 = float(sigma) ** 2
    s2 = max(s2, 1e-6)

    for (p1, p2) in peak_pairs:
        for (y0, x0) in (p1, p2):
            d2 = (yy - float(y0)) ** 2 + (xx - float(x0)) ** 2
            notch = 1.0 - np.exp(-d2 / (2.0 * s2))
            mask *= notch
    return np.clip(mask, 0.0, 1.0)


def remove_periodic_notch_auto(
    gray_or_rgb_u8: np.ndarray,
    *,
    sigma: float = 6.0,
    min_distance: int = 12,
    threshold_rel: float = 0.35,
    max_peaks: int = 16,
) -> tuple[np.ndarray, PeakDetectionResult, np.ndarray]:
    """
    Automatic notch filtering:
    - detect peak pairs
    - build Gaussian notch reject mask
    - reconstruct image

    Returns:
      restored_gray_u8, detection_result, filtered_spectrum_u8
    """
    ensure_uint8_image(gray_or_rgb_u8, name="image")
    gray = gray_or_rgb_u8 if gray_or_rgb_u8.ndim == 2 else to_gray_u8(gray_or_rgb_u8)
    det = detect_periodic_peaks(
        gray,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
        max_peaks=max_peaks,
    )
    mask = gaussian_notch_reject_mask(gray.shape, det.paired_peaks, sigma=float(sigma))
    restored, filtered_spec = apply_frequency_filter(gray, mask)
    return restored, det, filtered_spec


def remove_periodic_band_reject_auto(
    gray_or_rgb_u8: np.ndarray,
    *,
    bandwidth: float = 10.0,
    sigma: float = 2.5,
    min_distance: int = 12,
    threshold_rel: float = 0.35,
    max_peaks: int = 16,
) -> tuple[np.ndarray, PeakDetectionResult, np.ndarray]:
    """
    Automatic band-reject filtering:
    - detect peak pairs
    - estimate noisy radii from detected peaks
    - build a smooth radial band-reject mask centered at each radius

    This is a pragmatic band-reject approach (frequency-domain annular rejection).
    """
    ensure_uint8_image(gray_or_rgb_u8, name="image")
    gray = gray_or_rgb_u8 if gray_or_rgb_u8.ndim == 2 else to_gray_u8(gray_or_rgb_u8)
    det = detect_periodic_peaks(
        gray,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
        max_peaks=max_peaks,
    )
    h, w = gray.shape
    cy, cx = center_coords((h, w))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    # Radii from peak positions (deduplicate within tolerance)
    radii: List[float] = []
    for (p1, p2) in det.paired_peaks:
        y, x = p1
        r = float(np.hypot(y - cy, x - cx))
        radii.append(r)
    radii = sorted(radii)
    dedup: List[float] = []
    for r in radii:
        if not dedup or abs(r - dedup[-1]) > 6.0:
            dedup.append(r)

    bw = max(1.0, float(bandwidth))
    s = max(0.5, float(sigma))
    mask = np.ones((h, w), dtype=np.float32)

    # Smooth band reject around each radius using Gaussian in radial distance
    for r0 in dedup:
        # band core where |r-r0| small -> reject; far -> pass
        dr = np.abs(rr - r0)
        core = np.exp(-(dr**2) / (2.0 * (bw**2)))
        # convert to reject notch-like: 1 - core, then soften by sigma (s)
        band_reject = 1.0 - core
        if s != 1.0:
            band_reject = np.clip(band_reject, 0.0, 1.0) ** (1.0 / s)
        mask *= band_reject.astype(np.float32)

    restored, filtered_spec = apply_frequency_filter(gray, mask)
    return restored, det, filtered_spec


def mask_from_two_points(
    shape: Tuple[int, int],
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    *,
    radius: int = 6,
) -> tuple[np.ndarray, list[tuple[tuple[int, int], tuple[int, int]]]]:
    """
    Build a hard notch mask (0 in discs) from two user-selected points.
    Automatically creates symmetric points around the center.
    """
    h, w = int(shape[0]), int(shape[1])
    cy, cx = center_coords((h, w))

    def sym(p: Tuple[int, int]) -> Tuple[int, int]:
        y, x = int(p[0]), int(p[1])
        return int(2 * cy - y), int(2 * cx - x)

    points = [tuple(map(int, p1)), tuple(map(int, p2))]
    sym_points = [sym(points[0]), sym(points[1])]
    peak_pairs = [(points[0], sym_points[0]), (points[1], sym_points[1])]

    yy, xx = np.ogrid[:h, :w]
    mask = np.ones((h, w), dtype=np.float32)
    r = max(1, int(radius))
    for (y0, x0) in points + sym_points:
        disc = (yy - y0) ** 2 + (xx - x0) ** 2 <= r**2
        mask[disc] = 0.0
    return mask, peak_pairs


def remove_periodic_mask_click(
    gray_or_rgb_u8: np.ndarray,
    *,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    radius: int = 6,
    soften_sigma: float = 3.5,
) -> tuple[np.ndarray, list[tuple[tuple[int, int], tuple[int, int]]], np.ndarray]:
    """
    Mask-based periodic removal from two clicked points on the shifted spectrum.

    Returns:
      restored_gray_u8, peak_pairs_used, filtered_spectrum_u8
    """
    ensure_uint8_image(gray_or_rgb_u8, name="image")
    gray = gray_or_rgb_u8 if gray_or_rgb_u8.ndim == 2 else to_gray_u8(gray_or_rgb_u8)
    hard_mask, peak_pairs = mask_from_two_points(gray.shape, p1, p2, radius=radius)
    # soften hard mask via gaussian notch multiply for fewer ringing artifacts
    soft = gaussian_notch_reject_mask(gray.shape, peak_pairs, sigma=float(soften_sigma))
    mask = np.clip(hard_mask * soft, 0.0, 1.0).astype(np.float32)
    restored, filtered_spec = apply_frequency_filter(gray, mask)
    return restored, peak_pairs, filtered_spec

