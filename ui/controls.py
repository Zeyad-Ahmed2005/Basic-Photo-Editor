from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates


@dataclass(frozen=True)
class ApplyBar:
    apply: bool
    preview_only: bool


def render_apply_bar(*, disabled: bool = False, label: str = "Apply changes") -> ApplyBar:
    """
    A consistent action bar:
    - Apply changes: commits preview into working image and pushes history
    - Preview only: keep changes as preview without committing
    """
    col1, col2 = st.columns([1, 1])
    with col1:
        apply = st.button(label, width="stretch", disabled=disabled)
    with col2:
        preview_only = st.checkbox("Preview only", value=True, help="If off, applying will overwrite the working image.")
    return ApplyBar(apply=apply, preview_only=preview_only)


def upload_controls() -> Optional[bytes]:
    f = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
    if f is None:
        return None
    return f.getvalue()


@dataclass(frozen=True)
class HistEqControls:
    enabled: bool
    color_mode: Literal["ycrcb", "hsv"]


def histogram_controls() -> HistEqControls:
    st.markdown("### Histogram Equalization")
    enabled = st.toggle("Enable equalization", value=False)
    color_mode = st.selectbox("Color equalization mode", ["ycrcb", "hsv"], index=0)
    return HistEqControls(enabled=bool(enabled), color_mode=color_mode)


@dataclass(frozen=True)
class SpatialControls:
    kind: Literal["sobel", "laplacian"]
    sobel_dir: Literal["x", "y", "mag"]
    ksize: int
    lap_scale: float
    lap_delta: float


def spatial_filter_controls() -> SpatialControls:
    st.markdown("### Spatial Filtering")
    kind = st.selectbox("Filter", ["sobel", "laplacian"], index=0)
    ksize = st.slider("Kernel size (odd)", min_value=1, max_value=15, value=3, step=2)
    sobel_dir: Literal["x", "y", "mag"] = "mag"
    lap_scale = 1.0
    lap_delta = 0.0
    if kind == "sobel":
        sobel_dir = st.selectbox("Direction", ["x", "y", "mag"], index=2)  # type: ignore[assignment]
    else:
        lap_scale = float(st.slider("Scale", min_value=0.1, max_value=5.0, value=1.0, step=0.1))
        lap_delta = float(st.slider("Delta", min_value=-100.0, max_value=100.0, value=0.0, step=1.0))
    return SpatialControls(
        kind=kind, sobel_dir=sobel_dir, ksize=int(ksize), lap_scale=lap_scale, lap_delta=lap_delta
    )


@dataclass(frozen=True)
class FourierControls:
    zoom: float


def fourier_controls() -> FourierControls:
    st.markdown("### Fourier Spectrum")
    zoom = float(st.slider("Spectrum zoom", min_value=1.0, max_value=6.0, value=1.0, step=0.25))
    return FourierControls(zoom=zoom)


@dataclass(frozen=True)
class NoiseControls:
    kind: Literal["salt_pepper", "periodic"]
    sp_amount: float
    sp_ratio: float
    per_freq: float
    per_amp: float
    per_orient: float


def noise_controls() -> NoiseControls:
    st.markdown("### Add Noise")
    kind = st.selectbox("Noise type", ["salt_pepper", "periodic"], index=0)
    sp_amount = 0.03
    sp_ratio = 0.5
    per_freq = 8.0
    per_amp = 35.0
    per_orient = 45.0
    if kind == "salt_pepper":
        sp_amount = float(st.slider("Noise amount", min_value=0.0, max_value=0.25, value=0.03, step=0.005))
        sp_ratio = float(st.slider("Salt vs pepper", min_value=0.0, max_value=1.0, value=0.5, step=0.05))
    else:
        per_freq = float(st.slider("Frequency", min_value=0.0, max_value=30.0, value=8.0, step=0.5))
        per_amp = float(st.slider("Amplitude", min_value=0.0, max_value=80.0, value=35.0, step=1.0))
        per_orient = float(st.slider("Orientation (deg)", min_value=0.0, max_value=180.0, value=45.0, step=1.0))
    return NoiseControls(
        kind=kind,
        sp_amount=sp_amount,
        sp_ratio=sp_ratio,
        per_freq=per_freq,
        per_amp=per_amp,
        per_orient=per_orient,
    )


@dataclass(frozen=True)
class DenoiseControls:
    method: Literal["median", "notch_auto", "band_reject_auto", "mask_click"]
    median_ksize: int
    notch_sigma: float
    peak_min_dist: int
    peak_thr: float
    max_peaks: int
    band_bw: float
    mask_radius: int
    soften_sigma: float


def denoise_controls() -> DenoiseControls:
    st.markdown("### Noise Removal")
    method = st.selectbox("Method", ["median", "notch_auto", "band_reject_auto", "mask_click"], index=0)
    median_ksize = 3
    notch_sigma = 6.0
    peak_min_dist = 12
    peak_thr = 0.35
    max_peaks = 16
    band_bw = 10.0
    mask_radius = 6
    soften_sigma = 3.5

    if method == "median":
        median_ksize = int(st.slider("Median kernel (odd)", min_value=3, max_value=21, value=3, step=2))
    elif method == "notch_auto":
        notch_sigma = float(st.slider("Notch sigma", min_value=2.0, max_value=20.0, value=6.0, step=0.5))
        peak_min_dist = int(st.slider("Peak min distance", min_value=6, max_value=40, value=12, step=1))
        peak_thr = float(st.slider("Peak threshold (rel)", min_value=0.05, max_value=0.9, value=0.35, step=0.01))
        max_peaks = int(st.slider("Max peaks", min_value=4, max_value=40, value=16, step=1))
    elif method == "band_reject_auto":
        band_bw = float(st.slider("Bandwidth", min_value=2.0, max_value=60.0, value=10.0, step=1.0))
        peak_min_dist = int(st.slider("Peak min distance", min_value=6, max_value=40, value=12, step=1))
        peak_thr = float(st.slider("Peak threshold (rel)", min_value=0.05, max_value=0.9, value=0.35, step=0.01))
        max_peaks = int(st.slider("Max peaks", min_value=4, max_value=40, value=16, step=1))
    else:
        mask_radius = int(st.slider("Mask radius (px)", min_value=2, max_value=30, value=6, step=1))
        soften_sigma = float(st.slider("Softening sigma", min_value=1.0, max_value=15.0, value=3.5, step=0.5))

    return DenoiseControls(
        method=method,
        median_ksize=median_ksize,
        notch_sigma=notch_sigma,
        peak_min_dist=peak_min_dist,
        peak_thr=peak_thr,
        max_peaks=max_peaks,
        band_bw=band_bw,
        mask_radius=mask_radius,
        soften_sigma=soften_sigma,
    )


@dataclass(frozen=True)
class EditControls:
    tool: Literal["crop", "rotate", "resize", "brightness_contrast"]
    x1: int
    y1: int
    x2: int
    y2: int
    angle: float
    keep_size: bool
    width: int
    height: int
    keep_aspect: bool
    brightness: int
    contrast: float


def edit_controls(*, image_shape: Tuple[int, int]) -> EditControls:
    h, w = int(image_shape[0]), int(image_shape[1])
    st.markdown("### Image Editing")
    tool = st.selectbox("Tool", ["crop", "rotate", "resize", "brightness_contrast"], index=0)

    x1 = 0
    y1 = 0
    x2 = w
    y2 = h
    angle = 0.0
    keep_size = True
    width = w
    height = h
    keep_aspect = True
    brightness = 0
    contrast = 1.0

    if tool == "crop":
        x1, y1, x2, y2 = 0, 0, w, h
    elif tool == "rotate":
        angle = float(st.slider("Angle (deg)", -180.0, 180.0, 0.0, 1.0))
        keep_size = bool(st.checkbox("Keep same size", value=True))
    elif tool == "resize":
        keep_aspect = bool(st.checkbox("Keep aspect ratio", value=True))
        col1, col2 = st.columns(2)
        with col1:
            width = int(st.number_input("Width", min_value=1, max_value=5000, value=int(w), step=1))
        with col2:
            height = int(st.number_input("Height", min_value=1, max_value=5000, value=int(h), step=1))
    else:
        brightness = int(st.slider("Brightness", -255, 255, 0, 1))
        contrast = float(st.slider("Contrast", 0.0, 3.0, 1.0, 0.01))

    return EditControls(
        tool=tool,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        angle=angle,
        keep_size=keep_size,
        width=width,
        height=height,
        keep_aspect=keep_aspect,
        brightness=brightness,
        contrast=contrast,
    )


def click_two_points_on_spectrum(
    spectrum_u8: np.ndarray, *, key: str, existing_points: list[tuple[int, int]]
) -> tuple[list[tuple[int, int]], Optional[tuple[int, int]]]:
    """
    Render spectrum and collect a click coordinate using streamlit-image-coordinates.

    Returns:
        updated_points, last_clicked_point
    """
    if spectrum_u8 is None or not isinstance(spectrum_u8, np.ndarray) or spectrum_u8.size == 0:
        return existing_points, None
    # The component reports clicks in the *rendered* image coordinate system.
    # We render with a fixed width, so we must scale back to original spectrum pixels.
    if spectrum_u8.ndim == 2:
        rgb = np.stack([spectrum_u8, spectrum_u8, spectrum_u8], axis=2)
    else:
        rgb = spectrum_u8
    pil = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    render_w = 700
    click = streamlit_image_coordinates(pil, key=key, width=render_w)
    last = None
    pts = list(existing_points)
    if click is not None and "x" in click and "y" in click:
        orig_h, orig_w = int(spectrum_u8.shape[0]), int(spectrum_u8.shape[1])
        rendered_w = int(render_w)
        rendered_h = int(round(orig_h * (rendered_w / float(orig_w)))) if orig_w > 0 else orig_h

        # click x/y are relative to the rendered image size
        x_r = float(click["x"])
        y_r = float(click["y"])
        x = int(round(x_r * (orig_w / float(rendered_w))))
        y = int(round(y_r * (orig_h / float(rendered_h if rendered_h > 0 else orig_h))))

        x = max(0, min(orig_w - 1, x))
        y = max(0, min(orig_h - 1, y))

        last = (y, x)
        if len(pts) < 2:
            pts.append((y, x))
    return pts, last

def click_two_points_on_image(
    image,
    *,
    key,
    existing_points=None,
):
    points = existing_points or []

    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return points, None
    
    pil = Image.fromarray(image.astype(np.uint8), mode="RGB")

    render_w = 700
    click = streamlit_image_coordinates(pil, key=key, width=render_w)
    last = None
    pts = list(existing_points)

    if click is not None and "x" in click and "y" in click:
        orig_h, orig_w = image.shape[0], image.shape[1]
        rendered_w = int(render_w)
        rendered_h = int(round(orig_h * (rendered_w / float(orig_w))))

        x_r = float(click["x"])
        y_r = float(click["y"])
        x = int(round(x_r * (orig_w / float(rendered_w))))
        y = int(round(y_r * (orig_h / float(rendered_h))))

        x = max(0, min(orig_w - 1, x))
        y = max(0, min(orig_h - 1, y))

        last = (y, x)
        if len(pts) < 2:
            pts.append((y, x))

    return pts, last