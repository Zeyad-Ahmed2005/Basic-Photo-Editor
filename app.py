from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image

from core import editing, filters, fourier_ops, histogram_ops, image_loader, noise, restoration
from core.utils import ImageInfo, ImageValidationError, ensure_uint8_image, get_image_info, image_hash_uint8
from ui import controls, display, sidebar


@dataclass
class HistoryItem:
    image_rgb: np.ndarray
    label: str


def _inject_css() -> None:
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
          [data-testid="stSidebar"] { padding-top: 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    ss = st.session_state
    ss.setdefault("original_image_rgb", None)
    ss.setdefault("working_image_rgb", None)
    ss.setdefault("preview_image_rgb", None)
    ss.setdefault("history", [])
    ss.setdefault("selected_fft_points", [])
    ss.setdefault("fft_click_key_version", 0)
    ss.setdefault("last_uploaded_hash", None)


def _push_history(label: str) -> None:
    ss = st.session_state
    img = ss.get("working_image_rgb")
    if img is None:
        return
    st.session_state.history.append(HistoryItem(image_rgb=img.copy(), label=label))


def _undo() -> None:
    ss = st.session_state
    if not ss.history:
        return
    last = ss.history.pop()
    ss.working_image_rgb = last.image_rgb
    ss.preview_image_rgb = last.image_rgb.copy()


def _reset() -> None:
    ss = st.session_state
    if ss.original_image_rgb is None:
        return
    ss.history = []
    ss.working_image_rgb = ss.original_image_rgb.copy()
    ss.preview_image_rgb = ss.original_image_rgb.copy()
    ss.selected_fft_points = []
    ss.fft_click_key_version += 1


def _clear_fft_points() -> None:
    st.session_state.selected_fft_points = []
    st.session_state.fft_click_key_version += 1


@st.cache_data(show_spinner=False)
def cached_fft(gray_u8: np.ndarray) -> fourier_ops.FourierData:
    return fourier_ops.fft2_gray(gray_u8)


@st.cache_data(show_spinner=False)
def cached_gray(rgb_u8: np.ndarray) -> np.ndarray:
    return image_loader.to_gray_u8(rgb_u8)


def _as_rgb_from_gray(gray_u8: np.ndarray) -> np.ndarray:
    ensure_uint8_image(gray_u8, name="gray")
    if gray_u8.ndim != 2:
        raise ImageValidationError("Expected 2D grayscale.")
    return np.stack([gray_u8, gray_u8, gray_u8], axis=2).astype(np.uint8)


def _download_button(rgb_u8: np.ndarray, *, filename: str = "processed.png") -> None:
    ensure_uint8_image(rgb_u8, name="image")
    pil = Image.fromarray(rgb_u8[..., :3], mode="RGB")
    buf = BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    st.download_button(
        "Download processed image (PNG)",
        data=buf.getvalue(),
        file_name=filename,
        mime="image/png",
        width="stretch",
    )


def _png_bytes(rgb_u8: np.ndarray) -> bytes:
    ensure_uint8_image(rgb_u8, name="image")
    pil = Image.fromarray(rgb_u8[..., :3], mode="RGB")
    buf = BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _zoom_spectrum(spectrum_u8: np.ndarray, zoom: float) -> np.ndarray:
    if zoom <= 1.0:
        return spectrum_u8
    h, w = spectrum_u8.shape[:2]
    cy, cx = fourier_ops.center_coords((h, w))
    half_h = int(round(h / (2.0 * zoom)))
    half_w = int(round(w / (2.0 * zoom)))
    y1 = max(0, cy - half_h)
    y2 = min(h, cy + half_h)
    x1 = max(0, cx - half_w)
    x2 = min(w, cx + half_w)
    crop = spectrum_u8[y1:y2, x1:x2]
    return crop


def main() -> None:
    st.set_page_config(page_title="Image Processing Course App", layout="wide")
    _inject_css()
    init_state()

    ss = st.session_state
    working: Optional[np.ndarray] = ss.working_image_rgb
    info: Optional[ImageInfo] = get_image_info(working) if working is not None else None

    download_data = _png_bytes(working) if working is not None else None
    sb = sidebar.render_sidebar(
        image_info=info,
        history_len=len(ss.history),
        download_data=download_data,
        download_filename="processed.png",
    )
    if sb.actions.clear_fft_points:
        _clear_fft_points()
    if sb.actions.undo:
        _undo()
    if sb.actions.reset:
        _reset()

    st.title("Streamlit Image Processing Course App")
    st.caption("Upload an image and explore histogram operations, filtering, Fourier analysis, noise, restoration, and editing.")

    # Ensure preview defaults to working each run
    if ss.working_image_rgb is not None and ss.preview_image_rgb is None:
        ss.preview_image_rgb = ss.working_image_rgb.copy()

    try:
        if sb.page == "Upload Image":
            st.markdown("### Upload")
            data = controls.upload_controls()
            if data is not None:
                loaded = image_loader.load_image_from_bytes(data)
                h = image_hash_uint8(loaded.rgb)
                if ss.last_uploaded_hash != h:
                    ss.last_uploaded_hash = h
                    ss.original_image_rgb = loaded.rgb.copy()
                    ss.working_image_rgb = loaded.rgb.copy()
                    ss.preview_image_rgb = loaded.rgb.copy()
                    ss.history = []
                    ss.selected_fft_points = []

            if ss.working_image_rgb is None:
                st.info("Upload an image to begin.")
                return

            original_rgb = ss.original_image_rgb
            assert original_rgb is not None
            with st.container(border=True):
                st.markdown("### Workspace")
                display.show_image_pair(
                    original_rgb=original_rgb,
                    processed_rgb=ss.working_image_rgb,
                    original_title="Original (stored)",
                    processed_title="Working image",
                )

            gray = cached_gray(ss.working_image_rgb)
            with st.container(border=True):
                st.markdown("### Analysis (Grayscale)")
                col1, col2 = st.columns(2, gap="large")
                with col1:
                    st.subheader("Grayscale")
                    st.image(gray, clamp=True, width="stretch")
                with col2:
                    st.subheader("Histogram")
                    fig = histogram_ops.fig_hist_gray(gray, "Grayscale Histogram (Working)")
                    st.image(display.matplotlib_fig_to_png_bytes(fig), width="stretch")

            st.markdown("### Download")
            _download_button(ss.working_image_rgb, filename="working.png")

        elif sb.page == "Histogram Operations":
            if ss.working_image_rgb is None:
                st.info("Upload an image first.")
                return

            cfg = controls.histogram_controls()
            img = ss.working_image_rgb
            assert img is not None
            gray = cached_gray(img)

            if cfg.enabled:
                eq = histogram_ops.equalize_histogram(img, mode=cfg.color_mode)
                ss.preview_image_rgb = eq
            else:
                ss.preview_image_rgb = img.copy()

            preview = ss.preview_image_rgb
            assert preview is not None
            with st.container(border=True):
                st.markdown("### Workspace")
                display.show_image_pair(
                    original_rgb=img,
                    processed_rgb=preview,
                    original_title="Input (Working)",
                    processed_title="Preview",
                )

            pgray = cached_gray(preview)
            with st.container(border=True):
                st.markdown("### Analysis (Histograms grid)")
                # Row 1: grayscale histograms
                row1_a, row1_b = st.columns(2, gap="large")
                with row1_a:
                    st.subheader("Grayscale histogram (Working)")
                    display.show_matplotlib(histogram_ops.fig_hist_gray(gray, "Working"))
                with row1_b:
                    st.subheader("Grayscale histogram (Preview)")
                    display.show_matplotlib(histogram_ops.fig_hist_gray(pgray, "Preview"))

                # Row 2: RGB histograms if color
                if img.ndim == 3:
                    row2_a, row2_b = st.columns(2, gap="large")
                    with row2_a:
                        st.subheader("RGB histogram (Working)")
                        display.show_matplotlib(histogram_ops.fig_hist_rgb(img, "Working"))
                    with row2_b:
                        st.subheader("RGB histogram (Preview)")
                        display.show_matplotlib(histogram_ops.fig_hist_rgb(preview, "Preview"))

            ab = controls.render_apply_bar(disabled=not cfg.enabled)
            if ab.apply and cfg.enabled:
                _push_history("Histogram equalization")
                ss.working_image_rgb = preview.copy()
                st.rerun()

        elif sb.page == "Spatial Filtering":
            if ss.working_image_rgb is None:
                st.info("Upload an image first.")
                return
            cfg = controls.spatial_filter_controls()
            img = ss.working_image_rgb
            assert img is not None
            if cfg.kind == "sobel":
                out_g = filters.sobel(img, direction=cfg.sobel_dir, ksize=cfg.ksize)
            else:
                out_g = filters.laplacian(img, ksize=cfg.ksize, scale=cfg.lap_scale, delta=cfg.lap_delta)
            preview = _as_rgb_from_gray(out_g)
            ss.preview_image_rgb = preview

            display.show_image_pair(original_rgb=img, processed_rgb=preview, original_title="Input (Working)", processed_title="Filtered (Preview)")
            ab = controls.render_apply_bar()
            if ab.apply:
                _push_history(f"Spatial filter: {cfg.kind}")
                ss.working_image_rgb = preview.copy()
                st.rerun()

        elif sb.page == "Fourier Transform":
            if ss.working_image_rgb is None:
                st.info("Upload an image first.")
                return
            cfg = controls.fourier_controls()
            img = ss.working_image_rgb
            assert img is not None
            gray = cached_gray(img)
            fd = cached_fft(gray)
            spectrum = _zoom_spectrum(fd.spectrum_u8, cfg.zoom)

            st.markdown("### Fourier Spectrum")
            col1, col2 = st.columns([1, 1], gap="large")
            with col1:
                st.subheader("Working image")
                st.image(img, channels="RGB", width="stretch")
            with col2:
                st.subheader("Magnitude spectrum (log)")
                st.image(spectrum, clamp=True, width="stretch")
                st.caption("Low frequencies are centered (fftshift).")

        elif sb.page == "Add Noise":
            if ss.working_image_rgb is None:
                st.info("Upload an image first.")
                return
            cfg = controls.noise_controls()
            img = ss.working_image_rgb
            assert img is not None
            if cfg.kind == "salt & pepper":
                preview = noise.add_salt_and_pepper(img, amount=cfg.sp_amount, salt_vs_pepper=cfg.sp_ratio)
                label = "Salt & pepper noise"
            else:
                preview = noise.add_periodic_noise(
                    img, frequency=cfg.per_freq, amplitude=cfg.per_amp, orientation_deg=cfg.per_orient
                )
                label = "Periodic noise"
            ss.preview_image_rgb = preview
            with st.container(border=True):
                st.markdown("### Workspace")
                display.show_image_pair(
                    original_rgb=img,
                    processed_rgb=preview,
                    original_title="Input (Working)",
                    processed_title="Noisy (Preview)",
                )

            with st.container(border=True):
                st.markdown("### Analysis (Spectrum grid)")
                g_work = cached_gray(img)
                fd_work = cached_fft(g_work)
                g_prev = cached_gray(preview)
                fd_prev = cached_fft(g_prev)
                col_a, col_b = st.columns(2, gap="large")
                with col_a:
                    st.subheader("Spectrum (Working)")
                    st.image(fd_work.spectrum_u8, clamp=True, width="stretch")
                with col_b:
                    st.subheader("Spectrum (Preview)")
                    st.image(fd_prev.spectrum_u8, clamp=True, width="stretch")

            ab = controls.render_apply_bar()
            if ab.apply:
                _push_history(label)
                ss.working_image_rgb = preview.copy()
                st.rerun()

        elif sb.page == "Noise Removal":
            if ss.working_image_rgb is None:
                st.info("Upload an image first.")
                return
            cfg = controls.denoise_controls()
            img = ss.working_image_rgb
            assert img is not None

            extra_panels = []

            if cfg.method == "median":
                preview = restoration.median_filter(img, ksize=cfg.median_ksize)
                ss.preview_image_rgb = preview
            elif cfg.method == "notch_auto":
                restored_g, det, filtered_spec = restoration.remove_periodic_notch_auto(
                    img,
                    sigma=cfg.notch_sigma,
                    min_distance=cfg.peak_min_dist,
                    threshold_rel=cfg.peak_thr,
                    max_peaks=cfg.max_peaks,
                )
                preview = _as_rgb_from_gray(restored_g)
                ss.preview_image_rgb = preview
                extra_panels = [("Detected peaks", det.spectrum_u8, det.peaks), ("Filtered spectrum", filtered_spec, None)]
            elif cfg.method == "band_reject_auto":
                restored_g, det, filtered_spec = restoration.remove_periodic_band_reject_auto(
                    img,
                    bandwidth=cfg.band_bw,
                    min_distance=cfg.peak_min_dist,
                    threshold_rel=cfg.peak_thr,
                    max_peaks=cfg.max_peaks,
                )
                preview = _as_rgb_from_gray(restored_g)
                ss.preview_image_rgb = preview
                extra_panels = [("Detected peaks", det.spectrum_u8, det.peaks), ("Filtered spectrum", filtered_spec, None)]
            else:
                st.markdown("#### Mask-based removal (click two points)")
                gray = cached_gray(img)
                fd = cached_fft(gray)
                pts = list(ss.selected_fft_points)
                click_key = f"fft_click_{int(ss.fft_click_key_version)}"
                pts, last = controls.click_two_points_on_spectrum(
                    fd.spectrum_u8, key=click_key, existing_points=pts
                )
                ss.selected_fft_points = pts

                col_a, col_b = st.columns(2)
                with col_a:
                    st.write(f"**Selected points:** {pts if pts else 'None'}")
                    if last is not None:
                        st.caption(f"Last click: (y={last[0]}, x={last[1]})")
                with col_b:
                    fig = display.fig_spectrum_with_points(fd.spectrum_u8, points=pts)
                    display.show_matplotlib(fig, caption="Points are in shifted spectrum coordinates.")

                if len(pts) == 2:
                    restored_g, peak_pairs, filtered_spec = restoration.remove_periodic_mask_click(
                        img, p1=pts[0], p2=pts[1], radius=cfg.mask_radius, soften_sigma=cfg.soften_sigma
                    )
                    preview = _as_rgb_from_gray(restored_g)
                    ss.preview_image_rgb = preview
                    extra_panels = [("Filtered spectrum", filtered_spec, None)]
                else:
                    ss.preview_image_rgb = img.copy()
                    preview = ss.preview_image_rgb

            preview = ss.preview_image_rgb
            assert preview is not None
            with st.container(border=True):
                st.markdown("### Workspace")
                display.show_image_pair(
                    original_rgb=img,
                    processed_rgb=preview,
                    original_title="Input (Working)",
                    processed_title="Restored (Preview)",
                )

            if extra_panels:
                with st.container(border=True):
                    st.markdown("### Frequency-domain diagnostics (grid)")
                    cols = st.columns(len(extra_panels), gap="large")
                    for (title, spec, points), col in zip(extra_panels, cols, strict=False):
                        with col:
                            st.subheader(title)
                            if points is not None:
                                fig = display.fig_spectrum_with_points(spec, points=[(int(y), int(x)) for y, x in points])
                                display.show_matplotlib(fig)
                            else:
                                st.image(spec, clamp=True, width="stretch")

            ab = controls.render_apply_bar(disabled=cfg.method == "mask_click" and len(ss.selected_fft_points) != 2)
            if ab.apply:
                _push_history(f"Noise removal: {cfg.method}")
                ss.working_image_rgb = preview.copy()
                st.rerun()

        else:  # Image Editing
            if ss.working_image_rgb is None:
                st.info("Upload an image first.")
                return
            img = ss.working_image_rgb
            assert img is not None
            h, w = img.shape[:2]
            cfg = controls.edit_controls(image_shape=(h, w))

            if cfg.tool == "crop":
                preview = editing.crop(img, x1=cfg.x1, y1=cfg.y1, x2=cfg.x2, y2=cfg.y2)
            elif cfg.tool == "rotate":
                preview = editing.rotate(img, angle_deg=cfg.angle, keep_size=cfg.keep_size)
            elif cfg.tool == "resize":
                preview = editing.resize(img, width=cfg.width, height=cfg.height, keep_aspect=cfg.keep_aspect)
            else:
                preview = editing.brightness_contrast(img, brightness=cfg.brightness, contrast=cfg.contrast)

            ss.preview_image_rgb = preview
            with st.container(border=True):
                st.markdown("### Workspace")
                display.show_image_pair(
                    original_rgb=img,
                    processed_rgb=preview,
                    original_title="Input (Working)",
                    processed_title="Edited (Preview)",
                )
            ab = controls.render_apply_bar()
            if ab.apply:
                _push_history(f"Edit: {cfg.tool}")
                ss.working_image_rgb = preview.copy()
                st.rerun()

    except ImageValidationError as e:
        st.error(str(e))
    except Exception as e:  # noqa: BLE001
        st.exception(e)

    # Footer
    if ss.working_image_rgb is not None:
        st.markdown("---")
        st.markdown("### History")
        if ss.history:
            st.write([h.label for h in ss.history][-10:])
        else:
            st.caption("No committed operations yet. Use **Apply changes** to commit a preview.")


if __name__ == "__main__":
    main()

