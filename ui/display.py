from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.backends.backend_agg import FigureCanvasAgg

from core.utils import ImageInfo, ensure_uint8_image, get_image_info


def _safe_caption(text: str) -> None:
    if text:
        st.caption(text)


def show_image_pair(
    *,
    original_rgb: np.ndarray,
    processed_rgb: np.ndarray,
    original_title: str = "Original",
    processed_title: str = "Processed",
    original_caption: str = "",
    processed_caption: str = "",
) -> None:
    ensure_uint8_image(original_rgb, name="original")
    ensure_uint8_image(processed_rgb, name="processed")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader(original_title)
        st.image(original_rgb, channels="RGB", width="stretch")
        _safe_caption(original_caption)
    with col2:
        st.subheader(processed_title)
        st.image(processed_rgb, channels="RGB", width="stretch")
        _safe_caption(processed_caption)


def show_matplotlib(fig: plt.Figure, *, caption: str = "") -> None:
    st.pyplot(fig, clear_figure=False, width="stretch")
    _safe_caption(caption)


def matplotlib_fig_to_png_bytes(fig: plt.Figure, *, dpi: int = 140) -> bytes:
    """Render a Matplotlib Figure to PNG bytes (for symmetric st.image grids)."""
    canvas = FigureCanvasAgg(fig)
    fig.set_dpi(int(dpi))
    canvas.draw()
    buf = canvas.buffer_rgba()
    # Convert RGBA buffer to PNG via matplotlib's own savefig (more robust)
    import io

    out = io.BytesIO()
    fig.savefig(out, format="png", dpi=int(dpi), bbox_inches="tight")
    return out.getvalue()


def fig_spectrum_with_points(
    spectrum_u8: np.ndarray,
    *,
    points: Sequence[Tuple[int, int]] | None = None,
    title: str = "Fourier Magnitude Spectrum (log)",
) -> plt.Figure:
    if spectrum_u8 is None or not isinstance(spectrum_u8, np.ndarray) or spectrum_u8.size == 0:
        raise ValueError("Empty spectrum.")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(spectrum_u8, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
    if points:
        xs = [int(p[1]) for p in points]
        ys = [int(p[0]) for p in points]
        ax.scatter(xs, ys, s=45, c="lime", marker="x", linewidths=2)
    fig.tight_layout()
    return fig


def status_bar(*, info: Optional[ImageInfo]) -> None:
    if info is None:
        return
    st.caption(f"Image: {info.width}×{info.height} | Channels: {info.channels}")

