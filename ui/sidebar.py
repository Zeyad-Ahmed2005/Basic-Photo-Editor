from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import streamlit as st

from core.utils import ImageInfo


PAGES = [
    "Upload Image",
    "Histogram Operations",
    "Spatial Filtering",
    "Fourier Transform",
    "Add Noise",
    "Noise Removal",
    "Image Editing",
]


@dataclass(frozen=True)
class GlobalActions:
    reset: bool
    undo: bool
    clear_fft_points: bool


@dataclass(frozen=True)
class SidebarResult:
    page: str
    actions: GlobalActions


def render_sidebar(
    *,
    image_info: Optional[ImageInfo],
    history_len: int,
    download_data: Optional[bytes] = None,
    download_filename: str = "processed.png",
) -> SidebarResult:
    with st.sidebar:
        st.markdown("## Image Processing App")
        page = st.radio("Navigation", PAGES, index=0)

        st.markdown("---")
        st.markdown("### Global Actions")
        col_a, col_b = st.columns(2)
        with col_a:
            undo = st.button("Undo", width="stretch", disabled=history_len <= 0)
        with col_b:
            reset = st.button("Reset", width="stretch", disabled=image_info is None)
        clear_fft_points = st.button("Clear FFT points", width="stretch")
        st.download_button(
            "Download processed (PNG)",
            data=download_data if download_data is not None else b"",
            file_name=download_filename,
            mime="image/png",
            disabled=download_data is None,
            width="stretch",
        )

        st.markdown("---")
        st.markdown("### Image Info")
        if image_info is None:
            st.caption("No image loaded yet.")
        else:
            st.write(f"**Size:** {image_info.width} × {image_info.height}")
            st.write(f"**Channels:** {image_info.channels}")
            st.write(f"**History:** {history_len} step(s)")

    return SidebarResult(
        page=page,
        actions=GlobalActions(reset=reset, undo=undo, clear_fft_points=clear_fft_points),
    )

