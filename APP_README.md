## Image Processing Course App (Streamlit)

### Overview
Interactive desktop-style GUI built with **Python + Streamlit** for common image processing techniques. Upload an image, apply operations, and see **live updates** for the processed output, histograms, and Fourier spectrum.

### Features
- **Upload** JPG/PNG/BMP + show image info (shape/channels)
- **Histogram** (gray + per-channel), histogram equalization (gray + color luminance-only)
- **Spatial filtering**: Sobel (x/y/magnitude), Laplacian (ksize/scale/delta)
- **Fourier transform**: FFT2, shifted spectrum, log-magnitude view
- **Noise**: salt-and-pepper, periodic noise (frequency/amplitude/orientation)
- **Restoration**
  - Median filter (salt-and-pepper removal)
  - Periodic noise removal:
    - Notch filter with **automatic peak detection**
    - Band-reject filtering with adjustable bandwidth
    - **Mask-based removal** via clicking 2 pixels in the spectrum (symmetric masking)
- **Editing**: crop, rotate, resize, brightness/contrast
- **Bonus**: download processed image, reset, undo, processing history

### Installation
Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows PowerShell:
.\\.venv\\Scripts\\Activate.ps1
pip install -r image_processing_app/requirements.txt
```

### Run

```bash
streamlit run app.py --server.headless true --server.port 8511 --server.fileWatcherType none
```

### Folder structure
```
image_processing_app/
  app.py
  requirements.txt
  README.md
  assets/
  sample_images/
  outputs/
  core/
    image_loader.py
    histogram_ops.py
    filters.py
    fourier_ops.py
    noise.py
    restoration.py
    editing.py
    utils.py
  ui/
    sidebar.py
    display.py
    controls.py
```

### Screenshots
Add screenshots here after running the app:
- `assets/screenshot_main.png`
- `assets/screenshot_fourier.png`

