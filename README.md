## Basic Photo Editor (Streamlit Image Processing App)

### Overview
Interactive desktop-style GUI built with **Python + Streamlit** for an Image Processing course project. Upload an image, apply operations, and see **live updates** for the processed output, histograms, and Fourier spectrum.

### Run

```bash
pip install -r requirements.txt
streamlit run app.py --server.headless true --server.port 8511 --server.fileWatcherType none
```

### Features
- Upload JPG/PNG/BMP + show image info
- Histogram (gray + RGB) + histogram equalization (gray + luminance-only for color)
- Spatial filtering: Sobel (x/y/magnitude) + Laplacian (ksize/scale/delta)
- Fourier transform: FFT2, fftshift, log-magnitude spectrum
- Add noise: salt-and-pepper + periodic noise
- Remove noise:
  - Median filter
  - Periodic noise: automatic notch, automatic band-reject, interactive mask-based removal (click 2 spectrum points)
- Editing: crop, rotate, resize, brightness/contrast
- Undo, reset, download processed image

### Structure
```
app.py
requirements.txt
APP_README.md
assets/
sample_images/
outputs/
core/
ui/
```