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

---

### Processing pipelines

Each diagram shows the **data flow** for one operation: boxes are steps, arrows show what is passed to the next step. Most tools use **Preview → Apply** so you can compare before committing to the working image.

#### App workflow (all pages)

```mermaid
flowchart LR
  A[Upload image bytes] --> B[Decode to RGB uint8]
  B --> C[Working image in session]
  C --> D[Choose operation page]
  D --> E[Compute preview]
  E --> F{Apply changes?}
  F -->|Yes| G[Save to working image + history]
  F -->|No| C
  G --> C
  C --> H[Undo / Reset / Download]
```

---

#### 1. Image upload & analysis

**Upload** — load and normalize the file.

```mermaid
flowchart TD
  A[Image file JPG/PNG/BMP] --> B[PIL decode bytes]
  B --> C[EXIF auto-rotate]
  C --> D[Convert to RGB]
  D --> E[uint8 array H×W×3]
  E --> F[Store as original + working image]
```

**Histogram view** (Upload page) — intensity distribution of the working image.

```mermaid
flowchart TD
  A[RGB working image] --> B[Grayscale via BT.601 luma]
  B --> C[Count pixels per intensity 0–255]
  C --> D[Grayscale histogram plot]
  A --> E[Per-channel R, G, B counts]
  E --> F[RGB histogram plot]
```

---

#### 2. Histogram equalization (automatic contrast)

Redistributes pixel intensities so the histogram is flatter — this **improves contrast automatically** (different from the manual **Brightness / contrast** slider on the Image Editing page). Color images only equalize **luminance**, not hue.

```mermaid
flowchart TD
  A[Input image] --> B{Grayscale or color?}
  B -->|Grayscale| C[OpenCV equalizeHist]
  C --> Z[Equalized image]
  B -->|Color| D{Mode}
  D -->|YCrCb default| E[RGB → YCrCb]
  E --> F[Equalize Y channel only]
  F --> G[Merge Y, Cr, Cb → RGB]
  D -->|HSV| H[RGB → HSV]
  H --> I[Equalize V channel only]
  I --> J[Merge H, S, V → RGB]
  G --> Z
  J --> Z
```

---

#### 3. Spatial filtering (edges)

Both filters run on **grayscale**, then the result is shown as a gray RGB preview.

**Sobel** — horizontal, vertical, or gradient magnitude edges.

```mermaid
flowchart TD
  A[RGB input] --> B[Convert to grayscale]
  B --> C{Direction}
  C -->|X| D[Sobel ∂/∂x]
  C -->|Y| E[Sobel ∂/∂y]
  C -->|Magnitude| F[Sobel X and Y]
  F --> G[sqrt Gx² + Gy²]
  D --> H[Absolute value]
  E --> H
  G --> H
  H --> I[Normalize to 0–255]
  I --> J[Edge map preview]
```

**Laplacian** — second-derivative / sharpness emphasis.

```mermaid
flowchart TD
  A[RGB input] --> B[Convert to grayscale]
  B --> C[Laplacian kernel convolve]
  C --> D[Absolute value]
  D --> E[Scale + delta offset]
  E --> F[Normalize to 0–255]
  F --> G[Edge map preview]
```

---

#### 4. Fourier transform (spectrum view)

Shows **where** frequency energy lives in the image (no image change).

```mermaid
flowchart TD
  A[RGB working image] --> B[Grayscale]
  B --> C[2D FFT]
  C --> D[fftshift — DC at center]
  D --> E[Magnitude |F|]
  E --> F[log1p for display range]
  F --> G[Normalize to 0–255]
  G --> H[Log-magnitude spectrum image]
```

---

#### 5. Add noise

**Salt & pepper** — random impulse noise (white/black pixels).

```mermaid
flowchart TD
  A[RGB image] --> B[Pick random pixel locations]
  B --> C{Per pixel}
  C -->|Salt fraction| D[Set RGB = 255 white]
  C -->|Pepper| E[Set RGB = 0 black]
  D --> F[Noisy preview]
  E --> F
```

**Periodic noise** — sinusoidal pattern overlaid on the image.

```mermaid
flowchart TD
  A[RGB image] --> B[Build 2D sine wave]
  B --> C["pattern = A·sin(2πf·projection)"]
  C --> D[Add pattern to all channels]
  D --> E[Clip to 0–255]
  E --> F[Noisy preview]
  F --> G[FFT spectrum shows new peaks]
```

---

#### 6. Remove noise

**Median filter** — best for **salt & pepper** (spatial domain).

```mermaid
flowchart TD
  A[RGB noisy image] --> B[For each channel R, G, B]
  B --> C[Median filter k×k window]
  C --> D[Replace pixel with neighborhood median]
  D --> E[Restored RGB preview]
```

**Automatic notch filter** — removes **periodic** noise via frequency peaks.

```mermaid
flowchart TD
  A[RGB input] --> B[Grayscale]
  B --> C[FFT → shifted spectrum]
  C --> D[Suppress DC center blob]
  D --> E[Find local maxima peaks]
  E --> F[Pair each peak with symmetric opposite]
  F --> G[Gaussian notches at peak pairs]
  G --> H[Multiply spectrum by mask]
  H --> I[Inverse FFT]
  I --> J[Clip → restored grayscale]
  J --> K[Show as RGB preview]
```

**Automatic band-reject** — rejects whole **rings** of radius around the spectrum center.

```mermaid
flowchart TD
  A[RGB input] --> B[Grayscale]
  B --> C[FFT + peak detection same as notch]
  C --> D[Compute radius of each peak from center]
  D --> E[Smooth annular band-reject mask per radius]
  E --> F[Multiply spectrum by combined mask]
  F --> G[Inverse FFT → restored image]
```

**Interactive mask (2 clicks)** — you pick two points on the spectrum.

```mermaid
flowchart TD
  A[User clicks 2 points on spectrum] --> B[Mirror each point through center]
  B --> C[Hard circular masks at 4 locations]
  C --> D[Soften with Gaussian notch blend]
  D --> E[Multiply spectrum by mask]
  E --> F[Inverse FFT → restored image]
```

Shared **frequency-domain reconstruction** (used by all periodic removal methods):

```mermaid
flowchart LR
  A[Grayscale] --> B[FFT2]
  B --> C[fftshift]
  C --> D["× filter mask"]
  D --> E[ifftshift]
  E --> F[IFFT2 real part]
  F --> G[Clip 0–255 uint8]
```

---

#### 7. Image editing

**Crop**

```mermaid
flowchart LR
  A[RGB image] --> B[Select rectangle x1,y1 → x2,y2]
  B --> C[Extract sub-array]
  C --> D[Cropped RGB]
```

**Rotate**

```mermaid
flowchart TD
  A[RGB image] --> B[Rotation matrix around center]
  B --> C{Keep original size?}
  C -->|Yes| D[warpAffine same W×H]
  C -->|No| E[Expand canvas to fit rotated image]
  D --> F[Rotated RGB]
  E --> F
```

**Resize**

```mermaid
flowchart TD
  A[RGB image] --> B{Keep aspect ratio?}
  B -->|Yes| C[Fit inside target W×H box]
  B -->|No| D[Stretch to exact W×H]
  C --> E[Lanczos interpolation]
  D --> E
  E --> F[Resized RGB]
```

**Brightness & contrast** (Image Editing page)

Manual adjustment on every RGB channel using a linear formula (see `core/editing.py`):

`new_pixel = contrast × old_pixel + brightness`

| Control | Slider range | Effect |
|--------|----------------|--------|
| **Contrast** | 0.0 – 3.0 (default 1.0) | Scales how far each pixel is from black. **&lt; 1** flattens the image (low contrast); **&gt; 1** stretches differences (high contrast). |
| **Brightness** | −255 – 255 (default 0) | Shifts all intensities up or down after scaling. |

```mermaid
flowchart TD
  A[RGB input uint8] --> B[Convert to float32]
  B --> C[For each channel R, G, B]
  C --> D["Multiply by contrast factor"]
  D --> E["Add brightness offset"]
  E --> F[Clip values to 0–255]
  F --> G[Convert back to uint8]
  G --> H[Adjusted RGB preview]
```

**What contrast does (intuition)**

```mermaid
flowchart LR
  subgraph contrast_lt_1 ["Contrast &lt; 1 (e.g. 0.5)"]
    A1[Dark and bright pixels] --> B1[Move toward mid-gray]
    B1 --> C1[Muddy, flat look]
  end
  subgraph contrast_eq_1 ["Contrast = 1"]
    A2[Unchanged spread] --> B2[Same as input if brightness = 0]
  end
  subgraph contrast_gt_1 ["Contrast &gt; 1 (e.g. 1.5)"]
    A3[Pixel values] --> B3[Stretched from black]
    B3 --> C3[Stronger shadows and highlights]
  end
```

**Contrast vs histogram equalization**

| | Histogram equalization | Brightness / contrast slider |
|--|------------------------|------------------------------|
| **Where** | Histogram Operations page | Image Editing page |
| **How** | Remaps intensities using the image histogram | Simple linear scale + shift per pixel |
| **Goal** | Spread histogram automatically | You control strength with sliders |

---

#### 8. Undo, reset, download

```mermaid
flowchart TD
  A[Apply changes] --> B[Push copy to history stack]
  B --> C[Working image updated]
  D[Undo] --> E[Pop history → restore previous working image]
  F[Reset] --> G[Working image = original upload]
  H[Download] --> I[Encode working image as PNG bytes]
```

---

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
