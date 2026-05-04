# 🎨 Image Data Augmentation Studio
### A Machine Learning Project

---

## Overview

This project implements **16 image data augmentation techniques** commonly used in
machine learning and computer vision. The Streamlit frontend lets you upload any
image, toggle techniques, and instantly preview augmented versions — all without
writing a single line of code.

---

## Features

| Category | Techniques |
|---|---|
| **Geometric** | Rotation, Flip (H/V/Both), Zoom/Crop, Translation |
| **Photometric** | Brightness, Contrast, Color Jitter, Grayscale, Solarize |
| **Noise & Blur** | Gaussian Noise, Salt & Pepper, Gaussian Blur, Sharpen |
| **Advanced** | Cutout, Random Erasing, Elastic Transform |

Additional features:
- Real-time per-technique preview grid
- Pixel distribution (RGB histogram) comparison
- Batch generation of N augmented images
- Download all results as a ZIP file

---

## Setup & Run

```bash
# 1. Clone or copy the project files
cd image_augmentation_app/

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit app
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## File Structure

```
image_augmentation_app/
├── app.py              # Streamlit UI & application logic
├── augmentation.py     # All 16 augmentation implementations
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## Technique Details

### Geometric Transforms
- **Rotation** — rotates by ±angle; fills borders with mean colour
- **Flip** — horizontal, vertical, or both axes
- **Zoom / Crop** — zooms into centre and resizes back
- **Translation** — shifts image by % of width/height

### Photometric Transforms
- **Brightness** — multiplies pixel values by a factor
- **Contrast** — adjusts dynamic range
- **Color Jitter** — random brightness + contrast + saturation
- **Grayscale** — converts to 3-channel grey (common in self-supervised learning)
- **Solarize** — inverts pixels above threshold (used in SimCLR, BYOL)

### Noise & Blur
- **Gaussian Noise** — adds N(0, σ²) noise to every pixel
- **Salt & Pepper** — randomly sets pixels to 0 or 255
- **Gaussian Blur** — convolves with Gaussian kernel (radius r)
- **Sharpen** — unsharp-mask sharpening

### Advanced / Regularisation
- **Cutout** (DeVries & Taylor 2017) — zeros out a random square patch
- **Random Erasing** (Zhong et al. 2020) — fills a random rect with noise
- **Elastic Transform** (Simard et al. 2003) — smooth random warping; widely used in medical image segmentation (U-Net)

---

## Why Data Augmentation?

Training deep neural networks requires large amounts of labelled data.
Augmentation artificially multiplies the effective dataset size:

```
Original dataset: 1 000 images
Augmented (×10): 10 000 effective training samples
```

Benefits:
- **Reduces overfitting** — model sees each image in many forms
- **Improves generalisation** — learns transformation-invariant features
- **Free** — no extra annotation cost

---

## References

1. DeVries & Taylor (2017) — *Improved Regularization of CNNs with Cutout*
2. Zhong et al. (2020) — *Random Erasing Data Augmentation*
3. Simard et al. (2003) — *Best Practices for CNNs Applied to Visual Document Analysis* (Elastic)
4. Chen et al. (2020) — *A Simple Framework for Contrastive Learning* (SimCLR / Solarize)
5. Shorten & Khoshgoftaar (2019) — *A Survey on Image Data Augmentation for DL*
