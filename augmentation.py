"""
augmentation.py
================
Pure-Python / PIL / NumPy implementations of every augmentation
technique used in the Streamlit app.
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import random
import math


# ─────────────────────────────────────────────
#  GEOMETRIC TRANSFORMS
# ─────────────────────────────────────────────

def rotate_image(img: Image.Image, angle: float) -> Image.Image:
    """
    Rotate image by `angle` degrees.
    Fills background with the image's mean colour to avoid black borders.
    """
    # Cache mean calculation
    arr = np.array(img)
    fill = tuple(int(c) for c in arr.mean(axis=(0, 1)))
    return img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=fill)



def flip_image(img: Image.Image, mode: str = "horizontal") -> Image.Image:
    """
    Flip image.  mode ∈ {'horizontal', 'vertical', 'both'}
    """
    if mode == "horizontal":
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    elif mode == "vertical":
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    else:  # both
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img.transpose(Image.FLIP_TOP_BOTTOM)



def zoom_image(img: Image.Image, zoom: float = 1.3) -> Image.Image:
    """
    Zoom into the centre by factor `zoom`, then resize back to original dims.
    """
    W, H = img.size
    new_W = int(W / zoom)
    new_H = int(H / zoom)
    left = (W - new_W) // 2
    top  = (H - new_H) // 2
    cropped = img.crop((left, top, left + new_W, top + new_H))
    return cropped.resize((W, H), Image.Resampling.BILINEAR)



def translate_image(img: Image.Image, dx_pct: int = 15, dy_pct: int = 15) -> Image.Image:
    """
    Translate (shift) image by dx_pct % and dy_pct % of width/height.
    Empty regions are filled with mean colour.
    """
    W, H = img.size
    dx = int(W * dx_pct / 100)
    dy = int(H * dy_pct / 100)
    arr = np.array(img)
    fill_val = arr.mean(axis=(0, 1)).astype(np.uint8)
    result = np.full_like(arr, fill_val)

    src_x1 = max(0, -dx);  src_x2 = min(W, W - dx)
    dst_x1 = max(0,  dx);  dst_x2 = min(W, W + dx)
    src_y1 = max(0, -dy);  src_y2 = min(H, H - dy)
    dst_y1 = max(0,  dy);  dst_y2 = min(H, H + dy)

    result[dst_y1:dst_y2, dst_x1:dst_x2] = arr[src_y1:src_y2, src_x1:src_x2]
    return Image.fromarray(result)


# ─────────────────────────────────────────────
#  PHOTOMETRIC TRANSFORMS
# ─────────────────────────────────────────────


def brightness_adjust(img: Image.Image, factor: float = 1.4) -> Image.Image:
    """Multiply brightness by `factor`.  factor < 1 → darker, > 1 → brighter."""
    return ImageEnhance.Brightness(img).enhance(factor)



def contrast_adjust(img: Image.Image, factor: float = 1.4) -> Image.Image:
    """Adjust contrast by `factor`."""
    return ImageEnhance.Contrast(img).enhance(factor)



def color_jitter(img: Image.Image) -> Image.Image:
    """
    Random colour jitter: brightness, contrast, saturation, hue each
    perturbed independently.
    """
    br = random.uniform(0.6, 1.4)
    co = random.uniform(0.6, 1.4)
    sa = random.uniform(0.6, 1.4)
    img = ImageEnhance.Brightness(img).enhance(br)
    img = ImageEnhance.Contrast(img).enhance(co)
    img = ImageEnhance.Color(img).enhance(sa)
    return img



def grayscale_convert(img: Image.Image) -> Image.Image:
    """Convert to grayscale but keep 3-channel output."""
    return ImageOps.grayscale(img).convert("RGB")



def solarize_image(img: Image.Image, threshold: int = 128) -> Image.Image:
    """
    Invert pixel values above `threshold` (classic solarisation effect).
    Used in self-supervised learning (SimCLR, BYOL).
    """
    return ImageOps.solarize(img, threshold)


# ─────────────────────────────────────────────
#  NOISE TRANSFORMS
# ─────────────────────────────────────────────


def add_gaussian_noise(img: Image.Image, std: float = 25.0) -> Image.Image:
    """Add zero-mean Gaussian noise with standard deviation `std`."""
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, std, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)



def add_salt_pepper_noise(img: Image.Image, amount: float = 0.05) -> Image.Image:
    """
    Replace `amount` fraction of pixels with salt (255) or pepper (0).
    """
    arr = np.array(img, dtype=np.uint8).copy()
    total = arr.shape[0] * arr.shape[1]
    n_salt = int(total * amount / 2)
    n_pepper = int(total * amount / 2)

    ys = np.random.randint(0, arr.shape[0], n_salt)
    xs = np.random.randint(0, arr.shape[1], n_salt)
    arr[ys, xs] = 255

    ys = np.random.randint(0, arr.shape[0], n_pepper)
    xs = np.random.randint(0, arr.shape[1], n_pepper)
    arr[ys, xs] = 0

    return Image.fromarray(arr)



def blur_image(img: Image.Image, radius: int = 3) -> Image.Image:
    """Gaussian blur with given radius."""
    return img.filter(ImageFilter.GaussianBlur(radius=radius))



def sharpen_image(img: Image.Image, factor: float = 2.5) -> Image.Image:
    """Unsharp-mask based sharpening."""
    return ImageEnhance.Sharpness(img).enhance(factor)


# ─────────────────────────────────────────────
#  ADVANCED / REGULARISATION TRANSFORMS
# ─────────────────────────────────────────────


def cutout_augmentation(img: Image.Image, size_pct: int = 20) -> Image.Image:
    """
    Cutout (DeVries & Taylor, 2017): zero out a random square patch.
    Encourages the model to use the whole image, not a single cue.
    """
    arr = np.array(img).copy()
    H, W = arr.shape[:2]
    ph = int(H * size_pct / 100)
    pw = int(W * size_pct / 100)
    cy = random.randint(0, H)
    cx = random.randint(0, W)
    y1 = max(0, cy - ph // 2);  y2 = min(H, cy + ph // 2)
    x1 = max(0, cx - pw // 2);  x2 = min(W, cx + pw // 2)
    arr[y1:y2, x1:x2] = 0
    return Image.fromarray(arr)



def random_erasing(img: Image.Image) -> Image.Image:
    """
    Random Erasing (Zhong et al., 2020): erase a random rectangle and
    fill with random noise (more aggressive than Cutout).
    """
    arr = np.array(img).copy()
    H, W = arr.shape[:2]
    area = H * W
    for _ in range(100):
        e_area = random.uniform(0.02, 0.33) * area
        aspect = random.uniform(0.3, 3.3)
        eh = int(math.sqrt(e_area * aspect))
        ew = int(math.sqrt(e_area / aspect))
        if eh >= H or ew >= W:
            continue
        y = random.randint(0, H - eh)
        x = random.randint(0, W - ew)
        arr[y:y+eh, x:x+ew] = np.random.randint(0, 256, (eh, ew, 3), dtype=np.uint8)
        break
    return Image.fromarray(arr)



def elastic_transform(img: Image.Image, alpha: float = 40.0, sigma: float = 6.0) -> Image.Image:
    """
    Elastic deformation (Simard et al., 2003) — used heavily in medical
    image segmentation (U-Net).  Random smooth displacement fields are
    applied to warp the image.
    """
    arr = np.array(img, dtype=np.float32)
    H, W = arr.shape[:2]

    # Random displacement fields
    dx = np.random.uniform(-1, 1, (H, W)).astype(np.float32)
    dy = np.random.uniform(-1, 1, (H, W)).astype(np.float32)

    # Smooth with Gaussian (approximated by blur)
    dx_img = Image.fromarray(((dx + 1) / 2 * 255).astype(np.uint8))
    dy_img = Image.fromarray(((dy + 1) / 2 * 255).astype(np.uint8))
    dx_img = dx_img.filter(ImageFilter.GaussianBlur(radius=sigma))
    dy_img = dy_img.filter(ImageFilter.GaussianBlur(radius=sigma))
    dx = (np.array(dx_img, np.float32) / 255 * 2 - 1) * alpha
    dy = (np.array(dy_img, np.float32) / 255 * 2 - 1) * alpha

    # Build coordinate grid
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    map_x = np.clip(xs + dx, 0, W - 1).astype(np.float32)
    map_y = np.clip(ys + dy, 0, H - 1).astype(np.float32)

    # Nearest-neighbour remap
    map_xi = map_x.round().astype(int)
    map_yi = map_y.round().astype(int)
    result = arr[map_yi, map_xi].astype(np.uint8)
    return Image.fromarray(result)
