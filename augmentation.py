import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import random
import math


# ─────────────────────────────────────────────
#  GEOMETRIC TRANSFORMS
# ─────────────────────────────────────────────

def rotate_image(img: Image.Image, angle: float) -> Image.Image:
    # Cache mean calculation
    arr = np.array(img)
    fill = tuple(int(c) for c in arr.mean(axis=(0, 1)))
    return img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=fill)



def flip_image(img: Image.Image, mode: str = "horizontal") -> Image.Image:
    if mode == "horizontal":
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    elif mode == "vertical":
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    else:  # both
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img.transpose(Image.FLIP_TOP_BOTTOM)



def zoom_image(img: Image.Image, zoom: float = 1.3) -> Image.Image:
    W, H = img.size
    new_W = int(W / zoom)
    new_H = int(H / zoom)
    left = (W - new_W) // 2
    top  = (H - new_H) // 2
    cropped = img.crop((left, top, left + new_W, top + new_H))
    return cropped.resize((W, H), Image.Resampling.BILINEAR)



def translate_image(img: Image.Image, dx_pct: int = 15, dy_pct: int = 15) -> Image.Image:
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
    return ImageEnhance.Brightness(img).enhance(factor)



def contrast_adjust(img: Image.Image, factor: float = 1.4) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(factor)



def color_jitter(img: Image.Image) -> Image.Image:
    br = random.uniform(0.6, 1.4)
    co = random.uniform(0.6, 1.4)
    sa = random.uniform(0.6, 1.4)
    img = ImageEnhance.Brightness(img).enhance(br)
    img = ImageEnhance.Contrast(img).enhance(co)
    img = ImageEnhance.Color(img).enhance(sa)
    return img



def grayscale_convert(img: Image.Image) -> Image.Image:
    return ImageOps.grayscale(img).convert("RGB")



def solarize_image(img: Image.Image, threshold: int = 128) -> Image.Image:
    return ImageOps.solarize(img, threshold)


