import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import io
import zipfile
import random

from augmentation import (
    rotate_image, flip_image, zoom_image, translate_image,
    brightness_adjust, contrast_adjust,
    color_jitter, grayscale_convert, solarize_image
)

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Image Data Augmentation Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0e1117;
    color: #fafafa;
}

.stApp {
    max-width: 100%;
    margin: 0;
}

.main-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
    background: linear-gradient(135deg, #6366f1, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1.5px;
}

.sub-title {
    text-align: center;
    color: #94a3b8;
    font-size: 1.1rem;
    margin-bottom: 3rem;
    font-weight: 400;
}

h2, h3 {
    text-align: center;
    font-weight: 600;
    color: #f8fafc;
    margin-top: 2rem;
    margin-bottom: 1.5rem;
}

[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #1f2937;
}

.stMetric {
    background-color: #1f2937;
    border: 1px solid #374151;
    border-radius: 12px;
    padding: 1.5rem !important;
    text-align: center;
}

.stButton > button {
    width: 100%;
    background-color: #4f46e5;
    color: white;
    border-radius: 8px;
    padding: 0.75rem;
    font-weight: 600;
    border: none;
    transition: all 0.2s;
}

.stButton > button:hover {
    background-color: #4338ca;
    border: none;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
}

[data-testid="stFileUploader"] {
    border: 2px dashed #374151;
    border-radius: 12px;
    background-color: #1f2937;
}

.stImage img {
    border-radius: 12px;
    border: 1px solid #374151;
}

.center-box {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
}

.download-section {
    background-color: #111827;
    border: 1px solid #1f2937;
    border-radius: 16px;
    padding: 2rem;
    margin-top: 3rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:

    st.markdown("### ⚙️ Augmentation Studio")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Supports JPG, PNG, BMP, WEBP"
    )

# ─────────────────────────────────────────────
#  DEFAULT SETTINGS
# ─────────────────────────────────────────────
do_rotate = True
rot_angle = 45

do_flip = True
flip_mode = "horizontal"

do_zoom = False
zoom_factor = 1.3

do_translate = False
trans_pct = 15

do_bright = True
bright_r = (0.6, 1.4)

do_contrast = False
cont_r = (0.6, 1.4)

do_jitter = True

do_gray = False

do_solarize = False

num_augmented = 8

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def collect_active_techniques(cfg: dict):

    names = []

    if cfg["do_rotate"]:
        names.append("rotation")

    if cfg["do_flip"]:
        names.append("flip")

    if cfg["do_zoom"]:
        names.append("zoom")

    if cfg["do_translate"]:
        names.append("translation")

    if cfg["do_bright"]:
        names.append("brightness")

    if cfg["do_contrast"]:
        names.append("contrast")

    if cfg["do_jitter"]:
        names.append("color_jitter")

    if cfg["do_gray"]:
        names.append("grayscale")

    if cfg["do_solarize"]:
        names.append("solarize")

    return names


def apply_random_augmentation(img: Image.Image, cfg: dict):

    techniques = collect_active_techniques(cfg)

    if not techniques:
        return img, "original"

    tech = random.choice(techniques)

    if tech == "rotation":

        angle = random.uniform(
            -cfg["rot_angle"],
            cfg["rot_angle"]
        )

        return rotate_image(
            img,
            angle
        ), f"rotate {angle:.0f}°"

    elif tech == "flip":

        return flip_image(
            img,
            cfg["flip_mode"]
        ), f"flip ({cfg['flip_mode']})"

    elif tech == "zoom":

        return zoom_image(
            img,
            cfg["zoom_factor"]
        ), f"zoom ×{cfg['zoom_factor']}"

    elif tech == "translation":

        dx = random.randint(
            -cfg["trans_pct"],
            cfg["trans_pct"]
        )

        dy = random.randint(
            -cfg["trans_pct"],
            cfg["trans_pct"]
        )

        return translate_image(
            img,
            dx,
            dy
        ), f"translate ({dx},{dy})%"

    elif tech == "brightness":

        f = random.uniform(*cfg["bright_r"])

        return brightness_adjust(
            img,
            f
        ), f"brightness ×{f:.2f}"

    elif tech == "contrast":

        f = random.uniform(*cfg["cont_r"])

        return contrast_adjust(
            img,
            f
        ), f"contrast ×{f:.2f}"

    elif tech == "color_jitter":

        return color_jitter(img), "color jitter"

    elif tech == "grayscale":

        return grayscale_convert(img), "grayscale"

    elif tech == "solarize":

        return solarize_image(img), "solarize"

    return img, "original"


def images_to_zip(original: Image.Image, augmented: list):

    buf = io.BytesIO()

    with zipfile.ZipFile(buf, "w") as zf:

        img_buf = io.BytesIO()

        original.save(
            img_buf,
            format="PNG"
        )

        zf.writestr(
            "original.png",
            img_buf.getvalue()
        )

        for i, (img, label) in enumerate(augmented):

            img_buf = io.BytesIO()

            img.save(
                img_buf,
                format="PNG"
            )

            safe = "".join([
                c if c.isalnum() or c in "._-"
                else "_"
                for c in label
            ])

            zf.writestr(
                f"aug_{i+1:02d}_{safe}.png",
                img_buf.getvalue()
            )

    return buf.getvalue()

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
st.markdown(
    '<h1 class="main-title">Augmentation Studio</h1>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-title">A professional toolkit for image dataset expansion.</p>',
    unsafe_allow_html=True
)

if uploaded is None:

    st.markdown(
        '<div class="center-box">',
        unsafe_allow_html=True
    )

    st.info(
        "👋 Welcome! Please upload an image in the sidebar to begin."
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

else:

    original = Image.open(uploaded).convert("RGB")

    W, H = original.size

    cfg = dict(
        do_rotate=do_rotate,
        rot_angle=rot_angle,

        do_flip=do_flip,
        flip_mode=flip_mode,

        do_zoom=do_zoom,
        zoom_factor=zoom_factor,

        do_translate=do_translate,
        trans_pct=trans_pct,

        do_bright=do_bright,
        bright_r=bright_r,

        do_contrast=do_contrast,
        cont_r=cont_r,

        do_jitter=do_jitter,

        do_gray=do_gray,

        do_solarize=do_solarize,
    )

    active = collect_active_techniques(cfg)

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Original size", f"{W} × {H}")
    m2.metric("Active techniques", len(active))
    m3.metric("Images to generate", num_augmented)
    m4.metric("Dataset multiplier", f"{num_augmented + 1}×")

    st.markdown("---")

    st.markdown("### Configuration Overview")

    c1, c2, c3 = st.columns([1, 2, 1])

    with c2:

        st.image(
            original,
            caption="Original Image",
            use_container_width=True
        )

    st.markdown("---")

    if not active:

        st.error(
            "Please select at least one augmentation technique."
        )

    else:

        if "results" not in st.session_state:
            st.session_state.results = None

        if st.button(
            f"🚀 Generate {num_augmented} Augmented Images",
            use_container_width=True
        ):

            with st.spinner("Applying augmentations…"):

                results = []

                for _ in range(num_augmented):

                    aug_img, label = apply_random_augmentation(
                        original,
                        cfg
                    )

                    results.append((aug_img, label))

                st.session_state.results = results

            st.success(
                f"✅ Generated {num_augmented} augmented images!"
            )

        if st.session_state.results is not None:

            results = st.session_state.results

            st.markdown("## Augmented Results")

            cols_per_row = 4

            for row_start in range(
                0,
                len(results),
                cols_per_row
            ):

                row_items = results[
                    row_start:row_start + cols_per_row
                ]

                cols = st.columns(cols_per_row)

                for col, (img, label) in zip(cols, row_items):

                    with col:

                        st.image(
                            img,
                            caption=label,
                            use_container_width=True
                        )

            st.markdown("---")

            st.markdown(
                '<div class="download-section">',
                unsafe_allow_html=True
            )

            st.markdown("### 📦 Export Dataset")

            zip_bytes = images_to_zip(
                original,
                results
            )

            st.download_button(
                label="Download ZIP Archive",
                data=zip_bytes,
                file_name="augmented_dataset.zip",
                mime="application/zip",
                use_container_width=True
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )