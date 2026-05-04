import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import io
import zipfile
import random
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from augmentation import (
    rotate_image, flip_image, zoom_image, translate_image,
    brightness_adjust, contrast_adjust, add_gaussian_noise,
    add_salt_pepper_noise, blur_image, sharpen_image,
    color_jitter, cutout_augmentation, elastic_transform,
    random_erasing, grayscale_convert, solarize_image
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
#  CUSTOM CSS  (dark industrial / refined look)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono&display=swap');

/* ── Global Styles ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0e1117;
    color: #fafafa;
}

.stApp {
    max-width: 100%;
    margin: 0;
}

/* ── Centered Headings ── */
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

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #1f2937;
}

/* ── Cards & Containers ── */
.stMetric {
    background-color: #1f2937;
    border: 1px solid #374151;
    border-radius: 12px;
    padding: 1.5rem !important;
    text-align: center;
}

[data-testid="stExpander"] {
    background-color: #111827;
    border: 1px solid #1f2937 !important;
    border-radius: 8px !important;
}

/* ── Buttons ── */
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

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #374151;
    border-radius: 12px;
    background-color: #1f2937;
}

/* ── Images ── */
.stImage img {
    border-radius: 12px;
    border: 1px solid #374151;
}

/* ── Centered Content Helper ── */
.center-box {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
}

/* ── Clean Tabs ── */
[data-testid="stTabs"] {
    border-bottom: 1px solid #1f2937;
}

[data-testid="stTabs"] button {
    color: #94a3b8;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    color: #6366f1;
    border-bottom: 2px solid #6366f1;
}

/* ── Download Area ── */
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

    st.markdown("---")
    st.markdown("### ⚙️ Technique Selection")

    # ── GEOMETRIC
    with st.expander("📐 Geometric", expanded=True):
        do_rotate   = st.checkbox("Rotation", value=True)
        rot_angle   = st.slider("Max angle (°)", 5, 180, 45) if do_rotate else 45

        do_flip     = st.checkbox("Flip", value=True)
        flip_mode   = st.selectbox("Direction", ["horizontal","vertical","both"]) if do_flip else "horizontal"

        do_zoom     = st.checkbox("Zoom / Crop", value=False)
        zoom_factor = st.slider("Zoom factor", 1.1, 2.0, 1.3, 0.1) if do_zoom else 1.3

        do_translate = st.checkbox("Translation", value=False)
        trans_pct    = st.slider("Shift %", 5, 40, 15) if do_translate else 15

    # ── COLOR
    with st.expander("🎨 Color & Brightness"):
        do_bright = st.checkbox("Brightness", value=True)
        bright_r  = st.slider("Range", 0.3, 2.0, (0.6, 1.4)) if do_bright else (0.6, 1.4)

        do_contrast = st.checkbox("Contrast", value=False)
        cont_r      = st.slider("Contrast Range", 0.3, 2.0, (0.6, 1.4)) if do_contrast else (0.6, 1.4)

        do_jitter   = st.checkbox("Color Jitter", value=True)
        do_gray     = st.checkbox("Grayscale", value=False)
        do_solarize = st.checkbox("Solarize", value=False)

    # ── NOISE
    with st.expander("🌫️ Noise & Blur"):
        do_gauss  = st.checkbox("Gaussian Noise", value=True)
        gauss_std = st.slider("Noise std", 5, 80, 25) if do_gauss else 25

        do_sp   = st.checkbox("Salt & Pepper", value=False)
        sp_amt  = st.slider("S&P amount", 0.01, 0.2, 0.05) if do_sp else 0.05

        do_blur   = st.checkbox("Blur", value=False)
        blur_r    = st.slider("Blur radius", 1, 10, 3) if do_blur else 3

        do_sharp  = st.checkbox("Sharpen", value=False)

    # ── ADVANCED
    with st.expander("🔬 Advanced"):
        do_cutout = st.checkbox("Cutout", value=False)
        cutout_sz = st.slider("Cutout size %", 5, 40, 20) if do_cutout else 20

        do_erasing = st.checkbox("Random Erasing", value=False)
        do_elastic = st.checkbox("Elastic Transform", value=False)

    st.markdown("---")
    num_augmented = st.slider("# of augmented images", 4, 20, 8)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def collect_active_techniques(cfg: dict) -> list[str]:
    """Return list of active technique names."""
    names = []
    if cfg["do_rotate"]:    names.append("rotation")
    if cfg["do_flip"]:      names.append("flip")
    if cfg["do_zoom"]:      names.append("zoom")
    if cfg["do_translate"]: names.append("translation")
    if cfg["do_bright"]:    names.append("brightness")
    if cfg["do_contrast"]:  names.append("contrast")
    if cfg["do_jitter"]:    names.append("color_jitter")
    if cfg["do_gray"]:      names.append("grayscale")
    if cfg["do_solarize"]:  names.append("solarize")
    if cfg["do_gauss"]:     names.append("gaussian_noise")
    if cfg["do_sp"]:        names.append("salt_pepper")
    if cfg["do_blur"]:      names.append("blur")
    if cfg["do_sharp"]:     names.append("sharpen")
    if cfg["do_cutout"]:    names.append("cutout")
    if cfg["do_erasing"]:   names.append("random_erasing")
    if cfg["do_elastic"]:   names.append("elastic_transform")
    return names


def apply_random_augmentation(img: Image.Image, cfg: dict) -> tuple[Image.Image, str]:
    """Apply one random technique from active list."""
    techniques = collect_active_techniques(cfg)
    if not techniques:
        return img, "original"
    tech = random.choice(techniques)

    if tech == "rotation":
        angle = random.uniform(-cfg["rot_angle"], cfg["rot_angle"])
        return rotate_image(img, angle), f"rotate {angle:.0f}°"
    elif tech == "flip":
        return flip_image(img, cfg["flip_mode"]), f"flip ({cfg['flip_mode']})"
    elif tech == "zoom":
        return zoom_image(img, cfg["zoom_factor"]), f"zoom ×{cfg['zoom_factor']}"
    elif tech == "translation":
        dx = random.randint(-cfg["trans_pct"], cfg["trans_pct"])
        dy = random.randint(-cfg["trans_pct"], cfg["trans_pct"])
        return translate_image(img, dx, dy), f"translate ({dx},{dy})%"
    elif tech == "brightness":
        f = random.uniform(*cfg["bright_r"])
        return brightness_adjust(img, f), f"brightness ×{f:.2f}"
    elif tech == "contrast":
        f = random.uniform(*cfg["cont_r"])
        return contrast_adjust(img, f), f"contrast ×{f:.2f}"
    elif tech == "color_jitter":
        return color_jitter(img), "color jitter"
    elif tech == "grayscale":
        return grayscale_convert(img), "grayscale"
    elif tech == "solarize":
        return solarize_image(img), "solarize"
    elif tech == "gaussian_noise":
        return add_gaussian_noise(img, cfg["gauss_std"]), f"gaussian noise σ={cfg['gauss_std']}"
    elif tech == "salt_pepper":
        return add_salt_pepper_noise(img, cfg["sp_amt"]), f"salt&pepper {cfg['sp_amt']}"
    elif tech == "blur":
        return blur_image(img, cfg["blur_r"]), f"blur r={cfg['blur_r']}"
    elif tech == "sharpen":
        return sharpen_image(img), "sharpen"
    elif tech == "cutout":
        return cutout_augmentation(img, cfg["cutout_sz"]), f"cutout {cfg['cutout_sz']}%"
    elif tech == "random_erasing":
        return random_erasing(img), "random erasing"
    elif tech == "elastic_transform":
        return elastic_transform(img), "elastic"
    return img, "original"


@st.cache_data(show_spinner=False)
def generate_preview_thumbnails(img_id: str, img: Image.Image, cfg: dict) -> dict:
    """Generate downsampled previews for all active techniques."""
    # Downsample for preview speed
    thumb = img.copy()
    thumb.thumbnail((512, 512))
    
    previews = {}
    if cfg["do_rotate"]:   previews["Rotation"] = rotate_image(thumb, cfg["rot_angle"])
    if cfg["do_flip"]:     previews["Flip"] = flip_image(thumb, cfg["flip_mode"])
    if cfg["do_zoom"]:     previews["Zoom"] = zoom_image(thumb, cfg["zoom_factor"])
    if cfg["do_translate"]: previews["Translate"] = translate_image(thumb, cfg["trans_pct"], cfg["trans_pct"])
    if cfg["do_bright"]:   previews["Brightness↑"] = brightness_adjust(thumb, cfg["bright_r"][1])
    if cfg["do_contrast"]: previews["Contrast↑"] = contrast_adjust(thumb, cfg["cont_r"][1])
    if cfg["do_jitter"]:   previews["Color Jitter"] = color_jitter(thumb)
    if cfg["do_gray"]:     previews["Grayscale"] = grayscale_convert(thumb)
    if cfg["do_solarize"]: previews["Solarize"] = solarize_image(thumb)
    if cfg["do_gauss"]:    previews[f"Gauss σ={cfg['gauss_std']}"] = add_gaussian_noise(thumb, cfg["gauss_std"])
    if cfg["do_sp"]:       previews[f"Salt&Pepper"] = add_salt_pepper_noise(thumb, cfg["sp_amt"])
    if cfg["do_blur"]:     previews[f"Blur r={cfg['blur_r']}"] = blur_image(thumb, cfg["blur_r"])
    if cfg["do_sharp"]:    previews["Sharpen"] = sharpen_image(thumb)
    if cfg["do_cutout"]:   previews[f"Cutout {cfg['cutout_sz']}%"] = cutout_augmentation(thumb, cfg["cutout_sz"])
    if cfg["do_erasing"]:  previews["Rand Erasing"] = random_erasing(thumb)
    if cfg["do_elastic"]:  previews["Elastic"] = elastic_transform(thumb)
    return previews


def images_to_zip(original: Image.Image, augmented: list) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        img_buf = io.BytesIO()
        original.save(img_buf, format="PNG")
        zf.writestr("original.png", img_buf.getvalue())
        for i, (img, label) in enumerate(augmented):
            img_buf = io.BytesIO()
            img.save(img_buf, format="PNG")
            # More robust sanitization for cross-platform filenames
            safe = "".join([c if c.isalnum() or c in "._-" else "_" for c in label])
            zf.writestr(f"aug_{i+1:02d}_{safe}.png", img_buf.getvalue())
    return buf.getvalue()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
#  MAIN HEADER
# ─────────────────────────────────────────────
st.markdown('<h1 class="main-title">Augmentation Studio</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">A professional toolkit for image dataset expansion and regularisation.</p>', unsafe_allow_html=True)

if uploaded is None:
    # ── CLEAN WELCOME STATE
    st.markdown('<div class="center-box">', unsafe_allow_html=True)
    st.info("👋 Welcome! Please upload an image in the sidebar to begin.")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── MINIMAL THEORY FOOTER
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    with st.expander("ℹ️ Learn about Data Augmentation", expanded=False):
        tab1, tab2, tab3 = st.tabs(["Overview", "Benefits", "Techniques"])

    with tab1:
        st.markdown("""
**Data augmentation** is the process of creating new training samples by applying label-preserving transformations to existing data.

It is one of the most effective regularisation strategies in deep learning, especially when training data is scarce.

**Common use-cases:**
- Image classification (ResNet, VGG, EfficientNet …)
- Object detection (YOLO, Faster R-CNN …)
- Segmentation (U-Net, DeepLab …)
- Medical imaging where data is expensive
        """)

    with tab2:
        st.markdown("""
**Why augmentation reduces overfitting:**

| Without Augmentation | With Augmentation |
|---|---|
| Model memorises exact pixels | Model learns invariances |
| Poor generalisation | Better test accuracy |
| Needs huge datasets | Works with smaller datasets |
| High variance | Lower variance |

Techniques like **Cutout** and **Random Erasing** specifically force the model to rely on distributed features rather than a single discriminative region.
        """)

    with tab3:
        st.markdown("""
| Technique | Category | Effect |
|---|---|---|
| Rotation | Geometric | Orientation invariance |
| Flip | Geometric | Mirror invariance |
| Zoom/Crop | Geometric | Scale invariance |
| Translation | Geometric | Position invariance |
| Brightness | Photometric | Lighting robustness |
| Contrast | Photometric | Exposure robustness |
| Color Jitter | Photometric | Color robustness |
| Gaussian Noise | Noise | Sensor noise robustness |
| Salt & Pepper | Noise | Impulse noise robustness |
| Blur | Noise | Focus invariance |
| Cutout | Advanced | Occlusion robustness |
| Elastic Transform | Advanced | Deformation invariance |
| Random Erasing | Advanced | Partial occlusion |
        """)

else:
    # ─────────────────────────────────────────────
    #  IMAGE LOADED STATE
    # ─────────────────────────────────────────────
    # Reset results if a new file is uploaded
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded.name:
        st.session_state.results = None
        st.session_state.current_file = uploaded.name
        
    original = Image.open(uploaded).convert("RGB")
    W, H = original.size

    cfg = dict(
        do_rotate=do_rotate, rot_angle=rot_angle,
        do_flip=do_flip, flip_mode=flip_mode,
        do_zoom=do_zoom, zoom_factor=zoom_factor,
        do_translate=do_translate, trans_pct=trans_pct,
        do_bright=do_bright, bright_r=bright_r,
        do_contrast=do_contrast, cont_r=cont_r,
        do_jitter=do_jitter, do_gray=do_gray, do_solarize=do_solarize,
        do_gauss=do_gauss, gauss_std=gauss_std,
        do_sp=do_sp, sp_amt=sp_amt,
        do_blur=do_blur, blur_r=blur_r,
        do_sharp=do_sharp,
        do_cutout=do_cutout, cutout_sz=cutout_sz,
        do_erasing=do_erasing,
        do_elastic=do_elastic,
    )

    active = collect_active_techniques(cfg)

    # ── Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Original size", f"{W} × {H}")
    m2.metric("Active techniques", len(active))
    m3.metric("Images to generate", num_augmented)
    m4.metric("Dataset multiplier", f"{num_augmented + 1}×")

    st.markdown("---")

    # ── Original image centered
    st.markdown("### Configuration Overview")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.image(original, caption="Original Image", use_container_width=True)

    st.markdown("---")

    if not active:
        st.error("Please select at least one augmentation technique from the sidebar.")
    else:
        # Initialize session state for results if not present
        if "results" not in st.session_state:
            st.session_state.results = None
        if "last_config" not in st.session_state:
            st.session_state.last_config = None

        if st.button(f"🚀 Generate {num_augmented} Augmented Images", use_container_width=True):
            with st.spinner("Applying augmentations…"):
                results = []
                for _ in range(num_augmented):
                    aug_img, label = apply_random_augmentation(original, cfg)
                    results.append((aug_img, label))
                st.session_state.results = results
                st.session_state.last_config = cfg
            st.success(f"✅ Generated {num_augmented} augmented images!")

        # Show results if they exist in session state
        if st.session_state.results is not None:
            results = st.session_state.results
            
            st.markdown("## Augmented Results")
            cols_per_row = 4
            for row_start in range(0, len(results), cols_per_row):
                row_items = results[row_start:row_start + cols_per_row]
                cols = st.columns(cols_per_row)
                for col, (img, label) in zip(cols, row_items):
                    with col:
                        st.image(img, caption=label, use_container_width=True)

            # ── Pixel distribution comparison
            st.markdown("---")
            st.markdown("## 📊 Pixel Distribution Analysis")

            # Faster histogram calculation via sampling
            @st.cache_data(show_spinner=False)
            def get_hist_data(_img):
                arr = np.array(_img)
                # Subsample pixels for speed (e.g., 1 out of every 16 pixels)
                sample = arr[::4, ::4, :]
                hists = []
                for i in range(3):
                    counts, bins = np.histogram(sample[:,:,i], bins=64, range=(0, 255), density=True)
                    hists.append((counts, bins[:-1]))
                return hists

            fig = Figure(figsize=(12, 4), facecolor="#0e1117")
            axes = fig.subplots(1, 2)
            colors = ["#6366f1", "#10b981", "#f59e0b"]
            
            comparison_img, _ = results[0]
            
            for ax, (image, title) in zip(axes, [(original, "Original Distribution"), (comparison_img, "Augmented Distribution")]):
                hist_data = get_hist_data(image)
                for i, (counts, bins) in enumerate(hist_data):
                    ax.fill_between(bins, counts, color=colors[i], alpha=0.3)
                    ax.plot(bins, counts, color=colors[i], lw=1.5, label=["Red","Green","Blue"][i])
                
                ax.set_title(title, color="#f8fafc", fontsize=10, fontweight='bold')
                ax.set_facecolor("#111827")
                ax.tick_params(colors="#94a3b8", labelsize=8)
                ax.set_ylim(bottom=0)
                for spine in ax.spines.values():
                    spine.set_color("#1f2937")
                ax.legend(fontsize=7, facecolor="#111827", labelcolor="#94a3b8", frameon=False)
            
            fig.tight_layout()
            st.pyplot(fig)

            # ── Download Section
            st.markdown('<div class="download-section">', unsafe_allow_html=True)
            st.markdown("### 📦 Export Dataset")
            zip_bytes = images_to_zip(original, results)
            st.download_button(
                label="Download ZIP Archive",
                data=zip_bytes,
                file_name="augmented_dataset.zip",
                mime="application/zip",
                use_container_width=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Previews
        st.markdown("---")
        with st.expander("🔍 View Per-Technique Previews", expanded=False):
            st.caption("Thumbnails are downsampled for speed. Exports remain high-resolution.")
            
            preview_map = generate_preview_thumbnails(uploaded.name, original, cfg)
            
            if preview_map:
                items = list(preview_map.items())
                for row_start in range(0, len(items), 4):
                    row = items[row_start:row_start+4]
                    cols = st.columns(4)
                    for col, (name, img) in zip(cols, row):
                        with col:
                            st.image(img, caption=name, use_container_width=True)
