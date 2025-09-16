from __future__ import annotations

import io
import numpy as np
import streamlit as st
import requests
from PIL import Image, ImageOps
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)

def exif_safe_open(img_file) -> Image.Image:
    """Open image with EXIF orientation applied and convert to RGB."""
    img = Image.open(img_file).convert("RGB")
    return ImageOps.exif_transpose(img)

def center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))

def preprocess_image(img: Image.Image, target_size=(224, 224), do_center_crop=True):
    if do_center_crop:
        img = center_crop_square(img)
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    arr = np.array(img).astype(np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr

@st.cache_data(show_spinner=False)
def fetch_image_from_url(url: str) -> Image.Image:
    """Minimal URL fetch with a couple of sanity checks and a browser-like UA."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0 Safari/537.36"
        )
    }
    r = requests.get(url, timeout=10, headers=headers, allow_redirects=True)
    r.raise_for_status()
    ctype = r.headers.get("content-type", "")
    if "image" not in ctype.lower():
        # Try to open anyway — some servers don't send proper content-type
        pass
    return exif_safe_open(io.BytesIO(r.content))

@st.cache_resource(show_spinner=False)
def load_model():
    return MobileNetV2(weights="imagenet")

def main():
    st.set_page_config(page_title="AI Image Classifier", page_icon="🖼️", layout="centered")

    st.title("AI Image Classifier 🖼️")
    st.write("Upload an image or paste a URL and let AI tell you what is in it!")

    with st.sidebar:
        st.header("Settings")
        topk = st.slider("Top‑K", min_value=1, max_value=5, value=3, step=1)
        threshold = st.slider("Confidence threshold", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
        do_center_crop = st.checkbox("Center‑crop (more stable results)", value=True)

    model = load_model()

    if "img_bytes" not in st.session_state:
        st.session_state.img_bytes = None
    if "img_source" not in st.session_state:
        st.session_state.img_source = None

    # 1) File upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"]) 
    if uploaded_file is not None:
        try:
            st.session_state.img_bytes = uploaded_file.read()
            st.session_state.img_source = "file"
        except Exception as e:
            st.error(f"Failed to read uploaded image: {e}")

    # 2) URL input (minimalist UI)
    url = st.text_input("or paste an Image URL", placeholder="https://...")
    load_btn = st.button("Load from URL")

    if load_btn and url:
        try:
            img = fetch_image_from_url(url)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            st.session_state.img_bytes = buf.getvalue()
            st.session_state.img_source = "URL"
        except Exception as e:
            st.error(f"Failed to load from URL: {e}")

    image = None
    if st.session_state.img_bytes is not None:
        try:
            image = exif_safe_open(io.BytesIO(st.session_state.img_bytes))
        except Exception as e:
            st.error(f"Could not decode image from memory: {e}")

    if image is not None:
        st.image(image, caption=f"Source: {st.session_state.img_source}", use_container_width=True)
        st.info("Clicking Classify will be enabled in the next step.")

    else:
        st.info("Upload a file, or paste a URL and click \"Load from URL\".")


if __name__ == "__main__":
    main()
