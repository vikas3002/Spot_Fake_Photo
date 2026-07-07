"""
streamlit_app.py
----------------
A small UI around predict.py's logic - upload a photo or snap one with
your webcam, see the REAL vs SCREEN score live.

Run:
    pip install -r requirements.txt streamlit
    streamlit run streamlit_app.py
"""

import os
import time

import cv2
import numpy as np
import streamlit as st

from features import extract_features, FEATURE_ORDER
from predict import heuristic_score, MODEL_PATH

st.set_page_config(page_title="Spot the Fake Photo", page_icon="📸", layout="centered")

st.title("📸 Spot the Fake Photo")
st.caption(
    "Upload a photo or use your webcam. The model scores it from 0 (REAL) "
    "to 1 (PHOTO OF A SCREEN / recapture)."
)

# ---- load model once, cache across reruns ----
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        import joblib
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            st.warning(f"Found model.joblib but couldn't load it ({e}). Using heuristic instead.")
    return None

model_bundle = load_model()

if model_bundle is not None:
    st.success("Using your trained model (`model.joblib`).")
else:
    st.info(
        "No `model.joblib` found yet - using the built-in hand-tuned heuristic. "
        "Run `python train.py` on your real/ and screen/ folders for a more "
        "accurate, calibrated model."
    )

# ---- input: tabs for upload vs webcam ----
tab_upload, tab_camera = st.tabs(["📁 Upload a photo", "🎥 Use webcam"])

image_bgr = None
source_label = None

with tab_upload:
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded is not None:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        source_label = uploaded.name

with tab_camera:
    snap = st.camera_input("Take a photo")
    if snap is not None:
        file_bytes = np.frombuffer(snap.getvalue(), np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        source_label = "webcam capture"

# ---- run prediction ----
if image_bgr is not None:
    st.divider()
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption=source_label, use_container_width=True)

    with col2:
        t0 = time.time()
        feats, vec = extract_features(image_bgr)

        if model_bundle is not None:
            pipeline = model_bundle["pipeline"]
            proba = pipeline.predict_proba([vec])[0]
            classes = list(pipeline.classes_)
            score = float(proba[classes.index(1)])
            method = "trained model"
        else:
            score = heuristic_score(feats)
            method = "heuristic"

        dt_ms = (time.time() - t0) * 1000

        if score > 0.6:
            st.error(f"🖥️ Likely a SCREEN / recapture — score {score:.3f}")
        elif score < 0.4:
            st.success(f"✅ Likely REAL — score {score:.3f}")
        else:
            st.warning(f"🤔 Uncertain — score {score:.3f}")

        st.progress(score, text=f"REAL 0.0 ←→ 1.0 SCREEN  ({method})")
        st.caption(f"Inference: {dt_ms:.1f} ms (feature extraction + scoring, model already warm)")

        with st.expander("Raw feature values"):
            for name in FEATURE_ORDER:
                st.write(f"**{name}**: {feats[name]:.4f}")
else:
    st.write("👆 Upload a photo or take one with your webcam to get a score.")

st.divider()
st.caption(
    "Tip: try photographing a real object, then hold your phone/laptop screen "
    "showing a picture of the same thing up to the camera - watch the score flip."
)
