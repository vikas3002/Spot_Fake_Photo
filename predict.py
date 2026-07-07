#!/usr/bin/env python3
"""
predict.py
----------
Usage:
    python predict.py some_image.jpg

Prints a single number from 0 to 1:
    0 = REAL photo
    1 = PHOTO OF A SCREEN (recapture)

Behavior:
    - If model.joblib exists (produced by `python train.py` on your real/
      and screen/ folders), uses that trained classifier.
    - Otherwise, falls back to a hand-tuned heuristic built directly from
      the feature physics in features.py, so the tool is usable on day
      zero before you've collected/trained on any data. Train a model as
      soon as you have your 100 photos - it will be more accurate than
      the heuristic.
"""

import argparse
import math
import os
import sys
import time

from features import extract_features, FEATURE_ORDER

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.joblib")


def _sigmoid(x, mid, scale):
    z = (x - mid) / scale
    z = max(-30, min(30, z))
    return 1.0 / (1.0 + math.exp(-z))


def heuristic_score(feats: dict) -> float:
    """Hand-tuned combination of features, no training required.
    Weights reflect how strongly/reliably each signal indicates a screen
    recapture (see features.py docstring for the reasoning)."""

    # Moire/pixel-grid peakiness -> higher means more likely a screen.
    # This is the strongest, most reliable signal in testing.
    s_fft_peak = _sigmoid(feats["fft_peak_score"], mid=4.0, scale=0.5)

    # High-frequency energy ratio -> higher means more likely a screen
    s_fft_hf = _sigmoid(feats["fft_high_freq_ratio"], mid=0.11, scale=0.02)

    # RGB channel decorrelation -> in theory, LOWER correlation means more
    # likely a screen (subpixel fringing). Kept in the vector for the
    # trainable model (train.py), but given a small weight here since the
    # hand-tuned midpoint hasn't been calibrated on real photos yet.
    s_decorr = 1.0 - _sigmoid(feats["channel_decorr"], mid=0.5, scale=0.2)

    # Glare / specular blowout -> higher means more likely a screen
    s_specular = _sigmoid(feats["specular_ratio"], mid=0.006, scale=0.006)

    # Rectangular bezel/frame inside the shot -> higher means more likely
    # a screen or printout. Very reliable when a bezel is visible; silent
    # (0) when the shot is cropped tight, which is common and fine.
    s_bezel = _sigmoid(feats["bezel_score"], mid=0.30, scale=0.15)

    weights = {
        "fft_peak": 0.40,
        "fft_hf": 0.15,
        "decorr": 0.10,
        "specular": 0.15,
        "bezel": 0.20,
    }
    score = (
        weights["fft_peak"] * s_fft_peak
        + weights["fft_hf"] * s_fft_hf
        + weights["decorr"] * s_decorr
        + weights["specular"] * s_specular
        + weights["bezel"] * s_bezel
    )
    return float(min(1.0, max(0.0, score)))


def model_score(image_path, model_bundle) -> float:
    _, vec = extract_features(image_path)
    pipeline = model_bundle["pipeline"]
    # class 1 = "screen" (see train.py labeling)
    proba = pipeline.predict_proba([vec])[0]
    classes = list(pipeline.classes_)
    screen_idx = classes.index(1)
    return float(proba[screen_idx])


def predict(image_path: str) -> float:
    feats, _ = extract_features(image_path)

    if os.path.exists(MODEL_PATH):
        try:
            import joblib
            bundle = joblib.load(MODEL_PATH)
            return model_score(image_path, bundle)
        except Exception as e:
            sys.stderr.write(f"[predict.py] model load/predict failed ({e}), "
                              f"falling back to heuristic\n")

    return heuristic_score(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="path to the image to classify")
    ap.add_argument("--timing", action="store_true", help="print latency to stderr")
    args = ap.parse_args()

    t0 = time.time()
    score = predict(args.image)
    dt_ms = (time.time() - t0) * 1000.0

    print(f"{score:.4f}")
    if args.timing:
        sys.stderr.write(f"[predict.py] {dt_ms:.1f} ms\n")


if __name__ == "__main__":
    main()
