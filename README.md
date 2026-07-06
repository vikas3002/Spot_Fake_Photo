# Spot the Fake Photo — real vs. screen-recapture detector

## Approach

No deep learning, on purpose — with only ~100 training photos a CNN would
either overfit or need heavy augmentation to generalize, and it'd be
overkill for a phone. Instead I hand-crafted 9 features that target the
actual *physics* of "photo of a screen/printout," then feed them to either:

- a hand-tuned heuristic (works with **zero** training data — `predict.py`
  falls back to this automatically if no model has been trained yet), or
- a small `RandomForestClassifier` (`train.py`) once you have your real
  50+50 photos — this is what you should actually ship.

**The features** (full reasoning in `features.py` docstrings):

| Feature | Signal |
|---|---|
| `fft_peak_score`, `fft_high_freq_ratio` | Moiré: a screen's pixel grid beats against the camera sensor's grid, producing sharp non-natural spikes in the 2D frequency spectrum. **Strongest single signal in testing.** |
| `channel_decorr` | Screens render color via spatially-separated R/G/B subpixels → color fringing that decorrelates channels at high frequency. Real edges move together across channels. |
| `saturation_mean/std`, `specular_ratio` | Screens run punchier/backlit color and produce glare/reflections (blown-out, low-saturation highlights). |
| `bezel_score` | Detects a rectangular contour (screen edge/bezel/printout border) sitting inside the frame via contour + `minAreaRect`. Very reliable when present, silent (0) on tight crops. |
| `laplacian_var`, `edge_density` | Sharpness/edge profile — recapture goes through two optical systems. |

All features are computed on a 512px-resized image with just OpenCV +
NumPy — no GPU, no neural net weights to ship.

## Validating the code

I don't have your phone's photos, so I generated synthetic "real-like"
(smooth correlated-RGB content, no grid) and "screen-like" (added
periodic grid + phase-shifted per-channel fringing + glare + bezel)
images to confirm the whole pipeline runs end-to-end — see
`sanity_check_synthetic_data/`. On that synthetic set: heuristic
correctly separates real (0.26) vs. screen (0.75), and a trained
RandomForest hits 100% cross-validated accuracy. **This is a code
smoke-test, not a real accuracy number** — synthetic moiré is much
cleaner than real-world moiré. Once you drop your real ~50/~50 photos
into `real/` and `screen/` and run `python train.py`, it will print
real cross-validated accuracy — that's the number to trust for the 95%
bar.

## How to run it

```bash
pip install -r requirements.txt

# 1. Put ~50 real photos in real/, ~50 screen-recapture photos in screen/
# 2. Train (optional but recommended — more accurate than the heuristic)
python train.py
# -> prints cross-validated accuracy, saves model.joblib

# 3. Predict
python predict.py some_image.jpg
# -> 0.93   (0 = real, 1 = screen; uses model.joblib if present, else heuristic)
```

## Two ways to try it interactively

**Streamlit app (recommended — uses the real Python model/heuristic):**

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Opens a local web page where you can upload a photo or use `st.camera_input`
(works on desktop and mobile browsers) and see the live score, the
verdict, latency, and the raw feature values. It uses `model.joblib` if
present, otherwise the same heuristic fallback as `predict.py`.

**`demo.html` (browser-only, JS port, no Python needed):**

Browsers block camera access on a plain `file://` page, so it needs to
be served, not double-clicked open:

```bash
# from the spot_fake_photo/ folder
python3 -m http.server 8000
# then open http://localhost:8000/demo.html on the same machine
```

On a phone, open it over HTTPS (e.g. deploy to GitHub Pages / Netlify,
or use `ngrok http 8000` for a quick tunnel) — mobile browsers require a
secure context for camera access, `localhost` on the same device is the
only HTTP exception. This page ports a simplified version of the
moiré/glare/bezel logic to vanilla JS with a small in-browser FFT,
running fully on-device — it's a separate, lighter-weight approximation
of `features.py`, not a 1:1 copy.

## Required numbers

**Latency** (this sandboxed container's CPU, single image, 512px resize):
- Warm (model already loaded in memory, as it would be in a real service
  or app): **~55–60 ms/image** — this is the number that matters for
  production, since you load the model/weights once and reuse it.
- Cold process (fresh `python3` invocation incl. importing OpenCV/
  scikit-learn and loading the model from disk): **~2 s** — a one-time
  cost per process, not per image; irrelevant once running as a
  persistent service or compiled into a phone app.
- No GPU used anywhere.

**Cost per image:**
- **On-device (phone):** effectively $0 marginal cost — no network call,
  no server. This is the right target given the stated "runs on a
  phone" constraint; the feature set is cheap enough (basic OpenCV ops,
  no neural net) to port to a mobile CV library.
- **Cloud server (if centralizing instead):** at ~60 ms CPU-only per
  image, a single modest CPU core does ~16 images/sec ≈ 1,000/min. On a
  $0.05/hr small cloud CPU instance, that's roughly **$0.05–$0.10 per
  million images** for compute alone (assumption: steady load, no
  per-request overhead, ignoring network/storage/orchestration, which
  usually dominate real cloud bills more than the compute itself).

## What I'd improve with more time

1. **Calibrate on real photos.** The heuristic's thresholds and the RF's
   decision boundary are only as good as the data — synthetic moiré
   isn't a substitute for real screens (LCD vs. OLED vs. printed halftone
   all look different in frequency space).
2. **Multi-scale moiré analysis** — moiré period depends on
   screen-distance-from-camera; analyzing a couple of crop scales would
   catch both close-up and far shots.
3. **Robustness to adversarial cropping** — someone could zoom in tight
   enough on a screen to hide the bezel and blur out the grid; edge/
   glare features help here but I'd want real adversarial examples.
4. **On-device port** — re-implement `features.py` in a mobile-native CV
   library (Android CameraX + OpenCV, or Core Image on iOS) instead of
   the JS demo, for true production latency/battery numbers.
5. **Active-light trick** — if the app controls flash, firing it and
   checking for expected specular response could add a very cheap,
   very hard-to-fake extra signal (screens/printouts respond differently
   to flash than real 3D scenes).
