# 📸 Spot the Fake Photo

Detect whether an image is a **real photo** or a **photo of a screen/printout** (a "recapture") — built for catching people who screenshot or photograph a screen instead of taking a genuine photo.

```
python predict.py some_image.jpg
0.93
```
`0.0` = real photo · `1.0` = photo of a screen

No deep learning required — this uses hand-crafted computer-vision features that target the actual physics of photographing a screen (moiré interference, color-channel decorrelation, glare, bezels), so it stays small, fast, and explainable enough to eventually run on a phone.

---

## Table of contents

- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [Try it interactively](#try-it-interactively)
- [Training on your own photos](#training-on-your-own-photos)
- [Project structure](#project-structure)
- [Results](#results)
- [Latency & cost](#latency--cost)
- [Roadmap](#roadmap--whats-next)
- [Pushing to GitHub](#pushing-to-github-without-your-dataset-or-venv)

---

## How it works

Every "photo of a screen" goes through **two optical systems** instead of one (screen/printer + camera), and that leaves fingerprints a real photo doesn't have. Nine features capture those fingerprints:

| Feature | What it detects | Why it works |
|---|---|---|
| `fft_peak_score` | Moiré interference | A screen's pixel grid beats against the camera sensor's grid, producing sharp, non-natural spikes in the 2D frequency spectrum. **Strongest single signal.** |
| `fft_high_freq_ratio` | High-frequency energy | Screens/halftone prints carry more structured high-frequency content than natural scenes. |
| `channel_decorr` | RGB channel correlation | Screens render color via spatially separate R/G/B subpixels → fringing that decorrelates channels at high frequency. Real edges move together across channels. |
| `saturation_mean` / `saturation_std` | Color gamut | Screens tend to run punchier, backlit color compared to ambient lighting. |
| `specular_ratio` | Glare / reflections | Screens produce blown-out, low-saturation highlights that real scenes rarely do. |
| `bezel_score` | Screen edge / frame | Detects a rectangular contour (bezel or printout border) sitting inside the shot. Very reliable when present, silent when tightly cropped. |
| `laplacian_var` | Sharpness | Recapture goes through two lenses/optics, shifting the sharpness profile. |
| `edge_density` | Edge structure | Complements sharpness as a texture signal. |

All features run on a 512px-resized image using only OpenCV + NumPy — no GPU, no model weights required to get started.

**Two ways to score an image:**
1. **Heuristic** (default, zero training data needed) — a hand-tuned weighted combination of the features above. Works immediately.
2. **Trained model** (`train.py`) — a small `RandomForestClassifier` fit on *your* real/screen photos. More accurate once you have ~50+50 examples; `predict.py` picks it up automatically if `model.joblib` exists.

---

## Quick start

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd spot_fake_photo
pip install -r requirements.txt

python predict.py path/to/image.jpg
# -> 0.12
```

That's it — works out of the box with the heuristic, no training required.

---

## Try it interactively

**Streamlit app (recommended):**
```bash
streamlit run app.py
```
Opens a local page where you can upload a photo or use your webcam (`st.camera_input`, works on mobile browsers too) and see the live score, verdict, latency, and every raw feature value.

**Browser-only demo (`demo.html`, no Python needed once served):**
```bash
python3 -m http.server 8000
# open http://localhost:8000/demo.html
```
Browsers block camera access on a plain `file://` page, so it must be served. For a phone, you'll need HTTPS (mobile browsers require a secure context) — e.g. `ngrok http 8000` for a quick tunnel, or deploy to GitHub Pages/Netlify. This page is a lightweight vanilla-JS port of the moiré/glare/bezel logic with its own in-browser FFT — a simplified approximation of `features.py`, not a 1:1 copy.

---

## Training on your own photos

1. Take ~50 normal photos of real things → `real/`
2. Take ~50 photos of a screen or printout showing a picture → `screen/`
   (vary lighting, angle, and screen type for a more robust model)
3. Train:
   ```bash
   python train.py
   ```
   Prints cross-validated accuracy and feature importances, saves `model.joblib`.
4. `predict.py` and `app.py` will automatically use `model.joblib` from then on.

---

## Project structure

```
spot_fake_photo/
├── features.py                     # feature extraction - the actual detection logic
├── train.py                        # trains RandomForest on real/ + screen/
├── predict.py                      # CLI: python predict.py image.jpg -> 0.93
├── app.py                          # local web UI (upload or webcam)
├── demo.html                       # browser-only vanilla-JS live demo (optional)
├── requirements.txt
├── .gitignore
├── real/.gitkeep                   # <- put your ~50 real photos here (not committed)
├── screen/.gitkeep                 # <- put your ~50 screen photos here (not committed)
└── sanity_check_synthetic_data/    # synthetic images proving the code runs end-to-end
                                     #  (NOT real accuracy data - see its own README)
```

---

## Results

`sanity_check_synthetic_data/` contains synthetic "real-like" and "screen-like" images (generated with NumPy/OpenCV, not real photos) used only to confirm the pipeline runs end-to-end:

- Heuristic: correctly separates real (~0.26) vs. screen (~0.75)
- Trained RandomForest: 100% cross-validated accuracy

**This is a code smoke-test, not a real accuracy number** — synthetic moiré is far cleaner than the real thing. Run `python train.py` on your own real photos to get the number that matters for the 95% bar.

---

## Latency & cost

Measured on a sandboxed container CPU (no GPU used anywhere), single 512px image:

| | Time |
|---|---|
| Warm (model already loaded, as in a real service/app) | **~55–60 ms/image** |
| Cold process (fresh `python3` start incl. imports + model load) | **~2 s**, one-time per process, not per image |

**Cost per image:**
- **On-device (phone):** ~$0 marginal cost — no network call, no server. The feature set (basic OpenCV ops, no neural net) is light enough to port to a mobile CV library.
- **Cloud server (if centralized):** at ~60 ms CPU-only per image, one modest CPU core handles ~1,000 images/min. On a ~$0.05/hr small cloud instance that's roughly **$0.05–$0.10 per million images** for compute alone (assumes steady load, ignores network/storage/orchestration overhead which usually dominates real cloud bills more than compute).

---

## Roadmap / what's next

1. **Calibrate on real photos** — thresholds are only as good as real-world data; LCD vs. OLED vs. printed halftone all look different in frequency space.
2. **Multi-scale moiré analysis** — moiré period depends on screen distance; analyzing multiple crop scales would catch both close-up and far shots.
3. **Adversarial robustness** — tight crops could hide the bezel/blur out the grid; want real adversarial test examples.
4. **Native on-device port** — reimplement `features.py` in Android CameraX + OpenCV or iOS Core Image for real production latency/battery numbers.
5. **Active-light trick** — firing flash and checking the specular response could add a cheap, hard-to-fake extra signal.

---

## Pushing to GitHub (without your dataset or venv)

A `.gitignore` is included and excludes `venv/`/`.venv/`, `__pycache__/`, your `real/`/`screen/` photos (keeps `.gitkeep` so the folder structure survives), `sanity_check_synthetic_data/`, and `model.joblib` (regenerate locally instead of committing a binary).

```bash
cd spot_fake_photo
git init
git add -A
git status --short   # sanity check: should NOT list *.jpg, venv/, model.joblib
git commit -m "Real vs screen-recapture detector"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

If your venv has a non-standard name, add it to `.gitignore` before staging:
```bash
echo "your-venv-folder/" >> .gitignore
```

Want to ship the trained model too (it's tiny)? Force past the ignore rule:
```bash
git add -f model.joblib
```

## License

Add a license of your choice (MIT is a common default for small tools like this) before making the repo public.
