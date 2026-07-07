"""
features.py
-----------
Hand-crafted features that separate REAL photos from SCREEN RECAPTURES
(a photo of a phone/laptop/monitor/printout showing a picture).

Why these features work (the physics of "photographing a photo"):

1. MOIRE / PIXEL-GRID PERIODICITY
   Screens (and halftone-printed photos) are made of a regular grid of
   pixels/dots. When you photograph that grid with a camera sensor that
   has its OWN regular grid, the two grids beat against each other and
   create moire interference patterns - visible as sharp, non-smooth
   spikes in the 2D frequency spectrum. A real-world scene almost never
   has this kind of regular high-frequency periodicity.
   -> fft_peak_score, fft_high_freq_ratio

2. RGB SUBPIXEL DECORRELATION
   Screens render color using separate R/G/B subpixels laid out in a
   fixed spatial pattern. Photographing that pattern introduces color
   fringing that is only weakly correlated across channels at high
   frequency. Real photos have highly-correlated edges across R/G/B
   (an edge is an edge in all three channels together).
   -> channel_decorrelation

3. SCREEN COLOR GAMUT / GLARE / WHITE POINT
   Screens (esp. LCD/OLED) tend to have punchier saturation, blown
   highlights and glare/reflections, and specific white-balance
   signatures compared to ambient real-world lighting.
   -> specular_ratio, saturation_mean, saturation_std

4. RECTANGULAR BEZEL / FRAME
   People rarely crop perfectly - a screen or printout often shows a
   physical edge/bezel/border as a straight rectangular contour inside
   the frame.
   -> bezel_score

5. SHARPNESS PROFILE
   Recaptures go through two optical systems (screen + camera lens),
   often adding slight blur or, conversely, an unnaturally crisp pixel
   grid up close. Either way the sharpness/edge statistics differ from
   a single real capture.
   -> laplacian_var, edge_density

All features are cheap (a handful of OpenCV/numpy ops), scale-invariant
(computed on a fixed resize), and fast enough for a phone.
"""

import cv2
import numpy as np


TARGET_SIZE = 512  # resize longest edge to this before analysis (speed + consistency)


def _load_and_resize(path_or_array):
    if isinstance(path_or_array, str):
        img = cv2.imread(path_or_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image: {path_or_array}")
    else:
        img = path_or_array

    h, w = img.shape[:2]
    scale = TARGET_SIZE / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def _fft_features(gray):
    """Detect moire/pixel-grid periodicity via the 2D frequency spectrum."""
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    mag_log = np.log1p(mag)

    h, w = mag_log.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = r.max()

    # Radial average profile (expected smooth falloff for natural images)
    r_int = r.astype(np.int32)
    radial_sum = np.bincount(r_int.ravel(), weights=mag_log.ravel())
    radial_count = np.bincount(r_int.ravel())
    radial_mean = radial_sum / np.maximum(radial_count, 1)

    # Deviation of each pixel from the expected radial average -> "peakiness"
    expected = radial_mean[r_int]
    deviation = mag_log - expected

    # Only look at a mid/high frequency band (ignore DC and very low freq
    # which is dominated by overall brightness/contrast, not texture)
    band_mask = (r > 0.08 * r_max) & (r < 0.85 * r_max)
    if band_mask.sum() == 0:
        peak_score = 0.0
    else:
        # Top 0.1% most "spiky" deviations in the band -> moire peak strength
        band_dev = deviation[band_mask]
        k = max(1, int(0.001 * band_dev.size))
        peak_score = float(np.mean(np.sort(band_dev)[-k:]))

    # High-frequency energy ratio (outer ring vs total) - screens & halftone
    # prints tend to carry more structured high-frequency energy
    outer_mask = r > 0.5 * r_max
    high_energy = mag[outer_mask].sum()
    total_energy = mag.sum() + 1e-6
    high_freq_ratio = float(high_energy / total_energy)

    return peak_score, high_freq_ratio


def _channel_decorrelation(img_bgr):
    """Correlation of high-frequency edge content across R/G/B channels.
    Real edges move together across channels; RGB-subpixel fringing from
    a screen does not."""
    b, g, r = [c.astype(np.float32) for c in cv2.split(img_bgr)]

    def hp(c):
        blur = cv2.GaussianBlur(c, (0, 0), sigmaX=2)
        return c - blur

    hb, hg, hr = hp(b), hp(g), hp(r)

    def corr(a, c):
        a = a.ravel()
        c = c.ravel()
        a = a - a.mean()
        c = c - c.mean()
        denom = (np.linalg.norm(a) * np.linalg.norm(c)) + 1e-6
        return float(np.dot(a, c) / denom)

    c_bg = corr(hb, hg)
    c_gr = corr(hg, hr)
    c_br = corr(hb, hr)
    mean_corr = (c_bg + c_gr + c_br) / 3.0
    # Lower correlation -> more likely a screen recapture
    return mean_corr


def _color_features(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)

    saturation_mean = float(s.mean())
    saturation_std = float(s.std())

    # Specular / glare: very bright, low-saturation, small clustered blobs
    bright_mask = (v > 245) & (s < 40)
    specular_ratio = float(bright_mask.mean())

    return saturation_mean, saturation_std, specular_ratio


def _sharpness_features(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_var = float(lap.var())

    edges = cv2.Canny(gray, 80, 160)
    edge_density = float((edges > 0).mean())

    return laplacian_var, edge_density


def _bezel_score(gray):
    """Look for a strong rectangular contour sitting inside the frame -
    a hint of a screen/printout edge or bezel."""
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    h, w = gray.shape
    img_area = h * w
    best = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.05 * img_area:
            continue
        rect = cv2.minAreaRect(c)
        (rw, rh) = rect[1]
        rect_area = rw * rh
        if rect_area <= 0:
            continue
        fill_ratio = area / rect_area  # close to 1 -> genuinely rectangular
        size_ratio = area / img_area   # how much of the frame it covers
        # reward contours that are rectangular AND cover a meaningful,
        # but not the entire, chunk of the frame (a bezel inset from edges)
        score = fill_ratio * min(size_ratio, 1.0)
        best = max(best, score)
    return float(best)


def extract_features(path_or_array):
    """Returns (feature_dict, feature_vector[list of floats in fixed order])."""
    img = _load_and_resize(path_or_array)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    fft_peak_score, fft_high_freq_ratio = _fft_features(gray)
    channel_decorr = _channel_decorrelation(img)
    saturation_mean, saturation_std, specular_ratio = _color_features(img)
    laplacian_var, edge_density = _sharpness_features(gray)
    bezel = _bezel_score(gray)

    feats = {
        "fft_peak_score": fft_peak_score,
        "fft_high_freq_ratio": fft_high_freq_ratio,
        "channel_decorr": channel_decorr,
        "saturation_mean": saturation_mean,
        "saturation_std": saturation_std,
        "specular_ratio": specular_ratio,
        "laplacian_var": laplacian_var,
        "edge_density": edge_density,
        "bezel_score": bezel,
    }
    vector = [feats[k] for k in FEATURE_ORDER]
    return feats, vector


# Fixed order used everywhere (train.py, predict.py) so vectors line up
FEATURE_ORDER = [
    "fft_peak_score",
    "fft_high_freq_ratio",
    "channel_decorr",
    "saturation_mean",
    "saturation_std",
    "specular_ratio",
    "laplacian_var",
    "edge_density",
    "bezel_score",
]
