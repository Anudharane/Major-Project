"""Four handcrafted ocular image features.

IMPORTANT: these are image-based PROXY measurements computed from an ordinary
RGB photograph. They are NOT clinical photorefraction or autorefraction values.

Inputs are 224x224 RGB uint8 eye ROIs (PreprocessResult.roi_raw).
Every function is NaN-safe and returns a finite float.
"""
from typing import Dict, Tuple

import cv2
import numpy as np

import config

EPS = 1e-8
Pupil = Tuple[float, float, float, bool]  # cx, cy, radius, detected


def _gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb


def _finite(x: float, default: float = 0.0) -> float:
    x = float(x)
    return x if np.isfinite(x) else default


# -------------------------------------------------------- 1. sharpness
def focal_sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian: higher = sharper focus."""
    return _finite(cv2.Laplacian(gray, cv2.CV_64F).var())


# ---------------------------------------------------- 2. asymmetry
def meridional_asymmetry(gray: np.ndarray) -> float:
    """Normalised difference between horizontal- and vertical-meridian edge energy.

    0 = equal gradient energy along both meridians, 1 = all energy in one direction.
    """
    g = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0)
    eh = float(np.mean(np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))))  # horizontal gradients
    ev = float(np.mean(np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))))  # vertical gradients
    return _finite(abs(eh - ev) / (eh + ev + EPS))


# ------------------------------------------------------ pupil detection
def detect_pupil(gray: np.ndarray) -> Pupil:
    """Find the pupil as the darkest compact blob near the image centre.

    Returns (cx, cy, r, True) on success; on failure returns a central default
    disc with detected=False so downstream features still work.
    """
    h, w = gray.shape
    fallback: Pupil = (w / 2.0, h / 2.0, 0.07 * min(h, w), False)
    try:
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        y0, y1, x0, x1 = int(.2 * h), int(.8 * h), int(.2 * w), int(.8 * w)
        thr = np.percentile(blur[y0:y1, x0:x1], 8)          # darkest 8% of the central window
        mask = np.zeros_like(gray, np.uint8)
        mask[y0:y1, x0:x1] = (blur[y0:y1, x0:x1] <= thr).astype(np.uint8)
        k = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
        n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
        best, best_score = None, 0.0
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 0.002 * h * w or area > 0.15 * h * w:
                continue
            bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            aspect = min(bw, bh) / max(bw, bh)
            fill = area / float(bw * bh)
            dist = np.hypot(cents[i][0] - w / 2, cents[i][1] - h / 2) / (0.5 * min(h, w))
            if aspect < 0.5:
                continue
            score = aspect * fill * max(0.0, 1.0 - dist)
            if score > best_score:
                best, best_score = i, score
        if best is None:
            return fallback
        r = float(np.sqrt(stats[best, cv2.CC_STAT_AREA] / np.pi))
        return (float(cents[best][0]), float(cents[best][1]), r, True)
    except Exception:
        return fallback


def _disc_mask(shape, cx, cy, r_in, r_out) -> np.ndarray:
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    d = np.hypot(xx - cx, yy - cy)
    return (d >= r_in) & (d <= r_out)


# ----------------------------------------------- 3. pupil contrast
def pupil_contrast_ratio(gray: np.ndarray, pupil: Pupil = None) -> float:
    """(iris - pupil) / (iris + pupil) brightness, in [0, 1]. Robust to specular highlights."""
    cx, cy, r, _ = pupil if pupil is not None else detect_pupil(gray)
    pupil_px = gray[_disc_mask(gray.shape, cx, cy, 0, 0.8 * r)]
    iris_mask = _disc_mask(gray.shape, cx, cy, 1.3 * r, 2.6 * r) & (gray < 240)  # drop glare
    iris_px = gray[iris_mask]
    if pupil_px.size < 5 or iris_px.size < 5:
        return 0.0
    p, i = float(np.median(pupil_px)), float(np.median(iris_px))
    return _finite(np.clip((i - p) / (i + p + EPS), 0.0, 1.0))


# --------------------------------------------------- 4. reflex symmetry
def reflex_symmetry(gray: np.ndarray, pupil: Pupil = None) -> float:
    """Brightness symmetry (0-100 %) of the iris/pupil disc: left vs right and top vs bottom.

    Image-based PROXY only; this is not a clinical photorefraction reflex measurement.
    """
    h, w = gray.shape
    cx, cy, r, _ = pupil if pupil is not None else detect_pupil(gray)
    R = min(2.5 * r, 0.4 * min(h, w))
    disc = _disc_mask(gray.shape, cx, cy, 0, R)
    yy, xx = np.mgrid[:h, :w]
    g = gray.astype(np.float64)

    def mean_of(region):
        m = disc & region
        return float(g[m].mean()) if m.sum() > 5 else None

    left, right = mean_of(xx < cx), mean_of(xx >= cx)
    top, bottom = mean_of(yy < cy), mean_of(yy >= cy)
    if None in (left, right, top, bottom):
        return 0.0
    asym = 0.5 * (abs(left - right) / (left + right + EPS) + abs(top - bottom) / (top + bottom + EPS))
    return _finite(np.clip(100.0 * (1.0 - asym), 0.0, 100.0))


# ------------------------------------------------------------ public API
def extract_handcrafted(rgb: np.ndarray) -> Dict[str, float]:
    """Return the four features (ordered as config.HANDCRAFTED_NAMES) for one ROI image."""
    gray = _gray(rgb)
    pupil = detect_pupil(gray)
    return {
        "focal_sharpness": focal_sharpness(gray),
        "meridional_asymmetry": meridional_asymmetry(gray),
        "pupil_contrast_ratio": pupil_contrast_ratio(gray, pupil),
        "reflex_symmetry": reflex_symmetry(gray, pupil),
    }


def handcrafted_vector(rgb: np.ndarray) -> np.ndarray:
    feats = extract_handcrafted(rgb)
    return np.array([feats[n] for n in config.HANDCRAFTED_NAMES], dtype=np.float32)


def pupil_was_detected(rgb: np.ndarray) -> bool:
    return detect_pupil(_gray(rgb))[3]
