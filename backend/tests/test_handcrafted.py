import cv2
import numpy as np
import config
from features import handcrafted_features as hf
from tests.helpers import synthetic_eye


def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def test_sharpness_blurred_is_lower():
    sharp, blurred = synthetic_eye(), synthetic_eye(blur=3)
    assert hf.focal_sharpness(gray(sharp)) > 5 * hf.focal_sharpness(gray(blurred))


def test_asymmetry_range_and_direction():
    stripes = np.tile((np.arange(224) % 8 < 4).astype(np.uint8)[None, :] * 255, (224, 1))  # only horizontal gradients
    flat = np.full((224, 224), 128, np.uint8)
    assert hf.meridional_asymmetry(stripes) > 0.9
    assert hf.meridional_asymmetry(flat) == 0.0          # numerically stable on zero gradients
    assert 0.0 <= hf.meridional_asymmetry(gray(synthetic_eye())) <= 1.0


def test_pupil_detected_on_synthetic_eye():
    cx, cy, r, ok = hf.detect_pupil(gray(synthetic_eye(pupil=18)))
    assert ok and abs(cx - 112) < 6 and abs(cy - 112) < 6 and 10 < r < 26


def test_pupil_detection_failure_is_handled():
    flat = np.full((224, 224), 128, np.uint8)
    assert hf.detect_pupil(flat)[3] is False
    assert 0.0 <= hf.pupil_contrast_ratio(flat) <= 1.0     # falls back, no exception / NaN


def test_pupil_contrast_dark_pupil_high():
    v = hf.pupil_contrast_ratio(gray(synthetic_eye()))
    assert 0.3 < v <= 1.0


def test_reflex_symmetry_symmetric_vs_not():
    img = gray(synthetic_eye(noise=0))
    sym = hf.reflex_symmetry(img)
    skewed = img.copy().astype(np.float32)
    skewed[:, :112] *= 0.4
    assert sym > 90 and hf.reflex_symmetry(skewed.astype(np.uint8)) < sym


def test_exactly_four_finite_features():
    feats = hf.extract_handcrafted(synthetic_eye())
    assert list(feats) == config.HANDCRAFTED_NAMES and len(feats) == 4
    assert all(np.isfinite(v) for v in feats.values())
    assert hf.handcrafted_vector(synthetic_eye()).shape == (4,)
    assert config.NUM_DEEP_FEATURES + config.NUM_HANDCRAFTED == config.NUM_TOTAL_FEATURES == 516
    assert len(config.RESNET_FEATURE_NAMES) == 512


def test_black_and_white_images_do_not_produce_nan():
    for v in (0, 255):
        feats = hf.extract_handcrafted(np.full((224, 224, 3), v, np.uint8))
        assert all(np.isfinite(x) for x in feats.values())
