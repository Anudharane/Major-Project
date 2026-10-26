import numpy as np
import pytest
import config
from preprocessing.image_preprocessing import preprocess
from tests.helpers import encode, synthetic_eye
from utils import validation as v


def rejected(fn, *a):
    try:
        fn(*a)
    except v.ImageRejected as e:
        return e.reasons
    return None


def test_unsupported_extension():
    assert rejected(v.validate_upload, "eye.gif", encode(synthetic_eye()))
    assert rejected(v.validate_upload, "notes.txt", b"hello")


def test_invalid_file_content():
    assert rejected(v.validate_upload, "eye.jpg", b"this is not an image")
    assert rejected(v.validate_upload, "eye.jpg", b"")


def test_oversize_file():
    assert rejected(v.validate_upload, "eye.jpg", b"0" * (config.MAX_UPLOAD_MB * 1024 * 1024 + 1))


def test_too_small_image():
    assert rejected(v.validate_upload, "eye.png", encode(synthetic_eye(64), ".png"))


def test_valid_upload_accepted():
    assert v.validate_upload("eye.jpg", encode(synthetic_eye(300))).shape[:2] == (300, 300)
    assert v.validate_upload("EYE.PNG", encode(synthetic_eye(300), ".png")) is not None


def test_quality_good_image_passes():
    q = v.assess_quality(preprocess(synthetic_eye(), "dataset").roi_raw)
    assert v.check_quality(q) == []


def test_quality_blurry_dark_bright_flat_fail():
    f = lambda img: v.check_quality(v.assess_quality(preprocess(img, "dataset").roi_raw))
    assert any("blurry" in m for m in f(synthetic_eye(blur=8)))
    assert any("dark" in m for m in f(synthetic_eye(brightness=0.1)))
    assert any("overexposed" in m for m in f(np.full((224, 224, 3), 250, np.uint8)))
    assert f(np.full((224, 224, 3), 120, np.uint8))                      # flat gray: blurry + low contrast
