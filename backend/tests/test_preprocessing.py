import numpy as np
import config
from preprocessing import image_preprocessing as ip
from tests.helpers import encode, synthetic_eye


def test_image_loading_from_bytes():
    rgb = ip.decode_image_bytes(encode(synthetic_eye(300)))
    assert rgb is not None and rgb.shape == (300, 300, 3) and rgb.dtype == np.uint8


def test_corrupt_bytes_return_none():
    assert ip.decode_image_bytes(b"not an image") is None
    assert ip.decode_image_bytes(b"") is None


def test_resizing_to_224():
    for shape in [(100, 100, 3), (500, 300, 3), (1000, 1000, 3)]:
        out = ip.resize_square(np.zeros(shape, np.uint8))
        assert out.shape == (224, 224, 3)


def test_pipeline_outputs_224_rgb():
    r = ip.preprocess(synthetic_eye(400), mode="auto")
    assert r.roi_raw.shape == r.enhanced.shape == (224, 224, 3)


def test_imagenet_normalization():
    x = ip.normalize_imagenet(np.full((224, 224, 3), 255, np.uint8))
    expected = (1.0 - np.array(config.IMAGENET_MEAN)) / np.array(config.IMAGENET_STD)
    assert np.allclose(x[0, 0], expected, atol=1e-5) and x.dtype == np.float32


def test_clahe_keeps_shape_and_increases_contrast_on_flat_image():
    flat = np.clip(np.random.default_rng(0).normal(120, 4, (224, 224, 3)), 0, 255).astype(np.uint8)
    out = ip.apply_clahe(flat)
    assert out.shape == flat.shape and out.std() > flat.std()


def test_illumination_normalization_reduces_gradient():
    ramp = np.tile(np.linspace(40, 200, 224, dtype=np.uint8)[None, :, None], (224, 1, 3))
    out = ip.normalize_illumination(ramp, blend=1.0)
    assert out.mean(axis=(0, 2)).std() < ramp.mean(axis=(0, 2)).std()
