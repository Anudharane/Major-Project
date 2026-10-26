"""Needs PyTorch. Verifies feature dimensions and the full API wiring with a randomly
initialised (UNTRAINED) model: this tests plumbing, not accuracy."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")
from sklearn.preprocessing import StandardScaler

import config
from models.resnet_fusion import ResNet18FeatureFusion
from tests.helpers import encode, synthetic_eye


def make_model():
    torch.manual_seed(0)
    return ResNet18FeatureFusion(pretrained=False).eval()


def test_feature_dimensions_512_4_516():
    m = make_model()
    logits, deep, fused = m.forward_with_features(torch.randn(2, 3, 224, 224), torch.randn(2, 4))
    assert deep.shape == (2, 512) and fused.shape == (2, 516) and logits.shape == (2, 3)
    assert torch.allclose(torch.softmax(logits, 1).sum(1), torch.ones(2), atol=1e-5)


def test_frozen_backbone_has_no_trainable_params_and_unfreeze_layer4():
    m = make_model()
    m.set_backbone_trainable([])
    assert not any(p.requires_grad for p in m.backbone.parameters())
    m.set_backbone_trainable(["layer4"])
    assert all(p.requires_grad for p in m.backbone.layer4.parameters())
    assert not any(p.requires_grad for p in m.backbone.layer1.parameters())


def _client():
    from fastapi.testclient import TestClient
    from main import app
    from prediction.predict import ScreeningPredictor
    scaler = StandardScaler().fit(np.random.default_rng(0).normal(size=(50, 4)) * [200, .1, .5, 5] + [200, .1, .6, 90])
    client = TestClient(app)
    client.__enter__()                                   # runs lifespan (model files absent -> predictor None)
    app.state.predictor = ScreeningPredictor(make_model(), scaler, ood=None, device=torch.device("cpu"))
    return client


def test_predict_endpoint_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HISTORY_DB_PATH", tmp_path / "h.db")
    c = _client()
    assert c.get("/health").json()["status"] == "ok"
    r = c.post("/predict?include_vector=true", files={"file": ("eye.jpg", encode(synthetic_eye(300)), "image/jpeg")})
    assert r.status_code == 200, r.text
    j = r.json()
    assert set(j["probabilities"]) == set(config.DISPLAY_NAMES)
    assert abs(sum(j["probabilities"].values()) - 100) < 0.5
    assert set(j["features"]) == set(config.HANDCRAFTED_NAMES)
    assert len(j["feature_vector"]["deep"]) == 512 and j["feature_counts"]["total"] == 516
    assert j["prediction"] in config.DISPLAY_NAMES + ["Uncertain", "Spectacles Required", "Spectacles Not Required"]
    assert len(c.get("/history").json()) == 1


def test_predict_rejects_bad_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HISTORY_DB_PATH", tmp_path / "h.db")
    c = _client()
    for name, data in [("x.txt", b"hello"), ("x.jpg", b"garbage"), ("x.jpg", encode(synthetic_eye(blur=8))),
                       ("x.jpg", encode(synthetic_eye(brightness=0.1)))]:
        r = c.post("/predict", files={"file": (name, data)})
        assert r.status_code in (415, 422) and (r.json().get("status") == "rejected" or r.json().get("detail", {}).get("status") == "rejected"), (name, r.text)
    assert c.get("/history").json() == []                # rejected images are not stored
