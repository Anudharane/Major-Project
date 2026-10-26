import os
import numpy as np
import torch
from tqdm import tqdm

torch.set_num_threads(min(12, os.cpu_count() or 8))

import config
from models.resnet_fusion import ResNet18FeatureFusion


@torch.no_grad()
def extract_deep_features(model: ResNet18FeatureFusion, loader, device, desc: str = "") -> np.ndarray:
    """Run the backbone over a DataLoader that yields (image, hand, label, index).

    Returns an (N, 512) array in dataset order (loader must not shuffle).
    """
    model.eval()
    out = []
    for images, _hand, _labels, _idx in tqdm(loader, desc=desc, ncols=80):
        feats = model.deep_features(images.to(device)).cpu().numpy()
        assert feats.shape[1] == config.NUM_DEEP_FEATURES, f"expected 512 features, got {feats.shape[1]}"
        out.append(feats)
    return np.concatenate(out, axis=0).astype(np.float32)
