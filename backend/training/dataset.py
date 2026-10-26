"""PyTorch dataset over the existing dataset/{train,validation,test} folders."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import config
from preprocessing.image_preprocessing import preprocess, read_image_rgb

Sample = Tuple[str, int]  # (path relative to dataset dir, label)


def list_samples(split: str, root: Path = config.DATASET_DIR) -> List[Sample]:
    """All images of one split, sorted for reproducibility. The existing split is never altered."""
    samples: List[Sample] = []
    for label, cls in enumerate(config.CLASS_NAMES):
        folder = root / split / cls
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing folder: {folder}. Run training/prepare_dataset.py first.")
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in config.IMAGE_EXTENSIONS:
                samples.append((p.relative_to(root).as_posix(), label))
    return samples


def build_transforms(train: bool):
    """Training: mild geometric/photometric augmentation (hue untouched so iris colour is preserved).
    Eval: deterministic."""
    norm = [transforms.ToTensor(), transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)]
    if not train:
        return transforms.Compose(norm)
    return transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.0),
        transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
        *norm,
    ])


class EyeDataset(Dataset):
    """Returns (image_tensor, scaled_handcrafted_tensor, label, index).

    `hand_raw` maps relative image path -> raw 4-vector (precomputed once, see
    feature_extraction.get_handcrafted_table). `scaler` is fitted on TRAIN only.
    Handcrafted features are measured on the deterministic ROI, not the augmented image.
    """

    def __init__(self, samples: List[Sample], hand_raw: Dict[str, np.ndarray], scaler,
                 augment: bool = False, root: Path = config.DATASET_DIR):
        self.samples, self.root = samples, root
        self.transform = build_transforms(augment)
        raw = np.stack([hand_raw[p] for p, _ in samples]).astype(np.float32)
        self.hand = torch.from_numpy(scaler.transform(raw).astype(np.float32))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i: int):
        rel, label = self.samples[i]
        rgb = read_image_rgb(self.root / rel)
        if rgb is None:
            raise IOError(f"Unreadable image: {rel}")
        enhanced = preprocess(rgb, mode="dataset").enhanced
        return self.transform(Image.fromarray(enhanced)), self.hand[i], label, i
