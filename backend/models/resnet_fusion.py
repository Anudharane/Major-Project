"""ResNet-18 + handcrafted feature-fusion network.

    image  -> ResNet-18 (ImageNet weights) -> 512 deep features --+
                                                                    +-> concat (516) -> FC -> 3 logits
    4 scaled handcrafted features ----------------------------------+
"""
from pathlib import Path
from typing import Iterable, Tuple

import torch
import torch.nn as nn
from torchvision import models

import config


class ResNet18FeatureFusion(nn.Module):
    def __init__(self, num_classes: int = len(config.CLASS_NAMES), pretrained: bool = True,
                 dropout: float = config.DROPOUT, hidden: int = config.HIDDEN_UNITS):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        assert backbone.fc.in_features == config.NUM_DEEP_FEATURES
        backbone.fc = nn.Identity()          # original 1000-class ImageNet layer removed -> 512-d output
        self.backbone = backbone
        self.classifier = nn.Sequential(      # receives 512 + 4 = 516 features
            nn.Dropout(dropout),
            nn.Linear(config.NUM_TOTAL_FEATURES, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    # ---- forward passes
    def deep_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)                                   # (B, 512)

    def forward_with_features(self, x: torch.Tensor, hand: torch.Tensor
                              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        deep = self.deep_features(x)
        fused = torch.cat([deep, hand], dim=1)                    # (B, 516)
        return self.classifier(fused), deep, fused

    def forward(self, x: torch.Tensor, hand: torch.Tensor) -> torch.Tensor:
        return self.forward_with_features(x, hand)[0]              # logits; apply softmax outside

    # ---- transfer-learning helpers
    def set_backbone_trainable(self, layers: Iterable[str] = ()):
        """Freeze the whole backbone, then unfreeze the named children (e.g. 'layer4')."""
        layers = set(layers)
        for name, child in self.backbone.named_children():
            for p in child.parameters():
                p.requires_grad = name in layers

    def train(self, mode: bool = True):
        """Keep BatchNorm of frozen backbone blocks in eval mode so ImageNet statistics are preserved."""
        super().train(mode)
        if mode:
            for _, child in self.backbone.named_children():
                if not any(p.requires_grad for p in child.parameters()):
                    child.eval()
        return self


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(path: Path = config.CHECKPOINT_PATH, device=None) -> Tuple[ResNet18FeatureFusion, dict]:
    device = device or get_device()
    ckpt = torch.load(path, map_location=device)
    model = ResNet18FeatureFusion(pretrained=False)   # weights come from the checkpoint
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), ckpt
