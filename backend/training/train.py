"""Train the ResNet-18 + handcrafted feature-fusion classifier.

Usage (from backend/):
    python training/train.py
    python training/train.py --drop-near-dups     # remove TRAIN images that nearly duplicate val/test images

Data handling guarantees:
  * the existing train/validation/test split is used as-is,
  * the StandardScaler for handcrafted features is fitted on TRAIN only,
  * validation drives early stopping / checkpointing / LR schedule,
  * the TEST set is evaluated exactly once, after training is finished.
"""
import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from features.feature_extraction import get_handcrafted_table, hand_dict
from models.resnet_fusion import ResNet18FeatureFusion, get_device
from training.dataset import EyeDataset, list_samples
from training.evaluate import confidence_threshold_analysis, run_evaluation


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    """Top-level (picklable on Windows) DataLoader worker seeding."""
    np.random.seed(config.SEED + worker_id)


def run_epoch(model, loader, device, criterion, optimizer=None, desc=""):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for images, hand, y, _ in tqdm(loader, desc=desc, ncols=80, leave=False):
            images, hand, y = images.to(device), hand.to(device), y.to(device)
            logits = model(images, hand)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)
    return total_loss / n, correct / n


def drop_near_duplicates(train):
    """Remove from TRAIN every image that is a near-copy of a validation/test image (see prepare_dataset.py)."""
    path = config.RESULTS_DIR / "leakage_flags.csv"
    if not path.exists():
        sys.exit("results/leakage_flags.csv missing. Run `python training/prepare_dataset.py` first.")
    f = pd.read_csv(path)
    bad = set(f[(f.split == "train") & f.near_duplicate].image_path)
    kept = [s for s in train if s[0] not in bad]
    print(f"--drop-near-dups: removed {len(train) - len(kept)} train images that nearly duplicate val/test images "
          f"({len(kept)} remain).")
    return kept


def plot_history(hist, out_dir: Path):
    ep = range(1, len(hist) + 1)
    for key, title, fname in (("acc", "Accuracy", "training_accuracy.png"), ("loss", "Loss", "training_loss.png")):
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.plot(ep, [h[f"train_{key}"] for h in hist], marker="o", label="Train")
        ax.plot(ep, [h[f"val_{key}"] for h in hist], marker="o", label="Validation")
        n_head = sum(1 for h in hist if h["phase"] == "head")
        if 0 < n_head < len(hist):
            ax.axvline(n_head + 0.5, color="gray", ls="--", lw=1, label="Unfreeze layer4")
        ax.set_xlabel("Epoch"); ax.set_ylabel(title); ax.set_title(f"Training vs Validation {title}")
        ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop-near-dups", action="store_true")
    ap.add_argument("--head-epochs", type=int, default=config.HEAD_EPOCHS)
    ap.add_argument("--finetune-epochs", type=int, default=config.FINETUNE_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--no-eval", action="store_true", help="skip the final test-set evaluation")
    args = ap.parse_args()

    set_seed(config.SEED)
    device = get_device()
    print(f"Device: {device}")
    config.MODELS_DIR.mkdir(exist_ok=True); config.RESULTS_DIR.mkdir(exist_ok=True)

    # 1-2. load + verify dataset (existing split preserved)
    train_s, val_s, test_s = (list_samples(s) for s in config.SPLITS)
    if args.drop_near_dups:
        train_s = drop_near_duplicates(train_s)
    for name, s in zip(config.SPLITS, (train_s, val_s, test_s)):
        counts = np.bincount([l for _, l in s], minlength=3)
        print(f"{name:11s}: {len(s):5d} images  " + "  ".join(f"{c}={k}" for c, k in zip(config.CLASS_NAMES, counts)))
    assert min(len(train_s), len(val_s), len(test_s)) > 0

    # 7-8. handcrafted features (cached) + scaler fitted on TRAIN ONLY
    table = get_handcrafted_table()
    hand = hand_dict(table)
    scaler = StandardScaler().fit(np.stack([hand[p] for p, _ in train_s]))
    joblib.dump(scaler, config.SCALER_PATH)
    print("Handcrafted scaler fitted on train only:",
          dict(zip(config.HANDCRAFTED_NAMES, scaler.mean_.round(3))))

    # 3. data loaders
    def make(samples, augment, shuffle):
        return DataLoader(EyeDataset(samples, hand, scaler, augment=augment), batch_size=args.batch_size,
                          shuffle=shuffle, num_workers=config.NUM_WORKERS,
                          pin_memory=device.type == "cuda", persistent_workers=config.NUM_WORKERS > 0,
                          worker_init_fn=seed_worker)
    train_loader, val_loader, test_loader = make(train_s, True, True), make(val_s, False, False), make(test_s, False, False)

    # 4-6, 9. model: pretrained ResNet-18 (fc removed -> 512) fused with 4 handcrafted features -> 3 classes
    model = ResNet18FeatureFusion(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    hist, best = [], {"val_loss": float("inf"), "state": None, "epoch": 0}
    t0 = time.time()

    def fit_phase(phase, epochs, optimizer, scheduler, patience):
        stale = 0
        for _ in range(epochs):
            ep = len(hist) + 1
            tl, ta = run_epoch(model, train_loader, device, criterion, optimizer, desc=f"epoch {ep} train")
            vl, va = run_epoch(model, val_loader, device, criterion, None, desc=f"epoch {ep} val")
            if scheduler is not None:
                scheduler.step(vl)
            hist.append({"epoch": ep, "phase": phase, "train_loss": tl, "train_acc": ta, "val_loss": vl, "val_acc": va,
                         "lr": optimizer.param_groups[0]["lr"]})
            flag = ""
            if vl < best["val_loss"] - 1e-4:               # 11-12. checkpoint on best validation loss
                best.update(val_loss=vl, val_acc=va, epoch=ep, state=copy.deepcopy(model.state_dict()))
                torch.save({"model_state": best["state"], "class_names": config.CLASS_NAMES,
                            "epoch": ep, "val_loss": vl, "val_acc": va}, config.CHECKPOINT_PATH)
                stale, flag = 0, "  <- best, checkpoint saved"
            else:
                stale += 1
            print(f"[{phase:8s}] epoch {ep:2d}  train loss {tl:.4f} acc {ta * 100:5.2f}%  |  "
                  f"val loss {vl:.4f} acc {va * 100:5.2f}%{flag}")
            if patience and stale >= patience:
                print(f"Early stopping: no validation-loss improvement for {patience} epochs.")
                break

    # Phase 1: backbone frozen, train the fusion head only
    print("\n=== Phase 1: frozen ResNet-18 backbone, training the fusion classifier ===")
    model.set_backbone_trainable([])
    opt = torch.optim.AdamW(model.classifier.parameters(), lr=config.LR_HEAD, weight_decay=config.WEIGHT_DECAY)
    fit_phase("head", args.head_epochs, opt, None, patience=0)

    # Phase 2: fine-tune last ResNet block(s) with a lower learning rate
    print(f"\n=== Phase 2: fine-tuning {config.UNFREEZE_LAYERS} ===")
    model.set_backbone_trainable(config.UNFREEZE_LAYERS)
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": backbone_params, "lr": config.LR_BACKBONE},
                             {"params": model.classifier.parameters(), "lr": config.LR_HEAD * 0.5}],
                            weight_decay=config.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=config.SCHEDULER_FACTOR, patience=config.SCHEDULER_PATIENCE)
    fit_phase("finetune", args.finetune_epochs, opt, sched, patience=config.EARLY_STOP_PATIENCE)

    # 14. save history, plots, configs
    minutes = (time.time() - t0) / 60
    plot_history(hist, config.RESULTS_DIR)
    (config.RESULTS_DIR / "training_history.json").write_text(json.dumps(hist, indent=2))
    config.CLASS_NAMES_PATH.write_text(json.dumps({str(i): n for i, n in enumerate(config.CLASS_NAMES)}, indent=2))
    cfg = {k: getattr(config, k) for k in ("SEED", "IMAGE_SIZE", "BATCH_SIZE", "HEAD_EPOCHS", "FINETUNE_EPOCHS", "LR_HEAD",
                                           "LR_BACKBONE", "WEIGHT_DECAY", "EARLY_STOP_PATIENCE", "DROPOUT", "HIDDEN_UNITS",
                                           "UNFREEZE_LAYERS", "CLAHE_CLIP_LIMIT", "ILLUMINATION_BLEND")}
    cfg.update(model="ResNet-18 (ImageNet pretrained, transfer learning)", deep_features=512, handcrafted_features=4,
               total_features=516, handcrafted_names=config.HANDCRAFTED_NAMES, device=str(device),
               epochs_run=len(hist), best_epoch=best["epoch"], best_val_loss=best["val_loss"], best_val_acc=best["val_acc"],
               train_images=len(train_s), validation_images=len(val_s), test_images=len(test_s),
               dropped_near_duplicates=bool(args.drop_near_dups), training_minutes=round(minutes, 1))
    config.TRAINING_CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    print(f"\nTraining finished in {minutes:.1f} min. Best epoch {best['epoch']} "
          f"(val loss {best['val_loss']:.4f}, val acc {best['val_acc'] * 100:.2f}%). Saved {config.CHECKPOINT_PATH}")

    # 15. final evaluation on the untouched TEST set (done once, with the best checkpoint)
    if not args.no_eval:
        model.load_state_dict(best["state"])
        confidence_threshold_analysis(model, val_loader, device)
        run_evaluation(model, test_loader, device)
        print("\nNext: python features/feature_extraction.py   (writes the feature CSVs and fits the non-eye detector)")


if __name__ == "__main__":
    main()
