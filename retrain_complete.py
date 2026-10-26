"""Complete retraining pipeline for the current 9,000-image dataset.

Steps:
1. Re-extract handcrafted features for all images
2. Re-extract deep (ResNet-18) features for all images  
3. Fit scaler on train handcrafted features only
4. Fit OOD statistics  
5. Train fusion classifier
6. Evaluate on test set
7. Save all artifacts
"""
import copy
import json
import os
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, 'backend')
import config
from features.handcrafted_features import extract_handcrafted
from models.resnet_fusion import ResNet18FeatureFusion, get_device
from preprocessing.image_preprocessing import preprocess, read_image_rgb
from training.dataset import list_samples

torch.set_num_threads(min(12, os.cpu_count() or 8))


def extract_all_handcrafted():
    """Extract handcrafted features for all images."""
    print("\n" + "="*70)
    print("STEP 1: Extracting handcrafted features for all images")
    print("="*70)
    
    all_rows = []
    for split in config.SPLITS:
        samples = list_samples(split)
        print(f"\n  Processing {split}: {len(samples)} images...")
        for rel, label in tqdm(samples, desc=f"  {split}", ncols=80):
            rgb = read_image_rgb(config.DATASET_DIR / rel)
            if rgb is None:
                raise IOError(f"Unreadable image: {rel}")
            feats = extract_handcrafted(preprocess(rgb, "dataset").roi_raw)
            all_rows.append({
                "image_path": rel,
                "split": split,
                "label": label,
                **feats
            })
    
    df = pd.DataFrame(all_rows)
    df.to_csv(config.HANDCRAFTED_CACHE_PATH, index=False)
    print(f"\nSaved handcrafted_cache.csv with {len(df)} rows")
    
    for split in config.SPLITS:
        sub = df[df['split'] == split]
        print(f"  {split}: {len(sub)} images, labels: {sub['label'].value_counts().to_dict()}")
    
    return df


def extract_all_deep_features(hand_df):
    """Extract ResNet-18 deep features for all images."""
    print("\n" + "="*70)
    print("STEP 2: Extracting ResNet-18 deep features for all images")
    print("="*70)
    
    device = get_device()
    # Use a fresh pretrained ResNet-18 for feature extraction (ImageNet weights)
    model = ResNet18FeatureFusion(pretrained=True).to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)
    ])
    
    hand_map = {r.image_path: r for r in hand_df.itertuples(index=False)}
    
    for split in config.SPLITS:
        samples = list_samples(split)
        print(f"\n  Extracting deep features for {split}: {len(samples)} images...")
        
        all_deep = []
        batch_size = 32
        
        for i in tqdm(range(0, len(samples), batch_size), desc=f"  {split}", ncols=80):
            batch_samples = samples[i:i+batch_size]
            imgs = []
            for rel, _ in batch_samples:
                rgb = read_image_rgb(config.DATASET_DIR / rel)
                if rgb is None:
                    raise IOError(f"Unreadable: {rel}")
                enhanced = preprocess(rgb, "dataset").enhanced
                imgs.append(transform(Image.fromarray(enhanced)))
            
            batch = torch.stack(imgs).to(device)
            with torch.inference_mode():
                feats = model.deep_features(batch).cpu().numpy().astype(np.float32)
            all_deep.append(feats)
        
        deep_all = np.concatenate(all_deep, axis=0)
        
        # Build full feature CSV
        rows = []
        for idx, (rel, label) in enumerate(samples):
            h_row = hand_map[rel]
            row_dict = {
                "image_path": rel,
                "split": split,
                "class": config.CLASS_NAMES[label],
                "label": label,
                "class_name": config.CLASS_NAMES[label],
            }
            for h_name in config.HANDCRAFTED_NAMES:
                row_dict[h_name] = getattr(h_row, h_name)
            for d_idx, d_name in enumerate(config.RESNET_FEATURE_NAMES):
                row_dict[d_name] = deep_all[idx, d_idx]
            rows.append(row_dict)
        
        df = pd.DataFrame(rows)
        csv_path = config.DATA_DIR / f"features_{split}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved {csv_path.name}: {len(df)} rows x {len(df.columns)} cols")


def fit_scaler_and_ood():
    """Fit StandardScaler on train handcrafted features and OOD stats."""
    print("\n" + "="*70)
    print("STEP 3: Fitting scaler and OOD statistics")
    print("="*70)
    
    tr_df = pd.read_csv(config.DATA_DIR / "features_train.csv")
    va_df = pd.read_csv(config.DATA_DIR / "features_validation.csv")
    
    # Fit scaler on TRAIN ONLY
    scaler = StandardScaler().fit(tr_df[config.HANDCRAFTED_NAMES].values)
    joblib.dump(scaler, config.SCALER_PATH)
    print(f"  Scaler fitted on {len(tr_df)} train samples")
    
    # Fit OOD stats
    from features.feature_extraction import fit_ood
    tr_deep = tr_df[config.RESNET_FEATURE_NAMES].values.astype(np.float32)
    tr_labels = tr_df["label"].values.astype(np.int64)
    va_deep = va_df[config.RESNET_FEATURE_NAMES].values.astype(np.float32)
    
    ood = fit_ood(tr_deep, tr_labels, va_deep)
    np.savez(config.OOD_STATS_PATH, **ood)
    print(f"  OOD stats saved (threshold={ood['threshold']:.2f})")
    
    return scaler


def train_classifier(scaler):
    """Train the fusion classifier on extracted features."""
    print("\n" + "="*70)
    print("STEP 4: Training Fusion Classifier")
    print("="*70)
    
    device = get_device()
    tr_df = pd.read_csv(config.DATA_DIR / "features_train.csv")
    va_df = pd.read_csv(config.DATA_DIR / "features_validation.csv")
    te_df = pd.read_csv(config.DATA_DIR / "features_test.csv")
    
    hand_cols = config.HANDCRAFTED_NAMES
    deep_cols = config.RESNET_FEATURE_NAMES
    
    print(f"  Train: {len(tr_df)}, Val: {len(va_df)}, Test: {len(te_df)}")
    
    def prepare_tensors(df):
        deep = df[deep_cols].values.astype(np.float32)
        hand = scaler.transform(df[hand_cols].values).astype(np.float32)
        fused = np.concatenate([deep, hand], axis=1)
        labels = df["label"].values.astype(np.int64)
        return torch.from_numpy(fused), torch.from_numpy(labels)
    
    x_train, y_train = prepare_tensors(tr_df)
    x_val, y_val = prepare_tensors(va_df)
    x_test, y_test = prepare_tensors(te_df)
    
    batch_size = 64
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
    
    # Build fresh model
    full_model = ResNet18FeatureFusion(pretrained=True).to(device)
    classifier = full_model.classifier.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    
    best = {"val_loss": float("inf"), "state": None, "epoch": 0, "val_acc": 0.0}
    epochs = 50
    patience = 10
    stale = 0
    hist = []
    t0 = time.time()
    
    print(f"\n  Training classifier (516 -> 128 -> 3) for up to {epochs} epochs...")
    for ep in range(1, epochs + 1):
        # Train
        classifier.train()
        t_loss, t_corr, t_n = 0.0, 0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = classifier(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * by.size(0)
            t_corr += (logits.argmax(1) == by).sum().item()
            t_n += by.size(0)
        
        # Validate
        classifier.eval()
        v_loss, v_corr, v_n = 0.0, 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = classifier(bx)
                loss = criterion(logits, by)
                v_loss += loss.item() * by.size(0)
                v_corr += (logits.argmax(1) == by).sum().item()
                v_n += by.size(0)
        
        tr_loss, tr_acc = t_loss / t_n, t_corr / t_n
        va_loss, va_acc = v_loss / v_n, v_corr / v_n
        scheduler.step(va_loss)
        
        flag = ""
        if va_loss < best["val_loss"] - 1e-4:
            best.update(val_loss=va_loss, val_acc=va_acc, epoch=ep,
                       state=copy.deepcopy(classifier.state_dict()))
            stale, flag = 0, " <- BEST"
        else:
            stale += 1
        
        hist.append({
            "epoch": ep, "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": va_loss, "val_acc": va_acc,
            "lr": optimizer.param_groups[0]["lr"]
        })
        print(f"  Epoch {ep:2d} | Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:5.2f}% | "
              f"Val Loss: {va_loss:.4f} Acc: {va_acc*100:5.2f}%{flag}")
        
        if stale >= patience:
            print(f"  Early stopping at epoch {ep}")
            break
    
    training_time = time.time() - t0
    
    # Save best model
    full_model.classifier.load_state_dict(best["state"])
    torch.save({
        "model_state": full_model.state_dict(),
        "class_names": config.CLASS_NAMES,
        "epoch": best["epoch"],
        "val_loss": best["val_loss"],
        "val_acc": best["val_acc"],
    }, config.CHECKPOINT_PATH)
    print(f"\n  Saved best model (epoch {best['epoch']}, val_acc={best['val_acc']*100:.2f}%)")
    
    # Save class names
    config.CLASS_NAMES_PATH.write_text(json.dumps(
        {str(i): n for i, n in enumerate(config.CLASS_NAMES)}, indent=2
    ))
    
    # ---- TEST EVALUATION ----
    print("\n" + "="*70)
    print("STEP 5: Evaluating on TEST set")
    print("="*70)
    
    classifier.load_state_dict(best["state"])
    classifier.eval()
    all_preds, all_probs, all_targets = [], [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            logits = classifier(bx)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_preds.append(probs.argmax(axis=1))
            all_targets.append(by.numpy())
    
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    y_probs = np.concatenate(all_probs, axis=0)
    
    test_acc = float((y_pred == y_true).mean())
    print(f"\n  Test Accuracy: {test_acc*100:.2f}% on {len(y_true)} samples")
    
    # Classification report
    rep = classification_report(y_true, y_pred, target_names=config.CLASS_NAMES, digits=4)
    print(f"\n{rep}")
    (config.RESULTS_DIR / "classification_report.txt").write_text(rep)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(config.CLASS_NAMES); ax.set_yticklabels(config.CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (Test Acc: {test_acc*100:.2f}%)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                   color="white" if cm[i, j] > cm.max()/2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    
    # Training history plot
    ep_range = range(1, len(hist) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1.plot(ep_range, [h["train_acc"] for h in hist], marker="o", label="Train")
    ax1.plot(ep_range, [h["val_acc"] for h in hist], marker="o", label="Val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy"); ax1.set_title("Accuracy"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(ep_range, [h["train_loss"] for h in hist], marker="o", label="Train")
    ax2.plot(ep_range, [h["val_loss"] for h in hist], marker="o", label="Val")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss"); ax2.set_title("Loss"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "training_accuracy.png", dpi=150)
    plt.close(fig)
    
    # Save metrics
    metrics = {
        "test_accuracy": round(test_acc, 4),
        "test_samples": int(len(y_true)),
        "validation_accuracy": round(best["val_acc"], 4),
        "validation_samples": int(len(va_df)),
        "training_samples": int(len(tr_df)),
        "best_epoch": int(best["epoch"]),
        "confusion_matrix": cm.tolist(),
    }
    (config.RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    
    # Save training config
    cfg = {
        "model": "ResNet-18 (ImageNet pretrained, transfer learning) + Handcrafted Feature Fusion",
        "deep_features": 512,
        "handcrafted_features": 4,
        "total_features": 516,
        "handcrafted_names": config.HANDCRAFTED_NAMES,
        "device": str(device),
        "epochs_run": len(hist),
        "best_epoch": best["epoch"],
        "best_val_loss": float(best["val_loss"]),
        "best_val_acc": float(best["val_acc"]),
        "test_accuracy": test_acc,
        "train_images": len(tr_df),
        "validation_images": len(va_df),
        "test_images": len(te_df),
        "total_images": len(tr_df) + len(va_df) + len(te_df),
        "training_minutes": round(training_time / 60, 2),
    }
    config.TRAINING_CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    
    # Confidence threshold analysis on validation
    val_probs = []
    with torch.no_grad():
        for bx, by in val_loader:
            bx = bx.to(device)
            pr = torch.softmax(classifier(bx), dim=1).cpu().numpy()
            val_probs.append(pr)
    val_probs = np.concatenate(val_probs, axis=0)
    v_conf, v_pred = val_probs.max(1), val_probs.argmax(1)
    y_val_np = y_val.numpy()
    
    thresh_rows = []
    for t in np.arange(0.40, 0.96, 0.05):
        keep = v_conf >= t
        acc_acc = float((v_pred[keep] == y_val_np[keep]).mean() * 100) if keep.any() else float("nan")
        thresh_rows.append({
            "threshold": round(float(t), 2),
            "coverage_pct": round(float(keep.mean() * 100), 2),
            "accuracy_on_accepted_pct": round(acc_acc, 2)
        })
    pd.DataFrame(thresh_rows).to_csv(config.RESULTS_DIR / "confidence_threshold_analysis.csv", index=False)
    
    (config.RESULTS_DIR / "training_history.json").write_text(json.dumps(hist, indent=2))
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print(f"  Best Epoch: {best['epoch']}")
    print(f"  Val Accuracy: {best['val_acc']*100:.2f}%")
    print(f"  Test Accuracy: {test_acc*100:.2f}%")
    print(f"  Training Time: {training_time/60:.1f} minutes")
    print("="*70)
    
    return test_acc


if __name__ == "__main__":
    config.RESULTS_DIR.mkdir(exist_ok=True)
    config.DATA_DIR.mkdir(exist_ok=True)
    config.MODELS_DIR.mkdir(exist_ok=True)
    
    hand_df = extract_all_handcrafted()
    extract_all_deep_features(hand_df)
    scaler = fit_scaler_and_ood()
    test_acc = train_classifier(scaler)
