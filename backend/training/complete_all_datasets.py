"""Extract all missing features and train the complete dataset.

Ensures:
1. handcrafted_cache.csv has all 12,900 images (9,300 train, 1,350 val, 2,250 test)
2. features_train.csv has all 9,300 images x 521 cols
3. features_validation.csv has all 1,350 images x 521 cols
4. features_test.csv has all 2,250 images x 521 cols
5. feature_statistics.csv computed across all 9,300 train samples
6. ood_stats.npz fitted on all 9,300 train samples (threshold on 1,350 val samples)
7. scaler.pkl fitted on all 9,300 train handcrafted features
8. Fusion classifier trained on all 9,300 train samples, validated on 1,350 val, evaluated on 2,250 test
9. All evaluation metrics, reports, and best_model.pth updated
"""
import copy
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from features.handcrafted_features import extract_handcrafted
from models.resnet_fusion import ResNet18FeatureFusion, get_device, load_checkpoint
from preprocessing.image_preprocessing import preprocess, read_image_rgb
from training.dataset import list_samples

torch.set_num_threads(min(12, os.cpu_count() or 8))


def _process_single_image(args):
    rel, label, split = args
    rgb = read_image_rgb(config.DATASET_DIR / rel)
    if rgb is None:
        raise IOError(f"Unreadable image: {rel}")
    feats = extract_handcrafted(preprocess(rgb, "dataset").roi_raw)
    return {"image_path": rel, "split": split, "label": label, **feats}


def complete_handcrafted_cache():
    print("\n--- 1. Checking and completing handcrafted_cache.csv ---")
    cache_path = config.HANDCRAFTED_CACHE_PATH
    existing_df = pd.DataFrame()
    existing_paths = set()
    if cache_path.exists():
        existing_df = pd.read_csv(cache_path)
        existing_paths = set(existing_df["image_path"].values)
        print(f"Existing cache has {len(existing_df)} images.")

    missing_samples = []
    for split in config.SPLITS:
        for rel, label in list_samples(split):
            if rel not in existing_paths:
                missing_samples.append((rel, label, split))

    if missing_samples:
        print(f"Extracting handcrafted features for {len(missing_samples)} new images...")
        new_rows = []
        with ThreadPoolExecutor(max_workers=8) as ex:
            for res in tqdm(ex.map(_process_single_image, missing_samples), total=len(missing_samples), desc="Handcrafted", ncols=80):
                new_rows.append(res)
        new_df = pd.DataFrame(new_rows)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        print("All handcrafted features already cached!")
        combined_df = existing_df

    # Sort in order of list_samples for reproducibility
    all_ordered = []
    for split in config.SPLITS:
        for rel, _ in list_samples(split):
            all_ordered.append(rel)
    path_to_row = {r.image_path: r for r in combined_df.itertuples(index=False)}
    combined_df = pd.DataFrame([path_to_row[p]._asdict() for p in all_ordered])
    combined_df.to_csv(cache_path, index=False)
    print(f"handcrafted_cache.csv now has {len(combined_df)} rows.")
    return combined_df


def extract_deep_batch(model, img_paths, device, transform):
    imgs = []
    for p in img_paths:
        rgb = read_image_rgb(config.DATASET_DIR / p)
        if rgb is None:
            raise IOError(f"Unreadable: {p}")
        enhanced = preprocess(rgb, "dataset").enhanced
        imgs.append(transform(Image.fromarray(enhanced)))
    batch = torch.stack(imgs).to(device)
    with torch.inference_mode():
        feats = model.deep_features(batch).cpu().numpy().astype(np.float32)
    return feats


def complete_feature_csvs():
    print("\n--- 2. Checking and completing feature CSV files ---")
    device = get_device()
    model, _ = load_checkpoint(config.CHECKPOINT_PATH, device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)
    ])

    hand_df = pd.read_csv(config.HANDCRAFTED_CACHE_PATH)
    hand_map = {r.image_path: r for r in hand_df.itertuples(index=False)}

    for split in config.SPLITS:
        csv_path = config.DATA_DIR / f"features_{split}.csv"
        existing_df = pd.DataFrame()
        existing_paths = set()
        if csv_path.exists():
            existing_df = pd.read_csv(csv_path)
            existing_paths = set(existing_df["image_path"].values)
            print(f"Existing {split} CSV has {len(existing_df)} images.")

        samples = list_samples(split)
        missing_samples = [(p, l) for p, l in samples if p not in existing_paths]

        if missing_samples:
            print(f"Extracting deep features for {len(missing_samples)} images in {split}...")
            batch_size = 64
            all_deep = []
            for i in tqdm(range(0, len(missing_samples), batch_size), desc=f"Deep [{split}]", ncols=80):
                batch_samples = missing_samples[i:i + batch_size]
                paths = [p for p, _ in batch_samples]
                deep_batch = extract_deep_batch(model, paths, device, transform)
                all_deep.append(deep_batch)
            deep_all = np.concatenate(all_deep, axis=0)

            new_rows = []
            for idx, (p, label) in enumerate(missing_samples):
                h_row = hand_map[p]
                row_dict = {
                    "image_path": p,
                    "split": split,
                    "class": config.CLASS_NAMES[label],
                    "label": label,
                    "class_name": config.CLASS_NAMES[label],
                }
                for h_name in config.HANDCRAFTED_NAMES:
                    row_dict[h_name] = getattr(h_row, h_name)
                for d_idx, d_name in enumerate(config.RESNET_FEATURE_NAMES):
                    row_dict[d_name] = deep_all[idx, d_idx]
                new_rows.append(row_dict)

            new_df = pd.DataFrame(new_rows)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            print(f"All images in {split} already have features extracted.")
            combined_df = existing_df

        # Reorder to match exact sample order
        path_to_row = {r.image_path: r for r in combined_df.itertuples(index=False)}
        ordered_rows = [path_to_row[p]._asdict() for p, _ in samples]
        final_df = pd.DataFrame(ordered_rows)
        final_df.to_csv(csv_path, index=False)
        print(f"Saved {csv_path.name} with {len(final_df)} rows x {len(final_df.columns)} cols.")

    # 3. Compute and save feature statistics on TRAIN split
    print("\n--- 3. Updating feature_statistics.csv (TRAIN split) ---")
    tr_df = pd.read_csv(config.DATA_DIR / "features_train.csv")
    cols = config.HANDCRAFTED_NAMES + config.RESNET_FEATURE_NAMES
    tr_feats = tr_df[cols]
    stats_df = pd.DataFrame({
        "feature_name": cols,
        "mean": tr_feats.mean().values,
        "std": tr_feats.std().values,
        "min": tr_feats.min().values,
        "max": tr_feats.max().values,
    })
    stats_path = config.DATA_DIR / "feature_statistics.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved {stats_path.name} ({len(stats_df)} features).")


def update_ood_and_scaler():
    print("\n--- 4. Updating scaler.pkl and ood_stats.npz ---")
    from features.feature_extraction import fit_ood
    tr_df = pd.read_csv(config.DATA_DIR / "features_train.csv")
    va_df = pd.read_csv(config.DATA_DIR / "features_validation.csv")

    # Fit scaler on TRAIN ONLY
    scaler = StandardScaler().fit(tr_df[config.HANDCRAFTED_NAMES].values)
    joblib.dump(scaler, config.SCALER_PATH)
    print(f"Scaler fitted on {len(tr_df)} train samples and saved to {config.SCALER_PATH.name}")

    # Fit OOD stats on TRAIN deep features, threshold on VAL deep features
    tr_deep = tr_df[config.RESNET_FEATURE_NAMES].values.astype(np.float32)
    tr_labels = tr_df["label"].values.astype(np.int64)
    va_deep = va_df[config.RESNET_FEATURE_NAMES].values.astype(np.float32)

    ood = fit_ood(tr_deep, tr_labels, va_deep)
    np.savez(config.OOD_STATS_PATH, **ood)
    print(f"OOD stats saved (threshold={ood['threshold']:.2f}, val median={ood['val_dist_median']:.2f})")
    return scaler


def train_and_evaluate(scaler):
    print("\n--- 5. Training Fusion Classifier on All 9,300 Training Images ---")
    device = get_device()
    tr_df = pd.read_csv(config.DATA_DIR / "features_train.csv")
    va_df = pd.read_csv(config.DATA_DIR / "features_validation.csv")
    te_df = pd.read_csv(config.DATA_DIR / "features_test.csv")

    hand_cols = config.HANDCRAFTED_NAMES
    deep_cols = config.RESNET_FEATURE_NAMES

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

    full_model = ResNet18FeatureFusion(pretrained=True).to(device)
    classifier = full_model.classifier.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

    best = {"val_loss": float("inf"), "state": None, "epoch": 0, "val_acc": 0.0}
    epochs = 35
    patience = 8
    stale = 0
    hist = []
    t0 = time.time()

    for ep in range(1, epochs + 1):
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
            best.update(val_loss=va_loss, val_acc=va_acc, epoch=ep, state=copy.deepcopy(classifier.state_dict()))
            stale = 0
            flag = " <- best"
        else:
            stale += 1

        hist.append({"epoch": ep, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc})
        print(f"Epoch {ep:2d} | Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:5.2f}% | Val Loss: {va_loss:.4f} Acc: {va_acc*100:5.2f}%{flag}")

        if stale >= patience:
            print(f"Early stopping at epoch {ep} (no improvement for {patience} epochs)")
            break

    # Save full best model checkpoint
    full_model.classifier.load_state_dict(best["state"])
    torch.save({
        "model_state": full_model.state_dict(),
        "class_names": config.CLASS_NAMES,
        "epoch": best["epoch"],
        "val_loss": best["val_loss"],
        "val_acc": best["val_acc"],
    }, config.CHECKPOINT_PATH)
    print(f"\nSaved updated checkpoint {config.CHECKPOINT_PATH.name} (Val Acc: {best['val_acc']*100:.2f}%)")

    # Evaluate on TEST SET (all 2,250 test images)
    print("\n--- 6. Evaluating on All 2,250 Test Images ---")
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
    print(f"Test Accuracy on 2,250 samples: {test_acc*100:.2f}%")

    # Classification report
    rep = classification_report(y_true, y_pred, target_names=config.CLASS_NAMES, digits=4)
    print("\nClassification Report:\n", rep)
    (config.RESULTS_DIR / "classification_report.txt").write_text(rep)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(config.CLASS_NAMES); ax.set_yticklabels(config.CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Confusion Matrix (Test Acc: {test_acc*100:.2f}%)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Training accuracy & loss plots
    ep_range = range(1, len(hist) + 1)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(ep_range, [h["train_acc"] for h in hist], marker="o", label="Train Acc")
    ax.plot(ep_range, [h["val_acc"] for h in hist], marker="o", label="Val Acc")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.set_title("Training vs Validation Accuracy")
    ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "training_accuracy.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(ep_range, [h["train_loss"] for h in hist], marker="o", label="Train Loss")
    ax.plot(ep_range, [h["val_loss"] for h in hist], marker="o", label="Val Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Training vs Validation Loss")
    ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "training_loss.png", dpi=150)
    plt.close(fig)

    # Metrics JSON
    metrics = {
        "test_accuracy": test_acc,
        "test_samples": len(y_true),
        "validation_accuracy": best["val_acc"],
        "validation_samples": len(va_df),
        "training_samples": len(tr_df),
        "best_epoch": best["epoch"],
        "confusion_matrix": cm.tolist(),
    }
    (config.RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Update training_config.json
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
        "training_minutes": round((time.time() - t0) / 60, 2),
    }
    config.TRAINING_CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    print(f"Updated {config.TRAINING_CONFIG_PATH.name}")
    print("\nAll datasets processed, CSVs created/updated, and model fully trained successfully!")


def main():
    complete_handcrafted_cache()
    complete_feature_csvs()
    scaler = update_ood_and_scaler()
    train_and_evaluate(scaler)


if __name__ == "__main__":
    main()
