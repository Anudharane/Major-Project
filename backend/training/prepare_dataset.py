"""Extract, verify and audit Dataset.zip. Never modifies or deletes images unless asked.

Usage (from backend/):
    python training/prepare_dataset.py --zip ../Dataset.zip      # extract into ../dataset then audit
    python training/prepare_dataset.py                           # audit an already-extracted ../dataset
    python training/prepare_dataset.py --corr 0.99                # near-duplicate sensitivity

Outputs (results/): dataset_report.txt, class_distribution.png, leakage_flags.csv
"""
import argparse
import hashlib
import shutil
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from preprocessing.image_preprocessing import read_image_rgb



def extract(zip_path: Path):
    if config.DATASET_DIR.exists() and any(config.DATASET_DIR.iterdir()):
        print(f"{config.DATASET_DIR} already exists - not overwriting. Delete it to re-extract.")
        return
    with tempfile.TemporaryDirectory() as tmp:
        print(f"Extracting {zip_path} ...")
        zipfile.ZipFile(zip_path).extractall(tmp)
        root = next((p.parent for p in Path(tmp).rglob("train") if p.is_dir()), None)
        if root is None:
            sys.exit("No 'train' folder found inside the ZIP.")
        shutil.copytree(root, config.DATASET_DIR, dirs_exist_ok=True)


def _vec(path) -> np.ndarray:
    """32x32 grayscale, zero-mean, unit-norm -> dot product == Pearson correlation."""
    g = cv2.cvtColor(read_image_rgb(path), cv2.COLOR_RGB2GRAY)
    v = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path)
    ap.add_argument("--corr", type=float, default=0.99, help="correlation above which two images count as near-duplicates")
    ap.add_argument("--remove-corrupt", action="store_true", help="delete unreadable images (default: only report)")
    args = ap.parse_args()
    if args.zip:
        extract(args.zip)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    out = []
    log = lambda s="": (print(s), out.append(s))

    # --- structure + counts
    log("=== DATASET REPORT ===")
    rows = []
    for split in config.SPLITS:
        for cls in config.CLASS_NAMES:
            d = config.DATASET_DIR / split / cls
            if not d.is_dir():
                sys.exit(f"Missing folder: {d}")
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in config.IMAGE_EXTENSIONS:
                    rows.append({"split": split, "class": cls, "rel": p.relative_to(config.DATASET_DIR).as_posix(), "path": p})
    df = pd.DataFrame(rows)
    counts = df.groupby(["split", "class"]).size().unstack().reindex(index=config.SPLITS, columns=config.CLASS_NAMES)
    log(f"\nImage counts (total {len(df)}):\n{counts.to_string()}")

    # --- readability, size, hashes
    bad, sizes, md5s, ok_idx = [], Counter(), defaultdict(list), []
    for i, r in enumerate(df.itertuples()):
        rgb = read_image_rgb(r.path)
        if rgb is None:
            bad.append(r.rel); continue
        sizes[rgb.shape[:2]] += 1
        md5s[hashlib.md5(r.path.read_bytes()).hexdigest()].append(r.rel)
        ok_idx.append(i)
    log(f"\nUnreadable/corrupted images: {len(bad)}" + (f"  e.g. {bad[:5]}" if bad else ""))
    log(f"Image sizes (HxW): {dict(sizes.most_common(5))}")
    if bad and args.remove_corrupt:
        for rel in bad:
            (config.DATASET_DIR / rel).unlink()
        log("  removed corrupted files (--remove-corrupt).")

    exact = [v for v in md5s.values() if len(v) > 1]
    cross_exact = [v for v in exact if len({x.split('/')[0] for x in v}) > 1]
    log(f"Exact duplicate groups: {len(exact)}  (spanning splits: {len(cross_exact)})")

        # --- near duplicates / identity leakage (32x32 grayscale Pearson correlation)
    ok = df.iloc[ok_idx].reset_index(drop=True)
    X = np.stack([_vec(p) for p in ok.path])
    sp = ok.split.values
    thr = args.corr
    max_c, nearest, flag = np.zeros(len(ok), np.float32), [""] * len(ok), np.zeros(len(ok), bool)
    tr_idx, ev_idx = np.where(sp == "train")[0], np.where(sp != "train")[0]
    for s0 in range(0, len(ok), 400):
        rows_ = np.arange(s0, min(s0 + 400, len(ok)))
        S = X[rows_] @ X[ev_idx].T                      # every image vs every val/test image
        S2 = X[rows_] @ X[tr_idx].T                     # every image vs every train image
        for k, i in enumerate(rows_):
            if sp[i] == "train":                        # train image: is it a near-copy of a val/test image?
                j = S[k].argmax(); max_c[i], nearest[i] = S[k, j], ok.rel[ev_idx[j]]
            else:                                       # val/test image: near-copy of a train image?
                j = S2[k].argmax(); max_c[i], nearest[i] = S2[k, j], ok.rel[tr_idx[j]]
            flag[i] = max_c[i] > thr
    flags = pd.DataFrame({"image_path": ok.rel, "split": sp, "class": ok["class"], "max_corr_to_other_split": max_c.round(5),
                          "nearest_image_in_other_split": nearest, "near_duplicate": flag})
    flags.to_csv(config.RESULTS_DIR / "leakage_flags.csv", index=False)
    log(f"\nIdentity / near-duplicate leakage (Pearson corr of 32x32 grayscale > {thr}):")
    for name in ("validation", "test"):
        m = sp == name
        log(f"  {name:11s}: {flag[m].mean() * 100:5.1f}% of images ({flag[m].sum()}/{m.sum()}) have a near-copy in TRAIN")
    m = sp == "train"
    log(f"  train      : {flag[m].mean() * 100:5.1f}% of images ({flag[m].sum()}/{m.sum()}) have a near-copy in VALIDATION/TEST")
    clean = (~flag[sp == "test"]).sum()
    log(f"  clean TEST subset (no near-copy in train): {clean} images -> evaluate.py reports metrics on it as well")
    # redundancy inside train
    sub = X[tr_idx[:2000]]; Sx = sub @ sub.T; np.fill_diagonal(Sx, 0)
    log(f"  within TRAIN (first 2000 imgs): {(Sx.max(1) > thr).mean() * 100:.1f}% have another train image with corr > {thr}")
    log("  => train/val/test were NOT built from independent subjects. A high test accuracy on the full test set")
    log("     partly measures memorisation of shared eye identities, not generalisation to new people.")

    # --- simple appearance stats per class (label sanity check)
    log("\nMean brightness per class (first 300 train images each):")
    for cls in config.CLASS_NAMES:
        sub = df[(df.split == "train") & (df["class"] == cls)].head(300)
        m = np.mean([read_image_rgb(p).mean() for p in sub.path])
        log(f"  {cls:10s} {m:6.1f}")
    log("\nNOTE: folder labels are NOT clinically verified. Inspect samples visually; see README 'Limitations'.")

    (config.RESULTS_DIR / "dataset_report.txt").write_text("\n".join(out) + "\n")
    ax = counts.plot(kind="bar", figsize=(6.5, 4)); ax.set_ylabel("Images"); ax.set_title("Class distribution")
    plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(config.RESULTS_DIR / "class_distribution.png", dpi=150)
    print(f"\nSaved results/dataset_report.txt, class_distribution.png, leakage_flags.csv")


if __name__ == "__main__":
    main()
