"""
train.py
--------
Reads images from ./real/ and ./screen/ (or paths passed via --real / --screen),
extracts hand-crafted features (see features.py), and trains a small
RandomForest classifier. Also prints cross-validated accuracy so you know
where you stand vs the 95% bar.

Usage:
    python train.py
    python train.py --real path/to/real --screen path/to/screen

Output:
    model.joblib   <- used automatically by predict.py if present
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

from features import extract_features, FEATURE_ORDER

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG", "*.webp", "*.HEIC")


def list_images(folder):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(set(files))


def build_dataset(real_dir, screen_dir):
    X, y, paths = [], [], []
    t0 = time.time()

    real_files = list_images(real_dir)
    screen_files = list_images(screen_dir)

    print(f"Found {len(real_files)} real images in '{real_dir}'")
    print(f"Found {len(screen_files)} screen images in '{screen_dir}'")

    if len(real_files) < 10 or len(screen_files) < 10:
        print("\nWARNING: very little data. Aim for 50+ per class for a reliable model.\n")

    for f in real_files:
        try:
            _, vec = extract_features(f)
            X.append(vec)
            y.append(0)  # 0 = real
            paths.append(f)
        except Exception as e:
            print(f"  skip {f}: {e}")

    for f in screen_files:
        try:
            _, vec = extract_features(f)
            X.append(vec)
            y.append(1)  # 1 = screen
            paths.append(f)
        except Exception as e:
            print(f"  skip {f}: {e}")

    print(f"Feature extraction done in {time.time() - t0:.1f}s for {len(X)} images")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="real")
    ap.add_argument("--screen", default="screen")
    ap.add_argument("--out", default="model.joblib")
    args = ap.parse_args()

    X, y, paths = build_dataset(args.real, args.screen)

    if len(set(y.tolist())) < 2 or len(X) < 6:
        print("\nNot enough data in both classes to train yet.")
        print("predict.py will fall back to the hand-tuned heuristic until you add data.")
        sys.exit(0)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    # Cross-validate to estimate real accuracy (small dataset -> use k-fold)
    n_splits = min(5, min(np.bincount(y)))
    n_splits = max(2, n_splits)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv)
    print(f"\nCross-validated accuracy ({n_splits}-fold): "
          f"{scores.mean()*100:.1f}%  (folds: {[round(s*100,1) for s in scores]})")

    # Fit final model on all data
    clf.fit(X, y)
    joblib.dump({"pipeline": clf, "feature_order": FEATURE_ORDER}, args.out)
    print(f"Saved model to {args.out}")

    # Feature importances (from the RF step) for interpretability
    rf = clf.named_steps["rf"]
    importances = sorted(zip(FEATURE_ORDER, rf.feature_importances_), key=lambda x: -x[1])
    print("\nFeature importances:")
    for name, imp in importances:
        print(f"  {name:24s} {imp:.3f}")


if __name__ == "__main__":
    main()
