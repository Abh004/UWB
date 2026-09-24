"""
x7_2s_eval.py
=============
Offline evaluation of the dual-sensor 5-class gesture CNN.

IMPORTANT -- what this script reports
-------------------------------------
By default it evaluates the VALIDATION split only, reproducing exactly the
deterministic stratified split that x7_2s_train.py uses (same seed, same
per-class proportion). Earlier revisions evaluated the entire dataset, i.e.
mostly the data the model had been trained on, which inflated every figure it
printed. Pass --all to reproduce that legacy behaviour, but treat the result as
a training-fit diagnostic, not as generalisation.

A caveat that --split cannot fix: each gesture class was captured in only 4-5
bursts of ~25 back-to-back trials, so trials within a burst are near-duplicates.
Any per-file split therefore leaves correlated samples on both sides. Use
--group to split by recording burst instead, which is the honest estimate.
"""

import argparse
import os
import random
import re
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from x7_2s_train import (
    X7DualSensorDataset, GestureCNN_X7_Dual, enumerate_recordings,
    N_CLASSES, GESTURE_NAMES, BATCH_SIZE,
    GESTURE_NAMES_5, GESTURE_NAMES_6, GESTURE_FOLDERS_5, GESTURE_FOLDERS_6,
)


def burst_id(path, gap_s=10):
    """Group trials recorded back-to-back into one session id."""
    m = re.match(r".*?_(\d{10})_\d+\.npy$", os.path.basename(path))
    return int(m.group(1)) if m else None


def build_indices(all_files, mode):
    """Return the evaluation indices for the requested split mode."""
    if mode == "all":
        return list(range(len(all_files)))

    random.seed(42)
    by_class = defaultdict(list)
    for idx, (_, label) in enumerate(all_files):
        by_class[label].append(idx)

    if mode == "split":
        # Mirrors x7_2s_train.py exactly.
        val = []
        for label in sorted(by_class):
            idxs = by_class[label][:]
            random.shuffle(idxs)
            val.extend(idxs[:max(1, int(round(0.20 * len(idxs))))])
        return sorted(val)

    # mode == "group": hold out whole recording bursts.
    val = []
    for label in sorted(by_class):
        groups = defaultdict(list)
        for idx in by_class[label]:
            ts = burst_id(str(all_files[idx][0]))
            groups[ts if ts is not None else idx].append(idx)
        keys = sorted(groups)
        # Cluster timestamps within gap_s into the same burst.
        merged, cur = [], [keys[0]]
        for k in keys[1:]:
            if isinstance(k, int) and isinstance(cur[-1], int) and k - cur[-1] <= 10:
                cur.append(k)
            else:
                merged.append(cur); cur = [k]
        merged.append(cur)
        random.shuffle(merged)
        n_val = max(1, int(round(0.20 * len(merged))))
        for burst in merged[:n_val]:
            for k in burst:
                val.extend(groups[k])
    return sorted(val)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_paths = [p.strip() for p in args.model.split(",") if p.strip()]
    models = []
    n_classes = None
    for mp in model_paths:
        ckpt = torch.load(mp, map_location=device)
        n_out = ckpt["classifier.6.weight"].shape[0]
        if n_classes is None:
            n_classes = n_out
        m = GestureCNN_X7_Dual(n_classes=n_out).to(device)
        m.load_state_dict(ckpt)
        m.eval()
        models.append(m)

    gesture_names = GESTURE_NAMES_6 if n_classes == 6 else GESTURE_NAMES_5
    folders = GESTURE_FOLDERS_6 if n_classes == 6 else GESTURE_FOLDERS_5

    all_files = enumerate_recordings(args.data, verbose=True, gesture_folders=folders)
    if not all_files:
        print(f"[ERROR] No valid data found in {args.data}.")
        return

    indices = build_indices(all_files, args.mode)
    dataset = X7DualSensorDataset(args.data, augment=False,
                                  file_indices=indices, all_files=all_files,
                                  gesture_folders=folders)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    label_desc = {
        "split": "VALIDATION split (per-file, mirrors training)",
        "group": "HELD-OUT RECORDING BURSTS (no near-duplicate leakage)",
        "all":   "ENTIRE dataset -- INCLUDES TRAINING DATA, fit diagnostic only",
    }[args.mode]
    print(f"\n[INFO] Evaluating {len(models)} model(s) on: {label_desc}")
    print(f"[INFO] {len(dataset)} of {len(all_files)} recordings\n")

    def evaluate(use1=True, use2=True):
        preds_all, labels_all = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.clone()
                if not use1:
                    x[:, 0] = 0
                if not use2:
                    x[:, 1] = 0
                x = x.to(device)
                probs = torch.zeros((x.size(0), n_classes), device=device)
                for m in models:
                    probs += torch.softmax(m(x), dim=1)
                preds = torch.argmax(probs, dim=1)
                preds_all.extend(preds.cpu().numpy())
                labels_all.extend(y.cpu().numpy())
        return np.array(labels_all), np.array(preds_all)

    labels, preds = evaluate(True, True)
    cm = confusion_matrix(labels, preds, labels=range(n_classes))

    _, preds_s2 = evaluate(False, True)
    _, preds_s1 = evaluate(True, False)

    print("--- RESULTS ---")
    print(f"Overall Accuracy (Both Sensors)       : {np.mean(labels == preds)*100:.2f}%")
    print(f"Accuracy w/o Sensor 1 (Sensor 2 Only) : {np.mean(labels == preds_s2)*100:.2f}%")
    print(f"Accuracy w/o Sensor 2 (Sensor 1 Only) : {np.mean(labels == preds_s1)*100:.2f}%")

    print("\n--- PER-CLASS ---")
    print(f"{'Class':<12} {'Prec':>7} {'Recall':>7} {'F1':>7} {'N':>5}")
    for i, name in enumerate(gesture_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"{name:<12} {prec*100:6.1f}% {rec*100:6.1f}% {f1*100:6.1f}% {cm[i].sum():5d}")

    print("\n--- CONFUSION MATRIX (row: true, col: predicted) ---")
    print("            " + " ".join(f"{n[:6]:>7}" for n in gesture_names))
    for i, name in enumerate(gesture_names):
        print(f"{name:<12}" + " ".join(f"{cm[i, j]:7d}" for j in range(n_classes)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate the 6-class gesture CNN")
    ap.add_argument("--data", default="gesture_dataset_2s_new")
    ap.add_argument("--model", default=os.path.join("models", "best_model_2s_5class.pth"))
    ap.add_argument("--mode", choices=["split", "group", "all"], default="split",
                    help="split: per-file val split (default) | "
                         "group: hold out whole recording bursts | "
                         "all: entire dataset (training-fit diagnostic)")
    run(ap.parse_args())
