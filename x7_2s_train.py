"""
x7_2s_train.py
====================
Trains a dual-sensor 3D gesture recognition CNN using Late Fusion 
(two separate feature branches) across 6 pure gesture classes:
1. SWIPE_LR
2. SWIPE_RL
3. SWIPE_UP
4. SWIPE_DOWN
5. PUSH_IN
6. PINCH

Trained natively on GPU (CUDA) using complex interferometric unit phasors.
"""

import argparse
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from x7_physics import (
    mti_filter, unit_phasor, complex_gaussian_noise, mti_noise_scale,
    max_unambiguous_velocity, BIN_LENGTH_M, max_range, EPS,
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
WINDOW_SIZE = 64   # 64 frames @ 40 FPS = 1.60 s
RANGE_BINS = 34    # 34 bins x 71.4 mm, offset -1.38 m -> -1.38 .. +0.98 m
                   # (measured; the old '6.5 mm -> 0.21 m' figure was wrong 11x)
RADAR_FPS = 40
N_FEATURE_CHANNELS = 10 # see to_interferometry_tensor()

# Cross-session augmentation. Measured by leave-one-burst-out (the only metric
# that predicts live behaviour), these lift held-out-session accuracy from
# 60.0% to 76.4%; a per-file split cannot see the difference.
RANGE_JITTER_BINS = 3        # +/-3 bins = +/-19.5 mm of standoff variation
TIME_SCALE_RANGE = (0.7, 1.4)  # gesture pace relative to the recorded pace
# PINCH is deliberately excluded. It was the gated-mode activation gesture, but
# it never triggered reliably on live hardware, and while present in the label
# space it also drew probability mass away from the five directional gestures
# (measured: 98.3% vs 96.8% on the same recordings once its logit is removed).
# The system now runs continuously with no activation gesture.
GESTURE_FOLDERS_5 = [
    "G01_SWIPE_LR", "G02_SWIPE_RL", "G03_SWIPE_UD", "G04_SWIPE_DU", "G05_PUSH_IN"
]
GESTURE_NAMES_5 = ["SWIPE_LR", "SWIPE_RL", "SWIPE_UD", "SWIPE_DU", "PUSH_IN"]

GESTURE_FOLDERS_6 = [
    "G01_SWIPE_LR", "G02_SWIPE_RL", "G03_SWIPE_UD", "G04_SWIPE_DU", "G05_PUSH_IN", "G06_PINCH"
]
GESTURE_NAMES_6 = ["SWIPE_LR", "SWIPE_RL", "SWIPE_UD", "SWIPE_DU", "PUSH_IN", "PINCH"]

# Active class set: 5-class, PINCH EXCLUDED.
#
# PINCH was re-added on 2026-09-09 and the 6-class ensemble (m6_v1_seed*)
# collapsed on live data: 33.0% vs 94.7% for the 5-class ensemble on the same
# 209 live captures. PINCH absorbed 95 of those 209 (45%) as false positives.
#
# Cause: debug_captures/ contains NO live PINCH recordings, so PINCH was trained
# only on the recorded corpus, whose geometry differs from the deployment
# environment by ~26 deg. The five classes that DO have live captures learned the
# live geometry; PINCH did not, and its unconstrained decision region swallowed
# live-geometry samples from every other class.
#
# To re-enable PINCH: capture ~35 live PINCH gestures with x7_2s_debug_live.py
# first, then switch these three lines to the _6 variants and retrain.
GESTURE_FOLDERS = GESTURE_FOLDERS_5
GESTURE_NAMES = GESTURE_NAMES_5
N_CLASSES = 5

# ── PREPROCESSING ─────────────────────────────────────────────────────────────
def to_interferometry_tensor(w: np.ndarray) -> torch.Tensor:
    """
    Transform a (64, 2, 2, 34) complex radar window into a (10, 34, 64) float32
    tensor of bounded, physically explicit channels:

      0    : Normalised MTI amplitude
      1,2  : Horizontal AoA unit phasor  (Re, Im)  -- TX0/TX1 measured on RX0
      3,4  : Vertical   AoA unit phasor  (Re, Im)  -- RX0/RX1 receive baseline
      5,6  : Inter-frame Doppler phasor  (Re, Im)
      7    : Horizontal amplitude skew
      8,9  : Second horizontal AoA phasor (Re, Im) -- TX0/TX1 measured on RX1

    Physics notes
    -------------
    * The two spatial phasors sample DIFFERENT apertures. The horizontal pair
      varies the transmitter with the receiver fixed, so its phase difference
      maps through the TX baseline d_TX; the vertical pair varies the receiver,
      mapping through d_RX. Both are one-way path differences:
          delta_phi = 2*pi * d * sin(theta) / lambda
      They are NOT interchangeable and share no common `d`.
    * The Doppler phasor aliases beyond +/-lambda*FPS/4 = +/-0.41 m/s at 40 FPS.
      Faster radial motion wraps; the CNN sees the wrapped phase. Encoding the
      phasor as (Re, Im) rather than as a raw angle keeps that wrap continuous
      instead of introducing a +pi/-pi discontinuity mid-gesture.
    * Unit phasors discard amplitude deliberately, so channels 1-6 stay bounded
      in [-1, 1] regardless of target reflectivity; channels 0 and 7 carry the
      amplitude information separately in normalised form.
    """
    # 1. MTI clutter removal
    w_mti = mti_filter(w, axis=0)

    tx0_rx0 = w_mti[:, 0, 0, :]
    tx1_rx0 = w_mti[:, 1, 0, :]
    tx0_rx1 = w_mti[:, 0, 1, :]
    tx1_rx1 = w_mti[:, 1, 1, :]

    # 2. Spatial unit phasors (angle of arrival).
    #
    # The X7 provides 2 TX x 2 RX = 4 virtual channels. Earlier revisions used
    # only three of them; tx1_rx1 was loaded and discarded. Pairing it with
    # tx0_rx1 yields a SECOND horizontal baseline measured on the other receiver,
    # which both extends the effective transmit aperture and gives a redundant
    # azimuth estimate that averages away per-session phase offsets.
    #
    # Measured by leave-one-burst-out (the metric that predicts live behaviour),
    # adding this pair is the single largest gain found: 5-class cross-session
    # accuracy rises from ~61% to 70.5%.
    H_phasor = unit_phasor(tx0_rx0, tx1_rx0)   # TX baseline on RX0 -> azimuth
    H2_phasor = unit_phasor(tx0_rx1, tx1_rx1)  # TX baseline on RX1 -> azimuth
    V_phasor = unit_phasor(tx0_rx0, tx0_rx1)   # RX baseline        -> elevation

    # 3. Temporal unit phasor (radial Doppler)
    D_phasor = np.zeros_like(tx0_rx0, dtype=np.complex64)
    D_phasor[1:] = unit_phasor(tx0_rx0[1:], tx0_rx0[:-1])

    # 4. Amplitude features
    amp = np.mean(np.abs(w_mti), axis=(1, 2))
    amp_norm = (amp - np.mean(amp)) / (np.std(amp) + EPS)

    amp_H_diff = np.abs(tx0_rx0) - np.abs(tx1_rx0)
    amp_H_norm = amp_H_diff / (np.abs(tx0_rx0) + np.abs(tx1_rx0) + EPS)

    stacked = np.stack([
        amp_norm,
        np.real(H_phasor), np.imag(H_phasor),
        np.real(V_phasor), np.imag(V_phasor),
        np.real(D_phasor), np.imag(D_phasor),
        amp_H_norm,
        np.real(H2_phasor), np.imag(H2_phasor),
    ], axis=0)

    # PyTorch expects (Channels, H, W) -> (8, bins, frames)
    norm = np.transpose(stacked, (0, 2, 1))
    return torch.from_numpy(norm).float()

def enumerate_recordings(root: str, verbose: bool = False, gesture_folders: list = None):
    """
    Return [(path, label_idx), ...] for every VALID recording under `root`.
    Scan root for recording files, returning a list of (Path, int_label) pairs.
    Recordings with non-standard tensor shapes are rejected and reported once.

    Validation happens here rather than inside the Dataset so that the index
    space used to build the train/val split is exactly the one the Dataset
    consumes. Previously the split was computed over the raw file count while
    the Dataset silently dropped malformed files afterwards, so a single bad
    recording shifted every subsequent index and leaked validation files into
    the training set.
    """
    root_path = Path(root)
    files, rejected = [], []
    folders = gesture_folders if gesture_folders is not None else GESTURE_FOLDERS
    for label_idx, folder in enumerate(folders):
        class_dir = root_path / folder
        if not class_dir.exists():
            continue
        for fp in sorted(class_dir.rglob("*.npy")):
            try:
                shape = np.load(fp, mmap_mode="r").shape
            except Exception as exc:
                rejected.append((fp.name, f"unreadable: {exc}"))
                continue
            if (len(shape) != 5 or shape[1] != 2 or shape[2] != 2
                    or shape[3] != 2 or shape[0] < 40):
                rejected.append((fp.name, f"bad shape {shape}"))
                continue
            files.append((fp, label_idx))

    if verbose and rejected:
        print(f"[WARN] Rejected {len(rejected)} malformed recording(s):")
        for name, why in rejected[:10]:
            print(f"         - {name}: {why}")
        if len(rejected) > 10:
            print(f"         ... and {len(rejected) - 10} more")
    return files


class X7DualSensorDataset(Dataset):
    def __init__(self, root: str, augment: bool = False, file_indices: list = None,
                 all_files: list = None, gesture_folders: list = None):
        self.augment = augment
        self.samples = []

        if all_files is None:
            all_files = enumerate_recordings(root, gesture_folders=gesture_folders)
        if file_indices is not None:
            all_files = [all_files[i] for i in file_indices]

        for fp, true_label in all_files:
            try:
                self.samples.append((np.load(fp), true_label))
            except Exception as e:
                print(f"[WARN] Error loading {fp}: {e}")

        print(f"[INFO] Loaded {len(self.samples)} recordings (augment={augment}).")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        raw, label = self.samples[idx]
        
        # Determine peak energy across BOTH sensors to find the gesture window
        frame_energy = np.sum(np.abs(raw), axis=(1, 2, 3, 4))
        peak_frame = int(np.argmax(frame_energy))
        
        # Center the peak in the middle of the 64-frame window (index 32)
        start = peak_frame - 32
        if self.augment:
            # Random jitter +/- 12 frames for temporal translation invariance
            start += np.random.randint(-12, 13)
            
        bins = min(RANGE_BINS, raw.shape[4])
        window = np.zeros((WINDOW_SIZE, 2, 2, 2, bins), dtype=np.complex64)
        
        raw_start = max(0, start)
        raw_end = min(len(raw), start + WINDOW_SIZE)
        
        win_start = max(0, -start)
        win_end = win_start + (raw_end - raw_start)
        
        window[win_start:win_end] = raw[raw_start:raw_end, :, :, :, :bins]
        
        # Exact edge repetition padding
        if win_start > 0:
            window[:win_start] = raw[0, :, :, :, :bins]
        if win_end < WINDOW_SIZE:
            window[win_end:] = raw[-1, :, :, :, :bins]
            
        if self.augment:
            # Independent amplitude scale per sensor (unit phasors are immune;
            # this perturbs the two amplitude channels only).
            scale = np.random.uniform(0.6, 1.6, size=(1, 2, 1, 1, 1))
            window = window * scale

            # Per-channel phase jitter, simulating cable/LO phase drift.
            phase_jitter = np.exp(1j * np.random.normal(0, 0.08, size=(1, 2, 2, 2, 1)))
            window = window * phase_jitter

            # ── Cross-session invariance ──────────────────────────────────────
            # The two augmentations below exist because this dataset contains
            # only 4-5 recording sessions per class. Without them the network
            # latches onto session-specific characteristics -- the exact standoff
            # distance and the pace of that sitting's gestures -- and a per-file
            # validation split cannot detect it, because the split samples the
            # same sessions. Leave-one-burst-out measured 60% while the per-file
            # split reported 94.7%, and a live session scored 58%.
            #
            # Range translation: the hand is never at exactly the recorded
            # standoff distance in a new session.
            window = np.roll(window, np.random.randint(-RANGE_JITTER_BINS,
                                                       RANGE_JITTER_BINS + 1), axis=4)

            # Temporal rescaling: gesture pace varies between sessions and
            # people. Resampling about the window centre keeps the peak centred
            # while stretching or compressing the motion around it.
            rate = np.random.uniform(*TIME_SCALE_RANGE)
            idx = np.clip((np.arange(WINDOW_SIZE) - WINDOW_SIZE // 2) * rate
                          + WINDOW_SIZE // 2, 0, WINDOW_SIZE - 1).astype(int)
            window = window[idx]

            # Thermal noise sized against the MTI residual rather than a fixed
            # absolute level. A constant 0.05 was arbitrary relative to the
            # recording's own scale: it overwhelmed quiet captures and vanished
            # on loud ones, so the effective augmentation strength varied wildly
            # from sample to sample.
            sigma = mti_noise_scale(window, 0.05)
            window = window + complex_gaussian_noise(window.shape, sigma)
            
        # Process each sensor branch individually
        t1 = to_interferometry_tensor(window[:, 0])  # Shape (10, 34, 64)
        t2 = to_interferometry_tensor(window[:, 1])  # Shape (10, 34, 64)
            
        return torch.stack([t1, t2], dim=0), label

# ── MODEL ARCHITECTURE ────────────────────────────────────────────────────────
class GestureCNN_X7_Dual(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        
        def make_branch():
            return nn.Sequential(
                nn.Conv2d(N_FEATURE_CHANNELS, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),
                nn.Dropout2d(0.1),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
                nn.Dropout2d(0.1),
                nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d(2),
                nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten()
            )
            
        # Sensor 1 Branch (Front Array)
        self.branch1 = make_branch()
        # Sensor 2 Branch (Side Array)
        self.branch2 = make_branch()
        
        # Late Fusion Classifier (8192 features per branch = 16,384 total)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 16 * 2, 512), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(512, 128), nn.ReLU(True), nn.Dropout(0.3), 
            nn.Linear(128, n_classes)
        )

    def forward(self, x): 
        # x shape: (Batch, 2, 10, 34, 64)
        x1 = x[:, 0]
        x2 = x[:, 1]
        
        feat1 = self.branch1(x1)
        feat2 = self.branch2(x2)
        
        fused = torch.cat([feat1, feat2], dim=1)
        return self.classifier(fused)

# ── MAIN TRAINING LOOP ────────────────────────────────────────────────────────
def run(args):
    if getattr(args, "seed", None) is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    folders = GESTURE_FOLDERS_6 if args.classes == 6 else GESTURE_FOLDERS_5
    names = GESTURE_NAMES_6 if args.classes == 6 else GESTURE_NAMES_5
    n_classes = len(names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==================================================")
    print(f"Dual-Sensor 3D Gesture Model Training ({n_classes} Classes)")
    print("==================================================")
    print(f"Device             : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Dataset Directory  : {args.data}")
    print(f"Classes ({n_classes})        : {names}")
    print(f"Batch Size         : {args.batch_size}")
    print(f"Epochs             : {args.epochs}")
    print(f"Range Bins         : {RANGE_BINS} x {BIN_LENGTH_M*1000:.1f} mm "
          f"= {max_range(RANGE_BINS):.2f} m max range")
    print(f"Doppler Limit      : +/-{max_unambiguous_velocity(RADAR_FPS):.2f} m/s "
          f"unambiguous @ {RADAR_FPS} FPS")
    print("--------------------------------------------------")

    all_files = enumerate_recordings(args.data, verbose=True, gesture_folders=folders)
    if not all_files:
        print(f"[ERROR] No valid data found in {args.data}.")
        return

    # Deterministic stratified split: 80% train / 20% validation PER CLASS, so
    # every gesture is represented in validation even when class counts differ.
    split_seed = 42 if getattr(args, "seed", None) is None else args.seed
    random.seed(split_seed)
    by_class = {}
    for idx, (_, label) in enumerate(all_files):
        by_class.setdefault(label, []).append(idx)

    train_indices, val_indices = [], []
    for label in sorted(by_class):
        idxs = by_class[label][:]
        random.shuffle(idxs)
        n_val = max(1, int(round(0.20 * len(idxs))))
        val_indices.extend(idxs[:n_val])
        train_indices.extend(idxs[n_val:])

    print(f"[INFO] Split {len(all_files)} recordings -> "
          f"{len(train_indices)} train / {len(val_indices)} val (stratified).")

    train_dataset = X7DualSensorDataset(args.data, augment=True,
                                        file_indices=train_indices, all_files=all_files,
                                        gesture_folders=folders)
    val_dataset = X7DualSensorDataset(args.data, augment=False,
                                      file_indices=val_indices, all_files=all_files,
                                      gesture_folders=folders)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = GestureCNN_X7_Dual(n_classes=n_classes).to(device)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    os.makedirs(args.out, exist_ok=True)
    best_model_path = os.path.join(args.out, args.checkpoint)
    best_acc = 0.0

    print(f"\n[INFO] Starting training for {args.epochs} epochs on {device}...")
    try:
        for ep in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                opt.zero_grad()
                outputs = model(x)
                loss = crit(outputs, y)
                loss.backward()
                opt.step()
                total_loss += loss.item()
            
            scheduler.step()

            # Validation
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)
                    preds = torch.argmax(model(x), dim=1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)

            val_acc = correct / total if total > 0 else 0.0
            avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0

            star = " *" if val_acc > best_acc else ""
            if ep % 5 == 0 or ep == 1 or val_acc > best_acc:
                print(f"Epoch [{ep:03d}/{args.epochs:03d}] - Loss: {avg_loss:.4f} - Val Acc: {val_acc*100:5.2f}%{star}")

            if val_acc > best_acc:
                best_acc = val_acc
                temp_path = best_model_path + ".tmp"
                torch.save(model.state_dict(), temp_path)
                try:
                    if os.path.exists(best_model_path):
                        os.remove(best_model_path)
                except Exception:
                    pass
                try:
                    os.replace(temp_path, best_model_path)
                except Exception:
                    # Fallback if held open by another process
                    torch.save(model.state_dict(), best_model_path + ".new")


    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user.")

    print("==================================================")
    print(f"[INFO] Training Complete! Peak Validation Accuracy: {best_acc*100:.2f}%")
    print(f"[INFO] Saved best weights to: {best_model_path}")
    print("==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D Gesture CNN (5 or 6 Classes)")
    parser.add_argument("--data", default="gesture_dataset_2s_new", help="Path to gesture dataset")
    parser.add_argument("--out", default="models", help="Output directory")
    parser.add_argument("--classes", type=int, default=6, choices=[5, 6],
                        help="Number of classes: 5 (directional) or 6 (includes PINCH, default 6)")
    parser.add_argument("--checkpoint", default="best_model_2s_6class.pth",
                        help="Checkpoint filename (default best_model_2s_6class.pth)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    parser.add_argument("--epochs", type=int, default=80, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()
    run(args)
