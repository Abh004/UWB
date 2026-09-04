"""
x7_2s_train.py
====================
Trains a dual-sensor gesture recognition CNN using Late Fusion 
(two separate feature branches) for improved 3D spatial accuracy.
"""

import argparse
import os
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ── CONFIG ────────────────────────────────────────────────────────────────────
BATCH_SIZE = 16
WINDOW_SIZE = 64
GESTURE_FOLDERS = [
    "G01_SWIPE_LR", "G02_SWIPE_RL", "G03_SWIPE_UD", "G04_SWIPE_DU", "G05_PUSH_IN", "G06_PINCH"
]
GESTURE_NAMES = ["SWIPE_LR", "SWIPE_RL", "SWIPE_UP", "SWIPE_DOWN", "PUSH_IN", "PINCH", "NOISE"]
N_CLASSES = 7

# ── PREPROCESSING ─────────────────────────────────────────────────────────────
def to_interferometry_tensor(w: np.ndarray) -> torch.Tensor:
    # 1. MTI Filter
    w_mti = w - np.mean(w, axis=0, keepdims=True)
    
    tx0_rx0 = w_mti[:, 0, 0, :]
    tx1_rx0 = w_mti[:, 1, 0, :]
    tx0_rx1 = w_mti[:, 0, 1, :]
    tx1_rx1 = w_mti[:, 1, 1, :]
    
    def to_phasor(z1, z2):
        interf = z1 * np.conj(z2)
        return interf / (np.abs(interf) + 1e-9)
        
    # 2. Spatial Unit Phasors (Angle of Arrival)
    H_phasor = to_phasor(tx0_rx0, tx1_rx0)
    V_phasor = to_phasor(tx0_rx0, tx0_rx1)
    
    # 3. Temporal Unit Phasor (Doppler Velocity)
    D_phasor = np.zeros_like(tx0_rx0, dtype=np.complex64)
    D_phasor[1:] = to_phasor(tx0_rx0[1:], tx0_rx0[:-1])
    
    # 4. Amplitude Features
    amp = np.mean(np.abs(w_mti), axis=(1, 2))
    amp_norm = (amp - np.mean(amp)) / (np.std(amp) + 1e-9)
    
    amp_H_diff = np.abs(tx0_rx0) - np.abs(tx1_rx0)
    amp_H_norm = amp_H_diff / (np.abs(tx0_rx0) + np.abs(tx1_rx0) + 1e-9)
    
    # Stack into 8 bounded, physically explicit channels
    stacked = np.stack([
        amp_norm,
        np.real(H_phasor), np.imag(H_phasor),
        np.real(V_phasor), np.imag(V_phasor),
        np.real(D_phasor), np.imag(D_phasor),
        amp_H_norm
    ], axis=0)
    
    # PyTorch expects (Channels, H, W) -> (8, 34, 64)
    norm = np.transpose(stacked, (0, 2, 1))
    return torch.from_numpy(norm).float()

# ── DATASET ───────────────────────────────────────────────────────────────────
class X7DualSensorDataset(Dataset):
    def __init__(self, root: str, augment: bool = False, file_ids: list = None):
        self.augment = augment
        self.samples = []
        root_path = Path(root)
        file_id = 0

        for label_idx, folder in enumerate(GESTURE_FOLDERS):
            class_dir = root_path / folder
            if not class_dir.exists(): continue
            npy_files = sorted(class_dir.rglob("*.npy"))
            for fp in npy_files:
                true_label = label_idx
                if file_ids is not None and file_id not in file_ids:
                    file_id += 1; continue
                try:
                    raw = np.load(fp)
                    # Expecting shape (Frames, 2 Sensors, 2 TX, 2 RX, Bins)
                    if raw.ndim != 5 or raw.shape[1] != 2:
                        print(f"[WARN] Skipping {fp} - not a dual sensor recording. Shape {raw.shape}")
                        continue
                    if raw.shape[0] < WINDOW_SIZE: continue
                    
                    self.samples.append((raw, true_label, "gesture"))
                    if np.random.rand() < 0.20:
                        self.samples.append((raw, 6, "noise")) # Only 20% to prevent bias
                except Exception as e: print(f"[WARN] Error {fp}: {e}")
                file_id += 1
        print(f"[INFO] Loaded {len(self.samples)} dual-sensor recordings.")

    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        raw, label, sample_type = self.samples[idx]
        
        # Determine peak energy across BOTH sensors to find the gesture window
        frame_energy = np.sum(np.abs(raw), axis=(1, 2, 3, 4))
        peak_frame = int(np.argmax(frame_energy))
        
        if sample_type == "noise":
            length = raw.shape[0]
            if peak_frame > length // 2:
                # Peak is in the second half, pick noise from the start
                start = np.random.randint(0, max(1, peak_frame - WINDOW_SIZE))
            else:
                # Peak is in the first half, pick noise from the end
                start = np.random.randint(min(peak_frame + 20, length - 1), max(peak_frame + 21, length))
        else:
            # Center the peak in the middle of the 64-frame window (index 32)
            start = peak_frame - 32
            if self.augment:
                # Shift peak randomly between index 12 and 52 for heavy translation invariance
                start += np.random.randint(-20, 21)
                
        start = max(-WINDOW_SIZE + 1, min(len(raw) - 1, start))
            
        # Extract 64 frames for both sensors
        bins = raw.shape[4]
        if bins > 34: bins = 34
        window = np.zeros((WINDOW_SIZE, 2, 2, 2, bins), dtype=np.complex64)
        
        raw_start = max(0, start)
        raw_end = min(len(raw), start + WINDOW_SIZE)
        
        win_start = max(0, -start)
        win_end = win_start + (raw_end - raw_start)
        
        window[win_start:win_end] = raw[raw_start:raw_end, :, :, :, :bins]
        
        # Pad missing parts
        if win_start > 0:
            window[:win_start] = raw[0, :, :, :, :bins]
        if win_end < WINDOW_SIZE:
            window[win_end:] = raw[-1, :, :, :, :bins]
            
        if self.augment:
            scale = np.random.uniform(0.4, 2.0, size=(1, 2, 1, 1, 1)) # Independent scale per sensor
            window = window * scale
            
            antenna_phase_jitter = np.exp(1j * np.random.normal(0, 0.1, size=(1, 2, 2, 2, 1)))
            window = window * antenna_phase_jitter
            
            if np.random.rand() < 0.15:
                sensor_drop = np.random.randint(0, 2)
                tx_drop = np.random.randint(0, 2)
                rx_drop = np.random.randint(0, 2)
                window[:, sensor_drop, tx_drop, rx_drop, :] = 0.0
            
            noise = (np.random.randn(*window.shape) + 1j * np.random.randn(*window.shape)) * 0.1
            window = window + noise.astype(np.complex64)
            
        # Process each sensor individually
        t1 = to_interferometry_tensor(window[:, 0]) # Shape (8, 34, 64)
        t2 = to_interferometry_tensor(window[:, 1]) # Shape (8, 34, 64)
            
        return torch.stack([t1, t2], dim=0), label

# ── MODEL ─────────────────────────────────────────────────────────────────────
class GestureCNN_X7_Dual(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        
        def make_branch():
            return nn.Sequential(
                nn.Conv2d(8, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),
                nn.Dropout2d(0.1),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
                nn.Dropout2d(0.1),
                nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d(2),
                nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten()
            )
            
        # Sensor 1 Branch (e.g. Front)
        self.branch1 = make_branch()
        # Sensor 2 Branch (e.g. Side)
        self.branch2 = make_branch()
        
        # Late Fusion Classifier (512 * 16 per branch = 8192 * 2 = 16384 total)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 16 * 2, 512), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(512, 128), nn.ReLU(True), nn.Dropout(0.3), 
            nn.Linear(128, n_classes)
        )

    def forward(self, x): 
        # x shape: (Batch, 2, 8, 34, 64)
        x1 = x[:, 0]
        x2 = x[:, 1]
        
        feat1 = self.branch1(x1)
        feat2 = self.branch2(x2)
        
        fused = torch.cat([feat1, feat2], dim=1)
        return self.classifier(fused)

# ── MAIN ──────────────────────────────────────────────────────────────────────
def run(args):
    full_ds = X7DualSensorDataset(args.data)
    if len(full_ds) == 0: 
        print("[ERROR] No data found. Ensure --data points to a directory with dual-sensor .npy files.")
        return

    from torch.utils.data import random_split
    
    dataset = X7DualSensorDataset(args.data, augment=True)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    val_dataset.dataset.augment = False
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GestureCNN_X7_Dual(n_classes=N_CLASSES).to(device)
    crit = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    
    try:
        best_acc = 0.0
        os.makedirs(args.out, exist_ok=True)
        print(f"[INFO] Starting dual-sensor Late Fusion training on {device}...")
        for ep in range(1, args.epochs + 1):
            # Training
            model.train()
            train_dataset.dataset.augment = True
            total_loss = 0
            for x, y in train_loader:
                opt.zero_grad()
                loss = crit(model(x.to(device)), y.to(device))
                loss.backward(); opt.step()
                total_loss += loss.item()
            
            scheduler.step()
                
            # Validation
            model.eval()
            train_dataset.dataset.augment = False 
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    preds = torch.argmax(model(x.to(device)), dim=1)
                    correct += (preds == y.to(device)).sum().item()
                    total += y.size(0)
            
            val_acc = correct / total if total > 0 else 0
            print(f"Epoch {ep:02d} - Loss: {total_loss/len(train_loader):.4f} - Val Acc: {val_acc*100:.1f}%")
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), os.path.join(args.out, "best_model_2s.pth"))
                
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted.")
    print(f"[INFO] Training Complete. Best validation accuracy: {best_acc*100:.1f}%.")
    print(f"[INFO] Saved dual-sensor model to {os.path.join(args.out, 'best_model_2s.pth')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="gesture_dataset_2s_new")
    parser.add_argument("--out", default="models")
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()
    run(args)
