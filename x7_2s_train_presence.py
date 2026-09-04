import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
BATCH_SIZE = 64
WINDOW_SIZE = 50  # 50 frames (5 seconds at 10 fps)
RANGE_BINS = 339

# ── GPU ACCELERATED PREPROCESSING ─────────────────────────────────────────────
def extract_presence_features(w) -> torch.Tensor:
    """
    Vectorized GPU/CPU Feature Extraction:
    w shape: (B, 50, 2, 2, 2, 339) or (50, 2, 2, 2, 339)
    Returns: (B, 2, 50, 339) or (2, 50, 339)
    """
    if not isinstance(w, torch.Tensor):
        w = torch.from_numpy(w)
    
    is_single = (w.dim() == 5)
    if is_single:
        w = w.unsqueeze(0)

    # 1. MTI Filter (Static Clutter Removal): subtract mean over temporal frames (dim 1)
    w_mti = w - torch.mean(w, dim=1, keepdim=True)

    # 2. Magnitude & Averaging across 2 TX (dim 3) and 2 RX (dim 4)
    mag = torch.abs(w_mti)
    mag_avg = torch.mean(mag, dim=(3, 4))  # Shape: (B, 50, 2, 339)

    # 3. Log-amplitude scaling (preserves absolute energy differences)
    mag_scaled = torch.log1p(mag_avg * 100.0).float()

    # Shape is (B, Frames, Sensors, Bins). Convert to PyTorch CNN format (B, Sensors, Frames, Bins)
    tensor = mag_scaled.permute(0, 2, 1, 3)

    return tensor.squeeze(0) if is_single else tensor


# ── DATASET ───────────────────────────────────────────────────────────────────
class X7DualPresenceDataset(Dataset):
    def __init__(self, root: str, classes: list = None):
        self.samples = []
        root_path = Path(root)

        # Automatically detect classes if not explicitly provided
        if classes is not None:
            self.classes = classes
        else:
            folders = sorted([f.name for f in root_path.iterdir() if f.is_dir() and "people" in f.name])
            self.classes = folders

        print(f"[INFO] Classes ({len(self.classes)}): {self.classes}")

        for label_idx, folder in enumerate(self.classes):
            class_dir = root_path / folder
            if not class_dir.exists():
                continue
            npy_files = sorted(class_dir.rglob("*.npy"))
            for fp in npy_files:
                try:
                    raw = np.load(fp)
                    if raw.ndim != 5 or raw.shape[1] != 2 or raw.shape[0] < WINDOW_SIZE:
                        continue
                    # Store as complex64 tensor for zero-overhead batch collation
                    tensor_raw = torch.from_numpy(raw[:WINDOW_SIZE].astype(np.complex64))
                    self.samples.append((tensor_raw, label_idx))
                except Exception:
                    pass

        print(f"[INFO] Loaded {len(self.samples)} recordings from '{root}'.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── MODEL ARCHITECTURE ────────────────────────────────────────────────────────
class PresenceCNN_X7_Dual(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            # Input: (B, 2, 50, 339)
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (16, 25, 169)

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (32, 12, 84)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (64, 6, 42)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (128, 3, 21)

            nn.AdaptiveAvgPool2d((1, 1))  # (128, 1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        f = self.features(x)
        return self.classifier(f)


# ── MAIN TRAINING LOOP ────────────────────────────────────────────────────────
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==================================================")
    print("Dual-Sensor Presence Model Training (GPU Accelerated)")
    print("==================================================")
    print(f"Device             : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Train Dataset      : {args.data}")
    print(f"Holdout Val Dataset: {args.val_data}")
    print(f"Batch Size         : {args.batch_size}")
    print(f"Epochs             : {args.epochs}")
    print("--------------------------------------------------")

    train_dataset = X7DualPresenceDataset(args.data)
    if len(train_dataset) == 0:
        print(f"[ERROR] No data found in {args.data}.")
        return

    n_classes = len(train_dataset.classes)
    os.makedirs(args.out, exist_ok=True)
    classes_path = os.path.join(args.out, "presence_classes.json")
    with open(classes_path, "w") as f:
        json.dump(train_dataset.classes, f, indent=2)
    print(f"[INFO] Saved {n_classes} classes to {classes_path}")

    # Holdout validation dataset
    if args.val_data and os.path.exists(args.val_data):
        print(f"[INFO] Using dedicated validation holdout set from: {args.val_data}")
        val_dataset = X7DualPresenceDataset(args.val_data, classes=train_dataset.classes)
    else:
        val_size = max(1, int(0.2 * len(train_dataset)))
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    pin_mem = (device.type == "cuda")
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
        pin_memory=pin_mem
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        pin_memory=pin_mem
    )

    model = PresenceCNN_X7_Dual(n_classes=n_classes).to(device)
    crit = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc = 0.0
    best_model_path = os.path.join(args.out, "best_model_2s_presence.pth")

    print(f"\n[INFO] Starting training for {args.epochs} epochs on {device}...")
    try:
        for ep in range(1, args.epochs + 1):
            # ── Train Epoch ──
            model.train()
            total_loss = 0.0

            for raw_x, y in train_loader:
                raw_x = raw_x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                feat_x = extract_presence_features(raw_x)

                opt.zero_grad()
                outputs = model(feat_x)
                loss = crit(outputs, y)
                loss.backward()
                opt.step()
                total_loss += loss.item()

            scheduler.step()

            # ── Validation Epoch ──
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for raw_x, y in val_loader:
                    raw_x = raw_x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)

                    feat_x = extract_presence_features(raw_x)
                    outputs = model(feat_x)
                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)

            val_acc = correct / total if total > 0 else 0.0
            avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0

            star = " *" if val_acc > best_acc else ""
            if ep % 5 == 0 or ep == 1 or val_acc > best_acc:
                print(f"Epoch [{ep:03d}/{args.epochs:03d}] - Loss: {avg_loss:.4f} - Holdout Val Acc: {val_acc*100:5.2f}%{star}")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), best_model_path)

    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user.")

    print("==================================================")
    print(f"[INFO] Training Complete! Peak Holdout Accuracy: {best_acc*100:.2f}%")
    print(f"[INFO] Saved best weights to: {best_model_path}")
    print("==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Dual-Sensor Presence CNN (GPU-Accelerated)")
    parser.add_argument("--data", default="human_dataset_2s_augmented", help="Training dataset directory")
    parser.add_argument("--val_data", default="human_dataset_2s_test_holdout", help="Validation holdout directory")
    parser.add_argument("--out", default="models", help="Output directory for model weights and metadata")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    args = parser.parse_args()
    run(args)
