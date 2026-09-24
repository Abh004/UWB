import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

from x7_physics import BIN_LENGTH_M, RANGE_OFFSET_M, FIRST_PHYSICAL_BIN, max_range

# ── CONFIG ────────────────────────────────────────────────────────────────────
BATCH_SIZE = 64
# 50 frames. The DURATION depends on the capture frame rate, so it must match
# whatever x7_2s_record_presence used:
#     10 FPS ->  5 s  (1.0-1.7 breath cycles -- too few to establish periodicity)
#      5 FPS -> 10 s  (2.0-3.3 cycles)  <- current recorder default
WINDOW_SIZE = 50

# Number of range bins the X7 baseband actually populates. Recordings on disk
# were saved with a 339-bin declaration, but only the first 112 ever carried
# signal -- the remainder was zero padding. Those empty bins are the dominant
# term in any mean over the bin axis, and because log1p() is steepest at zero
# they also amplify injected noise far more than real returns. Everything is
# therefore cropped to the populated span.
RANGE_BINS = 112
LEGACY_RANGE_BINS = 339

# Bins below FIRST_PHYSICAL_BIN sit at NEGATIVE range (bin 0 is -1.38 m, behind
# the antenna) and cannot hold a target return -- only TX->RX coupling. Measured
# on real captures they carry 4.3% of the MTI energy at 1/31 the amplitude.
# They are dropped because log1p() is steepest near zero and therefore amplifies
# a near-empty bin out of proportion to its information content.
FIRST_BIN = FIRST_PHYSICAL_BIN          # 20 at the measured constants
N_FEATURE_BINS = RANGE_BINS - FIRST_BIN # 92 bins, +0.05 m .. +6.55 m

# ── GPU ACCELERATED PREPROCESSING ─────────────────────────────────────────────
def extract_presence_features(w) -> torch.Tensor:
    """
    Vectorized GPU/CPU feature extraction for the occupancy model.

    w shape : (B, 50, 2, 2, 2, B_bins) or (50, 2, 2, 2, B_bins)
    Returns : (B, 10, 50, 92)         or (10, 50, 92)

    Ten channels, grouped by feature type with both sensors side by side:
        0,1  log reflected power            (occupant COUNT cue)
        2,3  Re horizontal AoA phasor       (TX0/TX1 -> azimuth)
        4,5  Im horizontal AoA phasor
        6,7  Re vertical   AoA phasor       (TX0 RX0/RX1 -> elevation)
        8,9  Im vertical   AoA phasor
    Within each pair the first index is sensor 1, the second sensor 2, so the
    network can read the per-sensor difference (parallax) directly.

    Why the AoA channels exist
    --------------------------
    In a car the seats are in fixed rows, so RANGE separates front from back but
    cannot separate driver from passenger -- they share range bins. Earlier
    revisions averaged over TX and RX, which discards phase and therefore angle,
    leaving total energy as the only cue for same-range occupants. That is
    exactly where the log compression hurts most (4-vs-3 is 2.41x harder than
    2-vs-1).

    Measured on 50 four-occupant recordings, the angular information is real:
        front row AoA coherence 0.568, back row 0.697  (empty far bins 0.429)
        front/back differ by 19 deg
        per-sensor energy ratio 1.13 front vs 1.41 back  (parallax = left/right)

    The 2-element aperture (~57 deg resolution) cannot RESOLVE two people in one
    row, but the phasor centroid and its coherence still shift measurably between
    one and two occupants, which is what these channels expose.

    Physics
    -------
    1. MTI removes the temporally-static clutter, leaving only returns that
       changed within the window -- for a seated person that is chest-wall
       breathing displacement of a few millimetres.
    2. Magnitudes are averaged INCOHERENTLY across the 4 virtual TX/RX channels.
       Coherent averaging would let independent per-channel phase offsets cancel
       real returns; magnitude averaging is phase-blind and so preserves energy.
    3. Power-domain summation across channels then a log. Reflected POWER is what
       scales with the number of scatterers, so summing |x|^2 before compressing
       keeps the "N people reflect ~N times the energy" relationship linear in
       the argument of the log, rather than mixing amplitude and power domains.
    """
    if not isinstance(w, torch.Tensor):
        w = torch.from_numpy(w)

    is_single = (w.dim() == 5)
    if is_single:
        w = w.unsqueeze(0)

    # Crop the zero-padded tail (legacy 339-bin recordings) and the
    # negative-range head, leaving only bins that can physically hold a target.
    if w.shape[-1] > RANGE_BINS:
        w = w[..., :RANGE_BINS]
    w = w[..., FIRST_BIN:]

    # 1. MTI Filter (static clutter removal): subtract the coherent temporal mean.
    w_mti = w - torch.mean(w, dim=1, keepdim=True)

    # 2. Incoherent power average across the 4 virtual channels (2 TX x 2 RX).
    power = (w_mti.real ** 2 + w_mti.imag ** 2)
    power_avg = torch.mean(power, dim=(3, 4))          # (B, 50, 2, bins)

    # 3. Logarithmic compression in the POWER domain.
    energy = torch.log1p(power_avg * 1.0e4).float()    # (B, 50, 2, bins)

    # 4. AoA unit phasors, kept PER SENSOR so angle survives.
    #    H pairs TX0/TX1 on RX0 (transmit baseline -> azimuth)
    #    V pairs TX0 RX0/RX1    (receive  baseline -> elevation)
    t00 = w_mti[:, :, :, 0, 0, :]
    t10 = w_mti[:, :, :, 1, 0, :]
    t01 = w_mti[:, :, :, 0, 1, :]

    def _phasor(a, b):
        z = a * torch.conj(b)
        return z / (torch.abs(z) + 1e-9)

    H = _phasor(t00, t10)                              # (B, 50, 2, bins)
    V = _phasor(t00, t01)

    # (B, Frames, Sensors, Bins) -> (B, Sensors, Frames, Bins) per feature,
    # then concatenate: [energy_s1, energy_s2, Hre_s1, Hre_s2, ...].
    feats = [energy, H.real.float(), H.imag.float(), V.real.float(), V.imag.float()]
    feats = [f.permute(0, 2, 1, 3) for f in feats]     # each (B, 2, 50, bins)
    tensor = torch.cat(feats, dim=1)                   # (B, 10, 50, bins)

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

        rejected = []
        for label_idx, folder in enumerate(self.classes):
            class_dir = root_path / folder
            if not class_dir.exists():
                continue
            npy_files = sorted(class_dir.rglob("*.npy"))
            for fp in npy_files:
                try:
                    raw = np.load(fp)
                    if (raw.ndim != 5 or raw.shape[1] != 2 or raw.shape[2] != 2
                            or raw.shape[3] != 2 or raw.shape[0] < WINDOW_SIZE):
                        rejected.append((fp.name, f"shape {raw.shape}"))
                        continue
                    if raw.shape[4] < RANGE_BINS:
                        rejected.append((fp.name, f"only {raw.shape[4]} range bins"))
                        continue
                    # Crop to the populated span and store as complex64 for
                    # zero-overhead batch collation.
                    tensor_raw = torch.from_numpy(
                        raw[:WINDOW_SIZE, :, :, :, :RANGE_BINS].astype(np.complex64))
                    self.samples.append((tensor_raw, label_idx))
                except Exception as exc:
                    rejected.append((fp.name, str(exc)))

        if rejected:
            print(f"[WARN] Skipped {len(rejected)} malformed recording(s) in '{root}':")
            for name, why in rejected[:10]:
                print(f"         - {name}: {why}")
            if len(rejected) > 10:
                print(f"         ... and {len(rejected) - 10} more")
        print(f"[INFO] Loaded {len(self.samples)} recordings from '{root}'.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── MODEL ARCHITECTURE ────────────────────────────────────────────────────────
N_FEATURE_CHANNELS = 10   # 5 per sensor; see extract_presence_features


class PresenceCNN_X7_Dual(nn.Module):
    def __init__(self, n_classes: int, n_channels: int = N_FEATURE_CHANNELS):
        super().__init__()

        self.features = nn.Sequential(
            # Input: (B, 10, 50, 92)
            nn.Conv2d(n_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (16, 25, 56)

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (32, 12, 28)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (64, 6, 14)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (128, 3, 7)

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
    print(f"Range Bins         : {RANGE_BINS} x {BIN_LENGTH_M*1000:.1f} mm "
          f"= {max_range(RANGE_BINS):.2f} m max range")
    print("--------------------------------------------------")

    train_dataset = X7DualPresenceDataset(args.data)
    if len(train_dataset) == 0:
        print(f"[ERROR] No data found in {args.data}.")
        return

    class_names = train_dataset.classes
    n_classes = len(class_names)
    os.makedirs(args.out, exist_ok=True)
    classes_path = os.path.join(args.out, "presence_classes.json")
    with open(classes_path, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"[INFO] Saved {n_classes} classes to {classes_path}")

    # Model selection must not peek at the reporting holdout. Selecting the
    # checkpoint on the same set that is later quoted as the headline accuracy
    # makes that figure a best-of-N order statistic rather than an unbiased
    # estimate -- with a 100-sample holdout, an early noisy spike routinely wins.
    #
    # So: carve a selection split out of TRAIN, and evaluate the untouched
    # holdout separately for reporting only.
    val_size = max(1, int(0.15 * len(train_dataset)))
    train_size = len(train_dataset) - val_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, sel_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size], generator=generator)
    print(f"[INFO] Model selection split: {train_size} train / {val_size} selection")

    val_dataset = sel_dataset
    holdout_dataset = None
    if args.val_data and os.path.exists(args.val_data):
        print(f"[INFO] Reporting holdout (never used for selection): {args.val_data}")
        holdout_dataset = X7DualPresenceDataset(args.val_data, classes=class_names)

    pin_mem = (device.type == "cuda")
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
        pin_memory=pin_mem
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        pin_memory=pin_mem
    )
    holdout_loader = None
    if holdout_dataset is not None:
        holdout_loader = DataLoader(
            holdout_dataset, batch_size=args.batch_size, shuffle=False,
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
                print(f"Epoch [{ep:03d}/{args.epochs:03d}] - Loss: {avg_loss:.4f} - Selection Acc: {val_acc*100:5.2f}%{star}")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), best_model_path)

    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user.")

    print("==================================================")
    print(f"[INFO] Training complete. Peak SELECTION accuracy: {best_acc*100:.2f}%")
    print(f"[INFO] Saved best weights to: {best_model_path}")

    if holdout_loader is not None and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for raw_x, y in holdout_loader:
                raw_x = raw_x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                preds = torch.argmax(model(extract_presence_features(raw_x)), dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        if total:
            print(f"[INFO] Untouched holdout accuracy: {correct/total*100:.2f}% "
                  f"({correct}/{total}) -- unbiased, not used for checkpoint selection")
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
