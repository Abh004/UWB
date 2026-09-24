"""
x7_2s_synthesize_data.py
========================
Generates intra-class synthetic radar dataset from real recordings.
- Splits real data into 80% Train and 20% Holdout Test.
- Synthesizes N new variations per sample using physics-based transformations:
  * Range-bin translation (simulating target distance variations)
  * Temporal phase-continuous shift (simulating breathing phase offset)
  * Complex phase rotation & micro-motion modulation
  * Per-sensor amplitude scaling (simulating body size and angle of arrival)
  * Complex Gaussian noise injection scaled to the MTI residual

Physics note: additive noise is sized relative to the MTI RESIDUAL, not the raw
signal. The raw standard deviation is dominated by static clutter that the MTI
filter removes downstream, so noise scaled to it arrives several times larger
than the surviving moving-target signal and inflates the log-energy features the
presence model consumes.
"""

import os
import argparse
import random
from pathlib import Path
import numpy as np

from x7_physics import mti_noise_scale, complex_gaussian_noise

# Presence capture runs at 10 FPS (see x7_2s_record_presence.RADAR_FPS).
FRAME_RATE_HZ = 10.0
# +/-15 bins at the MEASURED 71.4 mm/bin -> +/-1.07 m of range translation.
MAX_RANGE_SHIFT_BINS = 15


def _reset_dir(path: Path):
    """Create `path`, removing any .npy left over from a previous run."""
    path.mkdir(parents=True, exist_ok=True)
    for stale in path.glob("*.npy"):
        stale.unlink()


def shift_range_bins(raw: np.ndarray, shift: int) -> np.ndarray:
    """
    raw shape: (Frames, Sensors, TX, RX, Bins) -> shift along last axis (Bins)
    """
    if shift == 0:
        return raw.copy()
    shifted = np.zeros_like(raw)
    if shift > 0:
        shifted[..., shift:] = raw[..., :-shift]
    else:
        shifted[..., :shift] = raw[..., -shift:]
    return shifted

def synthesize_sample(raw: np.ndarray, is_empty: bool = False) -> np.ndarray:
    """
    Applies intra-class physical radar transformations.
    """
    out = raw.copy()
    num_frames, num_sensors, num_tx, num_rx, num_bins = out.shape

    if is_empty:
        # For 0_people (empty room):
        # 1. Slight amplitude jitter
        amp_scale = np.random.uniform(0.85, 1.15, size=(1, num_sensors, 1, 1, 1))
        out = out * amp_scale

        # 2. Base complex phase jitter (constant per channel; MTI-invariant)
        phase_jitter = np.exp(1j * np.random.uniform(0, 2 * np.pi, size=(1, num_sensors, num_tx, num_rx, 1)))
        out = out * phase_jitter

        # 3. Thermal noise sized against the MTI residual, not the raw signal
        sigma = mti_noise_scale(out, 0.05)
        out = out + complex_gaussian_noise(out.shape, sigma)
        return out.astype(np.complex64)

    # For occupied rooms (1..N people):
    # 1. Range bin shift (distance variation; BIN_LENGTH_M metres per bin)
    range_shift = random.randint(-MAX_RANGE_SHIFT_BINS, MAX_RANGE_SHIFT_BINS)
    out = shift_range_bins(out, range_shift)

    # 2. Amplitude scaling (body size / clothing reflection). Applied per sensor
    #    only -- a GLOBAL rescale would move the very absolute-energy cue that
    #    separates the occupancy classes, so it is deliberately omitted.
    sensor_imbalance = np.random.uniform(0.88, 1.12, size=(1, num_sensors, 1, 1, 1))
    out = out * sensor_imbalance

    # 3. Complex phase rotation & breathing micro-Doppler modulation.
    #    A random phase OFFSET replaces the previous np.roll: rolling wrapped the
    #    breathing waveform discontinuously across the window edge, injecting a
    #    non-physical step that MTI turns into a broadband transient.
    t_idx = np.arange(num_frames).reshape(num_frames, 1, 1, 1, 1)
    breath_freq = np.random.uniform(0.18, 0.45)   # ~11 to 27 breaths/min
    breath_depth = np.random.uniform(0.05, 0.25)  # radian phase modulation
    breath_phase0 = np.random.uniform(0, 2 * np.pi)
    phase_rot = np.random.uniform(0, 2 * np.pi, size=(1, num_sensors, num_tx, num_rx, 1))

    total_phase = phase_rot + breath_depth * np.sin(
        2 * np.pi * breath_freq * (t_idx / FRAME_RATE_HZ) + breath_phase0
    )
    out = out * np.exp(1j * total_phase)

    # 4. Additive Gaussian noise sized against the MTI residual
    sigma = mti_noise_scale(out, 0.04)
    out = out + complex_gaussian_noise(out.shape, sigma)

    return out.astype(np.complex64)

def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    src_root = Path(args.src)
    holdout_root = Path(args.holdout_out)
    train_root = Path(args.train_out)

    holdout_root.mkdir(parents=True, exist_ok=True)
    train_root.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted([d for d in src_root.iterdir() if d.is_dir() and "people" in d.name])
    if not class_dirs:
        print(f"[ERROR] No class directories found in {src_root}")
        return

    print("==================================================")
    print("Dual-Sensor Intra-Class Synthetic Data Generation")
    print("==================================================")
    print(f"Source Directory   : {src_root}")
    print(f"Holdout Directory  : {holdout_root} (20% real test set)")
    print(f"Augmented Directory: {train_root} (Real train + Synthetic)")
    print(f"Target Samples/Class: {args.target_per_class}")
    print("--------------------------------------------------")

    for cdir in class_dirs:
        cname = cdir.name
        files = []
        rejected = []
        for fp in sorted(cdir.glob("*.npy")):
            try:
                shape = np.load(fp, mmap_mode="r").shape
            except Exception as exc:
                rejected.append((fp.name, f"unreadable: {exc}"))
                continue
            # Expect (Frames, Sensors, TX, RX, Bins)
            if len(shape) != 5 or shape[1] != 2 or shape[2] != 2 or shape[3] != 2:
                rejected.append((fp.name, f"bad shape {shape}, expected (F, 2, 2, 2, B)"))
                continue
            files.append(fp)

        if rejected:
            print(f"\n[WARN] {cname}: rejecting {len(rejected)} malformed recording(s):")
            for name, why in rejected:
                print(f"         - {name}: {why}")

        if not files:
            print(f"[ERROR] {cname}: no valid recordings, skipping class.")
            continue

        random.shuffle(files)

        n_total = len(files)
        n_holdout = max(1, int(n_total * args.holdout_ratio))
        holdout_files = files[:n_holdout]
        train_files = files[n_holdout:]

        print(f"\n[{cname}] Total: {n_total} | Holdout: {len(holdout_files)} | Base Train: {len(train_files)}")

        # 1. Save Holdout Real Files.
        #    Existing files are purged first so repeated runs are idempotent --
        #    previously a re-run only topped up to target_per_class and left
        #    stale samples behind, silently oversampling that class.
        c_holdout_dir = holdout_root / cname
        _reset_dir(c_holdout_dir)
        for hf in holdout_files:
            data = np.load(hf)
            np.save(c_holdout_dir / hf.name, data)

        # 2. Save Real Train Files
        c_train_dir = train_root / cname
        _reset_dir(c_train_dir)
        for tf in train_files:
            data = np.load(tf)
            np.save(c_train_dir / tf.name, data)

        # 3. Generate Synthetic Samples to reach target_per_class
        n_synthetic_needed = max(0, args.target_per_class - len(train_files))
        print(f"  -> Generating {n_synthetic_needed} synthetic samples for {cname}...")

        is_empty = (cname == "0_people")
        for i in range(n_synthetic_needed):
            base_file = random.choice(train_files)
            base_data = np.load(base_file)
            synth_data = synthesize_sample(base_data, is_empty=is_empty)
            synth_filename = f"synth_{i+1:04d}_{base_file.stem}.npy"
            np.save(c_train_dir / synth_filename, synth_data)

        print(f"  -> Completed {cname}: {len(list(c_train_dir.glob('*.npy')))} total training samples saved.")

    print("\n==================================================")
    print("Synthetic dataset generation complete!")
    print("==================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="human_dataset_2s", help="Source directory containing real recordings")
    parser.add_argument("--holdout_out", default="human_dataset_2s_test_holdout", help="Output directory for test holdout")
    parser.add_argument("--train_out", default="human_dataset_2s_augmented", help="Output directory for augmented train set")
    parser.add_argument("--target_per_class", type=int, default=600, help="Target samples per class in train set")
    parser.add_argument("--holdout_ratio", type=float, default=0.20, help="Ratio of real data to hold out for testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    run(args)
