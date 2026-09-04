"""
x7_2s_synthesize_data.py
========================
Generates intra-class synthetic radar dataset from real recordings.
- Splits real data into 80% Train and 20% Holdout Test.
- Synthesizes N new variations per sample using physics-based transformations:
  * Range-bin translation (simulating target distance variations)
  * Temporal cyclic rolling (simulating continuous breathing phase shift)
  * Complex phase rotation & micro-motion modulation
  * Per-sensor amplitude scaling (simulating body size and angle of arrival)
  * Complex Gaussian noise injection
"""

import os
import argparse
import random
from pathlib import Path
import numpy as np

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
        # 1. Temporal rolling
        roll_t = random.randint(0, num_frames - 1)
        out = np.roll(out, roll_t, axis=0)

        # 2. Slight amplitude jitter
        amp_scale = np.random.uniform(0.85, 1.15, size=(1, num_sensors, 1, 1, 1))
        out = out * amp_scale

        # 3. Base complex phase jitter
        phase_jitter = np.exp(1j * np.random.uniform(0, 2 * np.pi, size=(1, num_sensors, num_tx, num_rx, 1)))
        out = out * phase_jitter

        # 4. Thermal noise addition
        noise_level = np.std(out) * 0.05
        noise = (np.random.randn(*out.shape) + 1j * np.random.randn(*out.shape)) * noise_level
        out = out + noise.astype(np.complex64)
        return out

    # For 1_people, 2_people, 3_people:
    # 1. Range bin shift (distance variation +/- 15 bins ~ 10cm steps)
    range_shift = random.randint(-15, 15)
    out = shift_range_bins(out, range_shift)

    # 2. Temporal rolling (continuous breathing cycle shift)
    roll_t = random.randint(0, num_frames - 1)
    out = np.roll(out, roll_t, axis=0)

    # 3. Amplitude scaling (body size / clothing reflection)
    global_scale = np.random.uniform(0.75, 1.35)
    sensor_imbalance = np.random.uniform(0.88, 1.12, size=(1, num_sensors, 1, 1, 1))
    out = out * (global_scale * sensor_imbalance)

    # 4. Complex phase rotation & breathing micro-Doppler modulation
    t_idx = np.arange(num_frames).reshape(num_frames, 1, 1, 1, 1)
    breath_freq = np.random.uniform(0.18, 0.45) # ~11 to 27 breaths/min
    breath_depth = np.random.uniform(0.05, 0.25) # radian phase modulation
    phase_rot = np.random.uniform(0, 2 * np.pi, size=(1, num_sensors, num_tx, num_rx, 1))
    
    total_phase = phase_rot + breath_depth * np.sin(2 * np.pi * breath_freq * (t_idx / 10.0))
    out = out * np.exp(1j * total_phase)

    # 5. Additive Gaussian noise
    noise_level = np.std(out) * 0.04
    noise = (np.random.randn(*out.shape) + 1j * np.random.randn(*out.shape)) * noise_level
    out = out + noise.astype(np.complex64)

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
        files = sorted(list(cdir.glob("*.npy")))
        random.shuffle(files)

        n_total = len(files)
        n_holdout = max(1, int(n_total * args.holdout_ratio))
        holdout_files = files[:n_holdout]
        train_files = files[n_holdout:]

        print(f"\n[{cname}] Total: {n_total} | Holdout: {len(holdout_files)} | Base Train: {len(train_files)}")

        # 1. Save Holdout Real Files
        c_holdout_dir = holdout_root / cname
        c_holdout_dir.mkdir(parents=True, exist_ok=True)
        for hf in holdout_files:
            data = np.load(hf)
            np.save(c_holdout_dir / hf.name, data)

        # 2. Save Real Train Files
        c_train_dir = train_root / cname
        c_train_dir.mkdir(parents=True, exist_ok=True)
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
