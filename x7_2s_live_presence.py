"""
x7_2s_live_presence.py
======================
Real-Time Dual-Sensor Presence & Multi-Person Occupancy Detection
Powered by PyTorch on NVIDIA GPU (CUDA).

Connects to two Novelda X7 UWB Radar sensors, captures real-time 5D complex
MIMO data, runs GPU-accelerated MTI filtering and logarithmic reflection energy
extraction, and infers occupancy count (0, 1, 2, 3 or 4 people).

The top class is exactly 4 people, not "4 or more": the model was trained on
discrete counts and has never seen a 5-person room, so occupancies above 4 are
out of distribution and will be reported as one of the trained classes.

Acquisition runs in a SEPARATE PROCESS (see x7_acq_worker). The Novelda SDK
calls into Python from its own threads and holds the GIL across the blocking
wait for the next frame, which throttled inference here from 1 ms to 13 s --
0.01 cores burned, so the main thread was simply blocked, not busy. Recording
never noticed because it does almost nothing between frames. Use --in-process
to reproduce the old behaviour; it is kept only for diagnosis.
"""

import os
if hasattr(os, "add_dll_directory"):
    dll_dir = r"C:\RR_02\bin"
    if os.path.isdir(dll_dir):
        os.add_dll_directory(dll_dir)

import sys
import time
import json
import argparse
import collections
import numpy as np
import torch
import torch.nn.functional as F

# Ensure callback SDK paths are resolvable
_DEMO_DIR = r"C:\RR_02\novelda-uwb-demos\Demos\RadarDirect\X7RadarDirectCallback"
_DEMO_PARENT = r"C:\RR_02\novelda-uwb-demos\Demos"
for p in [_DEMO_PARENT, _DEMO_DIR]:
    if p not in sys.path and os.path.isdir(p):
        sys.path.insert(0, p)

from x7_2s_record_presence import X7DualRadar, RANGE_BINS, RADAR_FPS
from x7_acq_worker import RadarProcess, clear_stale_workers
from x7_2s_train_presence import PresenceCNN_X7_Dual, extract_presence_features, WINDOW_SIZE
from x7_physics import BIN_LENGTH_M, max_range


def format_class_name(raw_name: str) -> str:
    """Format '0_people' -> '0 People (Empty Room)', '1_people' -> '1 Person', etc.

    Derived from the label itself so that adding an occupancy class needs no
    change here; a hardcoded lookup silently fell through to the raw folder name.
    """
    head = raw_name.split("_")[0]
    if not head.isdigit():
        return raw_name
    n = int(head)
    if n == 0:
        return "0 People (Empty Room)"
    if n == 1:
        return "1 Person"
    return f"{n} People"


def run_live_presence(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 60)
    print(" DUAL-SENSOR REAL-TIME HUMAN OCCUPANCY & PRESENCE DETECTOR")
    print("=" * 60)
    print(f"Device              : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Model Weights       : {args.model}")
    print(f"Window Size         : {WINDOW_SIZE} frames ({WINDOW_SIZE / RADAR_FPS:.1f} seconds @ {RADAR_FPS} FPS)")
    print(f"Range Bins          : {RANGE_BINS} x {BIN_LENGTH_M*1000:.1f} mm "
          f"= {max_range(RANGE_BINS):.2f} m max range")
    print(f"Sensor 1 Host ID    : {args.host1}")
    print(f"Sensor 2 Host ID    : {args.host2}")
    print("-" * 60)

    # Load class labels
    class_names = ["0_people", "1_people", "2_people", "3_people", "4_people"]
    if os.path.exists(args.classes):
        try:
            with open(args.classes, "r") as f:
                class_names = json.load(f)
            print(f"[INFO] Loaded {len(class_names)} classes from {args.classes}")
        except Exception as e:
            print(f"[WARN] Could not parse classes file ({e}). Using default: {class_names}")

    n_classes = len(class_names)

    # Initialize and load model
    if not os.path.exists(args.model):
        print(f"[ERROR] Model weights file '{args.model}' not found!")
        sys.exit(1)

    model = PresenceCNN_X7_Dual(n_classes=n_classes).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f"[INFO] Successfully loaded PyTorch weights on {device}.")

    # A leftover acquisition process keeps the FT4222 locked, and the SDK then
    # reports it as a jumper fault. Check before we add to the confusion.
    clear_stale_workers()

    # Connect to dual radars. Out of process by default -- see module docstring.
    print(f"[INFO] Initializing dual UWB radar stream (Sensor 1: {args.host1}, Sensor 2: {args.host2})...")
    if args.in_process:
        print("[WARN] --in-process: the SDK will hold the GIL and inference will")
        print("       run thousands of times slower. Diagnostic use only.")
        radar = X7DualRadar(args.host1, args.host2)
        time.sleep(2.0)
    else:
        radar = RadarProcess(args.host1, args.host2)
        radar.wait_ready()
        print("[INFO] Acquisition process up; inference has the interpreter to itself.")
        time.sleep(2.0)
    radar.flush_queue()

    print("\n[INFO] Radar stream active. Warming up sliding window buffer...")
    print("=" * 60)

    # Sliding ring buffer for 50 frames
    frame_buffer = collections.deque(maxlen=WINDOW_SIZE)
    voting_history = collections.deque(maxlen=args.smooth_window)
    rate_hist = collections.deque(maxlen=30)
    frame_count = 0

    try:
        while True:
            frame = radar.get_frame()
            if frame is None:
                continue

            # frame shape: (2 Sensors, 2 TX, 2 RX, RANGE_BINS) complex64
            frame_buffer.append(frame)
            frame_count += 1

            if len(frame_buffer) < WINDOW_SIZE:
                sys.stdout.write(f"\r[WARMUP] Buffering radar frames... {len(frame_buffer)}/{WINDOW_SIZE}   ")
                sys.stdout.flush()
                continue

            # Convert sliding window to tensor: shape (50, 2, 2, 2, RANGE_BINS)
            window_np = np.array(frame_buffer, dtype=np.complex64)
            window_tensor = torch.from_numpy(window_np).unsqueeze(0).to(device)  # (1, 50, 2, 2, 2, RANGE_BINS)

            with torch.no_grad():
                # Extract GPU-accelerated MTI & log-power features: (1, 2, 50, RANGE_BINS)
                feat = extract_presence_features(window_tensor)
                logits = model(feat)
                probs = F.softmax(logits, dim=1)[0]
                pred_idx = torch.argmax(probs).item()
                confidence = probs[pred_idx].item()

            # Update temporal smoothing voting window
            voting_history.append(pred_idx)
            most_common_pred = collections.Counter(voting_history).most_common(1)[0][0]

            # Live telemetry. `feat` is already log-compressed power, so its mean
            # is a relative loudness index, not an absolute energy in watts.
            energy_metric = feat.mean().item()

            now = time.perf_counter()
            rate_hist.append(now)
            fps_live = ((len(rate_hist) - 1) / (rate_hist[-1] - rate_hist[0])
                        if len(rate_hist) > 1 and rate_hist[-1] > rate_hist[0] else 0.0)

            raw_label = class_names[most_common_pred]
            display_label = format_class_name(raw_label)

            # Format per-class probability breakdown
            prob_breakdown = " | ".join([f"{c.split('_')[0]}p: {probs[i]*100:4.1f}%" for i, c in enumerate(class_names)])

            sys.stdout.write(
                f"\r[FRAME {frame_count:5d}] "
                f"OCCUPANCY: >> {display_label:<24} << | "
                f"Conf: {confidence*100:5.1f}% | "
                f"Energy: {energy_metric:5.2f} | "
                f"{fps_live:4.1f} fps | "
                f"[{prob_breakdown}]"
            )
            sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\n[INFO] Live inference stopped by user.")
    finally:
        radar.close()


def main():
    parser = argparse.ArgumentParser(description="Live Dual-Sensor Presence & Occupancy Inference")
    parser.add_argument("--host1", required=True, help="HostInterfaceId for Sensor 1 (e.g. 529)")
    parser.add_argument("--host2", required=True, help="HostInterfaceId for Sensor 2 (e.g. 545)")
    parser.add_argument(
        "--model", default=r"models\best_model_2s_presence.pth", help="Path to trained PyTorch presence model"
    )
    parser.add_argument(
        "--classes", default=r"models\presence_classes.json", help="Path to classes JSON metadata"
    )
    parser.add_argument(
        "--smooth_window", type=int, default=5, help="Number of frames for majority voting stabilization"
    )
    parser.add_argument(
        "--in-process", dest="in_process", action="store_true",
        help="Run the SDK in this process (diagnostic only: the SDK holds the "
             "GIL and inference slows by ~4 orders of magnitude)"
    )
    args = parser.parse_args()

    run_live_presence(args)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
