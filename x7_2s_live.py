"""
x7_2s_live.py
=============
Real-Time Dual-Sensor 3D Touchless Gesture Recognition (continuous, 5 classes).

Recognises SWIPE_LR, SWIPE_RL, SWIPE_UP, SWIPE_DOWN and PUSH_IN directly, with
no activation gesture: perform a gesture and it is classified.

There is no PINCH class and no gated mode. PINCH previously served as the
activation trigger but never fired reliably on live hardware, and carrying it in
the label space also cost accuracy on the five directional gestures.

Because the classifier has no NOISE class, ambient motion is rejected by two
independent mechanisms:
  1. An energy gate that must be crossed before anything is captured.
  2. A confidence floor applied to the model's output.
"""

import os
if hasattr(os, "add_dll_directory"):
    _dll = r"C:\RR_02\bin"
    if os.path.isdir(_dll):
        os.add_dll_directory(_dll)

import sys
import time
import argparse
import collections
import numpy as np
import torch
import torch.nn.functional as F

_DEMO_DIR = r"C:\RR_02\novelda-uwb-demos\Demos\RadarDirect\X7RadarDirectCallback"
_DEMO_PARENT = r"C:\RR_02\novelda-uwb-demos\Demos"
for _p in [_DEMO_PARENT, _DEMO_DIR]:
    if _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)

from x7_2s_record import X7DualRadar, RANGE_BINS, RADAR_FPS
from x7_2s_train import (GestureCNN_X7_Dual, to_interferometry_tensor,
                         N_CLASSES, GESTURE_NAMES, GESTURE_NAMES_5, GESTURE_NAMES_6)

WINDOW_SIZE = 64          # frames fed to the model (1.60 s at 40 FPS)
BUFFER_FRAMES = 200       # ring buffer depth; must exceed WINDOW + MAX_GATHER

# ── Motion gate ───────────────────────────────────────────────────────────────
MOTION_THRESHOLD = 800    # 5-frame sum|.| that starts a capture
REARM_RATIO = 0.75        # re-arm once energy falls below this fraction of it

# ── Capture timing (peak-driven, not fixed-length) ────────────────────────────
# The lag between crossing the threshold and the gesture's energy peak is not
# constant: measured across the corpus it ranges from 1 to 97 frames, and its
# mean differs per gesture (SWIPE_UP ~17, PUSH_IN ~34). A fixed gather length
# therefore centres the window correctly for some gestures and not others --
# with a fixed 50 frames, PUSH_IN fell to 2/25 while swipes were fine.
#
# So the capture tracks the running energy maximum and stops only once the
# gesture has demonstrably peaked: the energy has fallen for DECAY_FRAMES
# consecutive frames AND at least POST_PEAK_FRAMES have elapsed since the peak,
# so the second half of the window contains real data rather than edge padding.
DECAY_FRAMES = 6          # consecutive falling frames that confirm the peak
POST_PEAK_FRAMES = 40     # frames required after the peak before classifying
MAX_GATHER_FRAMES = 110   # hard stop, so a noisy plateau cannot stall the capture

# ── Rejection ─────────────────────────────────────────────────────────────────
MIN_CONFIDENCE = 0.45     # 45% threshold across all gestures
MIN_CONFIDENCE_VERTICAL = 0.45
MIN_CONFIDENCE_PINCH = 0.45
VERTICAL_CLASSES = (2, 3)

COOLDOWN_AFTER_HIT = 20   # frames of silence after a reported gesture
COOLDOWN_AFTER_MISS = 8   # frames of silence after a rejected capture


def extract_window(buf_arr: np.ndarray, peak_idx: int) -> np.ndarray:
    """Take the WINDOW_SIZE-frame window centred on peak_idx, clamped to the buffer."""
    start = max(0, min(len(buf_arr) - WINDOW_SIZE, peak_idx - WINDOW_SIZE // 2))
    return buf_arr[start:start + WINDOW_SIZE]


def run_live_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # An ensemble of independently-seeded models. Run-to-run variance on this
    # dataset is large -- identical configs scored 29%, 42% and 52% on the same
    # live captures -- and averaging several seeds is the standard remedy.
    # Measured by leave-one-burst-out, a 3-model ensemble adds ~3 points.
    # A forward pass is well under a millisecond on this GPU, so the cost is
    # irrelevant next to the 40 FPS frame period.
    model_paths = [p for p in args.model.split(",") if p.strip()]
    missing = [p for p in model_paths if not os.path.exists(p)]
    if missing:
        print(f"[ERROR] Model weights not found: {', '.join(missing)}")
        sys.exit(1)

    models = []
    model_classes = None
    for mp in model_paths:
        ckpt = torch.load(mp, map_location=device)
        n_out = ckpt["classifier.6.weight"].shape[0]
        if model_classes is None:
            model_classes = n_out
        elif model_classes != n_out:
            print(f"[ERROR] Inconsistent classes among ensemble checkpoints: {model_classes} vs {n_out}")
            sys.exit(1)
        m = GestureCNN_X7_Dual(n_classes=n_out).to(device)
        m.load_state_dict(ckpt)
        m.eval()
        models.append(m)

    gesture_names = GESTURE_NAMES_6 if model_classes == 6 else GESTURE_NAMES_5

    radar = X7DualRadar(args.host1, args.host2)
    time.sleep(2.0)
    radar.flush_queue()

    motion_threshold = args.threshold
    rearm_threshold = args.threshold * REARM_RATIO

    print("\n" + "=" * 62)
    print(" DUAL-SENSOR 3D TOUCHLESS GESTURE RECOGNITION (CONTINUOUS)")
    print("=" * 62)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device            : {device} ({gpu})")
    print(f"Model             : {len(models)} model(s)")
    for mp in model_paths:
        print(f"                    {mp}")
    print(f"Gestures ({model_classes})      : {', '.join(gesture_names)}")
    print(f"Sensors           : 1 (Host {args.host1}), 2 (Host {args.host2})")
    print(f"Rate / Window     : {RADAR_FPS} FPS | {WINDOW_SIZE} frames "
          f"({WINDOW_SIZE / RADAR_FPS:.2f} s)")
    print(f"Motion threshold  : {motion_threshold:.0f} "
          f"(re-arm below {rearm_threshold:.0f})")
    print("=" * 62)
    print("\n [READY] Perform a gesture -- no activation needed.\n")

    frame_buffer = collections.deque(maxlen=BUFFER_FRAMES)
    frames_seen = 0
    stalls = 0
    state = "LISTENING"
    cooldown = 0
    armed = True

    # Peak-tracking state, valid while state == "GATHERING"
    peak_energy = -1.0
    frames_since_peak = 0
    falling = 0
    gathered = 0

    try:
        while True:
            frame = radar.get_frame()
            if frame is None:
                # get_frame() times out after 2 s. Repeated timeouts mean the two
                # radars are not producing sequence-matched pairs -- check that
                # both host IDs are correct and both units are streaming.
                stalls += 1
                if stalls in (3, 15) or stalls % 30 == 0:
                    print(f"\n[WARN] No synchronised frame pair for ~{stalls * 2}s "
                          f"({frames_seen} frames received so far). Check that both "
                          f"sensors are connected and streaming.")
                continue
            stalls = 0

            frame_buffer.append(frame)
            frames_seen += 1
            if frames_seen < BUFFER_FRAMES:
                sys.stdout.write(f"\r[WARMUP] {frames_seen}/{BUFFER_FRAMES}  ")
                sys.stdout.flush()
                continue

            recent_energy = float(np.sum(np.abs(np.stack(list(frame_buffer)[-5:]))))

            if cooldown > 0:
                cooldown -= 1
                continue

            if not armed and recent_energy < rearm_threshold:
                armed = True

            if state == "LISTENING":
                if armed and recent_energy > motion_threshold:
                    state = "GATHERING"
                    peak_energy = recent_energy
                    frames_since_peak = 0
                    falling = 0
                    gathered = 0
                continue

            # ── GATHERING: follow the energy until the gesture has peaked ──
            gathered += 1
            if recent_energy > peak_energy:
                peak_energy = recent_energy
                frames_since_peak = 0
                falling = 0
            else:
                frames_since_peak += 1
                falling += 1

            peaked = falling >= DECAY_FRAMES and frames_since_peak >= POST_PEAK_FRAMES
            if not (peaked or gathered >= MAX_GATHER_FRAMES):
                continue

            buf_arr = np.stack(frame_buffer, axis=0)

            # Locate the peak by argmax of the PER-FRAME energy over the span
            # belonging to this capture.
            #
            # NOTE: x7_2s_debug_live.py instead uses `len(arr) - 1 - since`, the
            # running maximum of the trailing 5-frame energy. Aligning this file
            # to that rule was tried on 2026-09-10 because the debug tool
            # produced the fine-tuning captures; an offline test predicted
            # +4.3 points, but live accuracy got WORSE and it was reverted. The
            # offline test re-centred already-extracted windows, which injects
            # edge padding and biases the comparison toward the debug rule.
            span_start = max(0, len(buf_arr) - (gathered + WINDOW_SIZE // 2))
            energies = np.sum(np.abs(buf_arr[span_start:]), axis=(1, 2, 3, 4))
            peak_idx = span_start + int(np.argmax(energies))

            window = extract_window(buf_arr, peak_idx)
            x = torch.stack([to_interferometry_tensor(window[:, 0]),
                             to_interferometry_tensor(window[:, 1])],
                            dim=0).unsqueeze(0).to(device)

            with torch.no_grad():
                probs = torch.zeros(model_classes, device=device)
                for m in models:
                    probs += F.softmax(m(x), dim=1)[0]
                probs /= len(models)
            pred = int(torch.argmax(probs))
            conf = float(probs[pred])

            req = 0.45

            if conf >= req:
                print(f"\n{'=' * 56}")
                print(f" >>> {gesture_names[pred]:<12} ({conf * 100:5.1f}%) <<<")
                print(f"{'=' * 56}\n")
                cooldown = COOLDOWN_AFTER_HIT
            else:
                top2 = torch.topk(probs, 2).indices.tolist()
                detail = ", ".join(
                    f"{gesture_names[i]} {probs[i] * 100:.0f}%" for i in top2)
                print(f"  [low confidence] {detail}  (need {req * 100:.0f}%)")
                cooldown = COOLDOWN_AFTER_MISS

            armed = False
            state = "LISTENING"

    except KeyboardInterrupt:
        print("\n\n[INFO] Stopped by user.")
    finally:
        radar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Continuous 5-class dual-sensor gesture inference")
    parser.add_argument("--model",
                        default=",".join(os.path.join("models", f"m5_now_seed{i}.pth")
                                         for i in range(3)),
                        help="Comma-separated checkpoint paths; predictions are "
                             "averaged across them (ensemble)")
    parser.add_argument("--host1", required=True, help="HostInterfaceId for Sensor 1")
    parser.add_argument("--host2", required=True, help="HostInterfaceId for Sensor 2")
    parser.add_argument("--threshold", type=float, default=MOTION_THRESHOLD,
                        help="Motion energy threshold that starts a capture "
                             f"(default {MOTION_THRESHOLD})")
    run_live_inference(parser.parse_args())
