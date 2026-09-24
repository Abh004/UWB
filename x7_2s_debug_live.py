"""
x7_2s_debug_live.py
===================
Diagnostic capture tool.

Streams from both radars exactly as x7_2s_live.py does, but instead of only
printing a prediction it SAVES each captured window to disk and prints the full
probability vector. Run it, perform a few gestures while noting what you did,
then the saved .npy files can be compared directly against the training corpus.

This exists because replaying recorded files cannot reproduce a mismatch that
originates in the live signal itself.

Usage:
    python x7_2s_debug_live.py --host1 529 --host2 545 --label SWIPE_LR
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
from pathlib import Path

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
                         N_CLASSES, GESTURE_NAMES)
import x7_2s_live as L


def summarise_reference(n=40):
    """Statistics of the TRAINING corpus, for side-by-side comparison."""
    import glob
    amps, peaks = [], []
    for f in sorted(glob.glob("gesture_dataset_2s_new/*/*.npy"))[:n]:
        raw = np.load(f)
        amps.append(float(np.abs(raw).mean()))
        peaks.append(float(np.max(np.sum(np.abs(raw), axis=(1, 2, 3, 4)))))
    return float(np.mean(amps)), float(np.mean(peaks))


def main():
    ap = argparse.ArgumentParser(description="Capture and dump live gesture windows")
    ap.add_argument("--host1", required=True)
    ap.add_argument("--host2", required=True)
    ap.add_argument("--model",
                    default=",".join(os.path.join("models", f"m5_now_seed{i}.pth")
                                     for i in range(3)),
                    help="Comma-separated checkpoints; predictions are averaged")
    ap.add_argument("--label", default="UNKNOWN",
                    help="What gesture you are actually performing (for the filename)")
    ap.add_argument("--out", default="debug_captures")
    ap.add_argument("--session", default=None,
                    help="Sitting identifier, e.g. s1, s2, s3. Recorded into the "
                         "filename so training can hold out a WHOLE sitting. "
                         "Sittings were previously inferred from timestamp gaps, "
                         "which mis-grouped two gestures recorded 37s apart. "
                         "Defaults to the current date+hour.")
    ap.add_argument("--threshold", type=float, default=L.MOTION_THRESHOLD)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    session = args.session or time.strftime("%Y%m%d-%H")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_paths = [q for q in args.model.split(",") if q.strip()]
    missing = [q for q in model_paths if not os.path.exists(q)]
    if missing:
        print(f"[ERROR] Model weights not found: {', '.join(missing)}")
        sys.exit(1)
    models = []
    for mp in model_paths:
        m = GestureCNN_X7_Dual(n_classes=N_CLASSES).to(device)
        m.load_state_dict(torch.load(mp, map_location=device))
        m.eval()
        models.append(m)

    ref_amp, ref_peak = summarise_reference()

    radar = X7DualRadar(args.host1, args.host2)
    time.sleep(2.0)
    radar.flush_queue()

    print("\n" + "=" * 66)
    print(" LIVE CAPTURE DIAGNOSTIC")
    print("=" * 66)
    print(f"Saving windows to : {out_dir}/  (label: {args.label}, session: {session})")
    print(f"Motion threshold  : {args.threshold:.0f}")
    print(f"TRAINING reference: mean|frame| = {ref_amp:.3f}, "
          f"peak frame energy = {ref_peak:.0f}")
    print("=" * 66)
    print("\nPerform the gesture named by --label, repeatedly. Ctrl-C to stop.\n")

    buf = collections.deque(maxlen=L.BUFFER_FRAMES)
    seen = 0
    state = "LISTENING"
    peak_e = -1.0
    since = falling = gathered = 0
    cooldown = 0
    armed = True
    n_cap = 0

    try:
        while True:
            frame = radar.get_frame()
            if frame is None:
                continue
            buf.append(frame)
            seen += 1
            if seen < L.BUFFER_FRAMES:
                sys.stdout.write(f"\r[WARMUP] {seen}/{L.BUFFER_FRAMES}  ")
                sys.stdout.flush()
                continue

            e = float(np.sum(np.abs(np.stack(list(buf)[-5:]))))
            if cooldown > 0:
                cooldown -= 1
                continue
            if not armed and e < args.threshold * L.REARM_RATIO:
                armed = True

            if state == "LISTENING":
                if armed and e > args.threshold:
                    state, peak_e, since, falling, gathered = "GATHERING", e, 0, 0, 0
                continue

            gathered += 1
            if e > peak_e:
                peak_e, since, falling = e, 0, 0
            else:
                since += 1
                falling += 1
            if not ((falling >= L.DECAY_FRAMES and since >= L.POST_PEAK_FRAMES)
                    or gathered >= L.MAX_GATHER_FRAMES):
                continue

            arr = np.stack(buf, axis=0)
            pk = len(arr) - 1 - since
            win = L.extract_window(arr, pk)

            x = torch.stack([to_interferometry_tensor(win[:, 0]),
                             to_interferometry_tensor(win[:, 1])],
                            0).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = np.mean([F.softmax(m(x), 1)[0].cpu().numpy()
                                 for m in models], axis=0)

            n_cap += 1
            fn = out_dir / f"{args.label}__{session}__{int(time.time())}_{n_cap:03d}.npy"
            np.save(fn, win)

            energies = np.sum(np.abs(win), axis=(1, 2, 3, 4))
            order = np.argsort(probs)[::-1]
            print(f"\n--- capture {n_cap}  (you performed: {args.label}) ---")
            print(f"  saved            : {fn.name}")
            print("  probabilities    : " + "  ".join(
                f"{GESTURE_NAMES[i]}={probs[i]*100:.1f}%" for i in order))
            print(f"  window mean|.|   : {np.abs(win).mean():.3f}   "
                  f"(training {ref_amp:.3f})")
            print(f"  window peak E    : {energies.max():.0f}   "
                  f"(training {ref_peak:.0f})")
            print(f"  peak at index    : {int(np.argmax(energies))} of {len(win)}"
                  f"   (training centres at 32)")
            print(f"  per-sensor mean  : S1={np.abs(win[:,0]).mean():.3f}  "
                  f"S2={np.abs(win[:,1]).mean():.3f}")

            armed = False
            cooldown = L.COOLDOWN_AFTER_HIT
            state = "LISTENING"

    except KeyboardInterrupt:
        print(f"\n\n[INFO] Stopped. {n_cap} windows saved to {out_dir}/")
        print("Share a few of these files (or the printed stats) to compare "
              "against the training corpus.")
    finally:
        radar.close()


if __name__ == "__main__":
    main()
