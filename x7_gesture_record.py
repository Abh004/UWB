"""
x7_gesture_record.py
====================
Record gestures at the CURRENT mounting, in a shuffled order, through exactly
the trigger and windowing live detection uses (x7_gesture_core.Segmenter).

Each saved file is one 64-frame window, (64, 2, 2, 2, 34) complex64, named
    <LABEL>_<SESSION>_<unix time>_<n>.npy      e.g. SWIPE_LR_G0929a_1790142334_007.npy

Why shuffled: the old corpus recorded each gesture as one continuous burst, so
every label was tied to one block of time (one position, one pace). Here
gestures alternate in small blocks, and NO_GESTURE captures (non-gesture
movements that happen to trigger) teach the model to stay silent.

Automatic trigger (default): after GO, do the gesture the way you would in
live use; the first movement is captured, exactly as x7_gesture_live.py
captures it. Training data must be captured the way live captures it: on Oct 5
deliberate swipes recorded with the Enter trigger (--manual) turned out to
look like the OPPOSITE swipe of natural, automatically captured ones
(documentation/2026-10-12.md).
--manual: put your hand at the start position, hold still, press ENTER, then
do the gesture (captures only the stroke; does not match live detection).

After each swipe the current model (if one exists) scores the capture; if it
sees the OPPOSITE swipe it warns, so a wrong capture can be redone with x.

Keys during recording:  ENTER = arm the capture (--manual only)
                        x = discard the last capture (and redo it)
                        Ctrl+C = stop and keep everything saved so far

Usage:
    python x7_gesture_record.py --host1 529 --host2 545
    python x7_gesture_record.py --host1 529 --host2 545 --per-gesture 10 --no-gesture 15 --block 2
"""

import os
if hasattr(os, "add_dll_directory") and os.path.isdir(r"C:\RR_02\bin"):
    os.add_dll_directory(r"C:\RR_02\bin")

import sys
import time
import random
import argparse
import datetime
from pathlib import Path

import numpy as np

import x7_gesture_core as C

_DEMO_DIR = r"C:\RR_02\novelda-uwb-demos\Demos\RadarDirect\X7RadarDirectCallback"
_DEMO_PARENT = r"C:\RR_02\novelda-uwb-demos\Demos"
for _p in (_DEMO_PARENT, _DEMO_DIR):
    if _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)


def open_radar(host1, host2):
    """Gesture radar (default_preset.json, 40 FPS) in its OWN PROCESS.

    In-process, the radar SDK's callback threads compete with everything else in
    the interpreter; with the model loaded (live calibration) frame processing
    fell behind, the 10 s frame queue filled, and gestures arrived after the
    capture window had closed ("no movement detected"). Out of process the main
    loop only receives finished frames.
    """
    from x7_acq_worker import RadarProcess, clear_stale_workers
    clear_stale_workers()
    radar = RadarProcess(host1, host2, kind="gesture")
    radar.wait_ready()
    return radar


LAG_WARN_FRAMES = 20        # 0.5 s of unprocessed frames


def check_lag(radar):
    """Warn if frames are piling up (processing slower than the radar)."""
    try:
        n = radar.qsize()
    except Exception:
        return
    if n > LAG_WARN_FRAMES:
        print(f"\n[WARN] {n} frames waiting ({n / C.FPS:.1f} s behind) -- flushing")
        radar.flush_queue()


def report_range(radar):
    info = None
    for _ in range(20):                          # arrives with the first frame
        info = radar.range_info
        if info:
            break
        time.sleep(0.1)
    if not info:
        return
    off, bl, nb = info
    first = int(np.ceil(-off / bl)) if bl > 0 else -1
    print(f"[INFO] gesture preset range map (from the radar): offset {off:+.3f} m, "
          f"bin {bl * 1000:.1f} mm, {nb} bins delivered (34 used)")
    print(f"       first physical bin {first}  ->  bins {first}-33 = "
          f"{off + first * bl:+.2f} .. {off + 33 * bl:+.2f} m")
    if first != int(C.PHYS[0]):
        print(f"[WARN] x7_gesture_core assumes the hand region starts at bin {C.PHYS[0]}; "
              f"this radar says bin {first}. Tell whoever trains on this data.")


def keypress():
    try:
        import msvcrt
        if msvcrt.kbhit():
            return msvcrt.getwch().lower()
    except Exception:
        pass
    return None


def build_plan(per_gesture, no_gesture, block, seed, gestures=None):
    rng = random.Random(seed)
    blocks = []
    for g in (gestures or C.GESTURES[:-1]):
        n = per_gesture
        while n > 0:
            k = min(block, n)
            blocks.append([g] * k)
            n -= k
    n = no_gesture
    while n > 0:
        k = min(block, n)
        blocks.append(["NO_GESTURE"] * k)
        n -= k
    for _ in range(100):                     # avoid the same label twice in a row
        rng.shuffle(blocks)
        if all(a[0] != b[0] for a, b in zip(blocks, blocks[1:])):
            break
    return [g for b in blocks for g in b]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host1", required=True)
    ap.add_argument("--host2", required=True)
    ap.add_argument("--out", default="gesture_dataset_v3")
    ap.add_argument("--session", default=None,
                    help="session id (default: G + date/time); use a new one each sitting")
    ap.add_argument("--per-gesture", dest="per_gesture", type=int, default=10)
    ap.add_argument("--no-gesture", dest="no_gesture", type=int, default=15)
    ap.add_argument("--block", type=int, default=2, help="same gesture this many times in a row")
    ap.add_argument("--trigger-x", dest="trigger_x", type=float, default=C.TRIGGER_X)
    ap.add_argument("--timeout", type=float, default=8.0,
                    help="seconds to wait for a movement before moving on")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--gestures", nargs="+", default=None, choices=C.GESTURES[:-1],
                    help="record only these gestures (default: all five)")
    ap.add_argument("--manual", action="store_true",
                    help="arm each capture with ENTER (captures only the stroke; does NOT match "
                         "how live detection captures, so not for training the live model)")
    ap.add_argument("--auto", action="store_true",
                    help="automatic trigger (the default; kept for older command lines)")
    ap.add_argument("--check-model", dest="check_model",
                    default=os.path.join("models", "gesture_v3.json"),
                    help="model used to flag captures that look like the opposite swipe ('' = off)")
    args = ap.parse_args()
    args.auto = not args.manual

    session = args.session or "G" + datetime.datetime.now().strftime("%m%d%H%M")
    plan = build_plan(args.per_gesture, args.no_gesture, args.block,
                      args.seed if args.seed is not None else time.time(), args.gestures)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print(f" GESTURE RECORDING   session {session}   {len(plan)} captures -> {out}/")
    print(" Stand where you will stand for live use, hand ~0.3-0.5 m from the radar.")
    if args.auto:
        print(" AUTO trigger: do each gesture after GO.")
    else:
        print(" For each prompt: hand to the START position, hold still, press ENTER,")
        print(" then do the gesture once and stop (do not swing back).")
    print(" Press x to discard the last capture and redo it. Ctrl+C stops")
    print(" (everything saved is kept).")
    print("=" * 66)

    checker = None
    if args.check_model and os.path.exists(args.check_model):
        checker = C.GestureModel(args.check_model)
        print(f"[INFO] direction check with {args.check_model}")

    radar = open_radar(args.host1, args.host2)
    seg = C.Segmenter(args.trigger_x)
    saved, missed, ignored, flagged = {}, {}, 0, 0
    last = None                                   # (path, label) of the last capture
    queue = list(plan)
    idx = 0
    try:
        # warm-up: learn background and idle level
        while seg.state == "WARMUP":
            fr = radar.get_frame()
            if fr is None:
                print("\n[ERROR] no frames -- is the radar connected?")
                return
            seg.push(fr)
            sys.stdout.write(f"\r  {seg.status}      ")
            sys.stdout.flush()
        report_range(radar)
        print(f"\n[INFO] idle level {seg.level:.0f}, trigger at {seg.threshold:.0f} "
              f"({args.trigger_x:g}x idle)\n")

        while queue:
            label = queue.pop(0)
            idx += 1
            print(f"[{idx:3d}/{len(plan)}]  {label:<11}  {C.DESCRIPTION[label]}")
            if args.auto:
                # short settle so the previous movement is not captured again
                t_ready = time.time() + 1.0
                while time.time() < t_ready:
                    fr = radar.get_frame()
                    if fr is not None:
                        seg.push(fr)
                seg.reset_capture()
                radar.flush_queue()                  # never judge a gesture on stale frames
            elif C.wait_for_enter(radar, seg, label, keypress) == "x":
                if last is not None:
                    os.remove(last[0])
                    saved[last[1]] -= 1
                    queue.insert(0, label)
                    queue.insert(0, last[1])
                    idx -= 2
                    print(f"\n  -> discarded previous {last[1]}; redoing it")
                    last = None
                else:
                    queue.insert(0, label)
                    idx -= 1
                    print("\n  -> nothing to discard")
                continue
            print("\r        GO" + " " * 60, end="", flush=True)
            t_end = time.time() + args.timeout
            got = None
            last_frame, last_show = time.time(), 0.0
            while time.time() < t_end:
                k = keypress()
                if k == "x" and last is not None:
                    os.remove(last[0])
                    saved[last[1]] -= 1
                    queue.insert(0, label)
                    queue.insert(0, last[1])
                    idx -= 1
                    print(f"  -> discarded previous {last[1]}; redoing it")
                    last = None
                    label = None
                    break
                fr = radar.get_frame(timeout=0.5)
                now = time.time()
                if fr is None:
                    if now - last_frame > 2.0:
                        sys.stdout.write(f"\r        GO  [radar sends NO FRAMES for "
                                         f"{now - last_frame:.0f} s]   ")
                        sys.stdout.flush()
                    continue
                last_frame = now
                got = seg.push(fr)
                if now - last_show > 0.25:
                    sys.stdout.write(f"\r        GO  {seg.meter()}      ")
                    sys.stdout.flush()
                    last_show = now
                if got is not None:
                    break
            if label is None:
                idx -= 1
                continue
            if got is None:
                if label == "NO_GESTURE":
                    ignored += 1
                    print("  -> nothing triggered (already ignored; fine)")
                else:
                    missed[label] = missed.get(label, 0) + 1
                    if missed[label] <= 2:
                        queue.insert(min(len(queue), 2), label)
                    print("  -> no movement detected; will ask again"
                          + ("" if missed[label] <= 2 else " -- skipped after 3 tries"))
                continue
            fn = out / f"{label}_{session}_{int(time.time())}_{idx:03d}.npy"
            np.save(fn, got["window"])
            saved[label] = saved.get(label, 0) + 1
            last = (str(fn), label)
            seen = ""
            if checker is not None:
                p = checker.probs(got["window"])
                k = int(np.argmax(p))
                seen = f", model sees {checker.classes[k]} {p[k]:.2f}"
                if checker.classes[k] == C.REVERSE_OF.get(label) and p[k] >= 0.6:
                    flagged += 1
                    seen += ("\n  !! looks like the OPPOSITE swipe -- press x now to redo it"
                             " (start at the start position, one stroke, no swing back)")
            print(f"\n  -> saved ({got['peak_energy']:.0f}x idle, "
                  f"{got['gathered'] / C.FPS:.1f} s{seen})")
    except KeyboardInterrupt:
        print("\n[INFO] stopped by user")
    finally:
        radar.close()

    print("\n" + "=" * 66)
    print(f" session {session}: saved " + ", ".join(f"{k} {v}" for k, v in sorted(saved.items())))
    if missed:
        print(" gestures that did not trigger: "
              + ", ".join(f"{k} {v}x" for k, v in missed.items())
              + "   (if frequent, lower --trigger-x)")
    print(f" non-gesture movements that never triggered: {ignored}")
    if flagged:
        print(f" {flagged} capture(s) flagged as the opposite swipe (any redone with x are "
              f"already deleted) -- check the rest before training")
    print("=" * 66)


if __name__ == "__main__":
    main()
