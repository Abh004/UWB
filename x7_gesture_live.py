"""
x7_gesture_live.py
==================
Live gesture recognition with the model from x7_gesture_train.py.

Same trigger, windowing and features as x7_gesture_record.py (x7_gesture_core),
same 40 FPS timing as the old x7_2s_live.py -- so the same speed. A capture is
reported only if the model does not call it NO_GESTURE and it clears the
confidence and margin floors.

Session calibration (default: 2 of each gesture, ~2 min)
---------------------------------------------------------
Accuracy collapses between sessions (where you stand, your pace, the room),
not within one. So at startup the script asks for a few captures of each class,
fine-tunes the ensemble on them mixed with the saved dataset, and only then
starts recognising. Measured leave-one-session-out on gesture_dataset_v3:
    no calibration        53.9% accuracy, precision 52.6% at confidence >= 0.8
    2 per class (~2 min)  67.3% accuracy, precision 75.7% at confidence >= 0.8
    3 per class           69.2% accuracy, precision 79.8% at confidence >= 0.8
The calibration captures are saved to the dataset as a new labelled session,
so the base model improves every time it is retrained.

The fine-tuning runs in a SEPARATE PROCESS. The radar SDK's callback threads
hold the Python GIL while they wait for the next frame, which starves heavy
work in the same interpreter (measured for presence: 1 ms of work took 13 s).
Inference is tiny and unaffected; three fine-tuning passes are not.

    --calibrate 0   skip calibration
    --log DIR       save every live capture (named by the prediction)

Usage:
    python x7_gesture_live.py --host1 529 --host2 545
    python x7_gesture_live.py --host1 529 --host2 545 --calibrate 3 --log gesture_live_log
"""

import os
if hasattr(os, "add_dll_directory") and os.path.isdir(r"C:\RR_02\bin"):
    os.add_dll_directory(r"C:\RR_02\bin")

import sys
import time
import random
import argparse
from pathlib import Path

import numpy as np

import x7_gesture_core as C
from x7_gesture_record import open_radar, report_range, check_lag, keypress

CAL_EPOCHS = 15
CAL_LR = 5e-4
CAL_REPEAT = 8          # calibration captures weighted up against the stored data
CAL_FLOOR = 0.8         # confidence floor after calibration (measured ~76% precision)
CAL_TIMEOUT_S = 240     # give up and keep the base model after this long


def _finetune_worker(meta_path, cal_path, data_dir, out_dir, epochs, lr, repeat):
    """Child process: no radar SDK threads here, so training runs at full speed."""
    import torch
    import x7_gesture_core as C
    import x7_gesture_train as T
    model = C.GestureModel(meta_path)
    z = np.load(cal_path, allow_pickle=False)
    caps = [dict(w=w, label=str(l), session="cal", ts=0) for w, l in zip(z["w"], z["labels"])]
    base = T.load_windows([data_dir]) if os.path.isdir(data_dir) else []
    base = [it for it in base if it["label"] in model.classes and it["session"] != z["session"]]
    rev = model.meta.get("reverse", True)
    train = base + (C.time_reversed(base) if rev else [])
    train += (caps + (C.time_reversed(caps) if rev else [])) * repeat
    for i, m in enumerate(model.models):
        ft = T.fit(train, model.classes, model.channels, epochs, 50 + i, model.device,
                   init=m, lr=lr)
        torch.save(ft.state_dict(), os.path.join(out_dir, f"ft{i}.pt"))


def next_capture(radar, seg, timeout, label=None):
    """Calibration prompt: wait for ENTER (hand at the start position), then up
    to `timeout` s for a captured movement. label=None: automatic 1 s settle."""
    if label is None:
        t_ready = time.time() + 1.0
        while time.time() < t_ready:
            fr = radar.get_frame()
            if fr is not None:
                seg.push(fr)
        seg.reset_capture()
        radar.flush_queue()                          # never judge a gesture on stale frames
    else:
        while C.wait_for_enter(radar, seg, label, keypress) != "enter":
            pass
    t_end = time.time() + timeout
    last_frame, last_show, n = time.time(), 0.0, 0
    while time.time() < t_end:
        fr = radar.get_frame(timeout=0.5) if hasattr(radar, "_proc") else radar.get_frame()
        now = time.time()
        if fr is None:
            if now - last_frame > 2.0:
                sys.stdout.write(f"\r        GO  [radar sends NO FRAMES for {now - last_frame:.0f} s]   ")
                sys.stdout.flush()
            continue
        last_frame = now
        n += 1
        got = seg.push(fr)
        if now - last_show > 0.25:
            sys.stdout.write(f"\r        GO  {seg.meter()}      ")
            sys.stdout.flush()
            last_show = now
        if got is not None:
            return got
    return None


def calibrate(radar, seg, model, args):
    import x7_gesture_train as T
    plan = [g for g in model.classes for _ in range(args.calibrate)]
    for _ in range(100):
        random.shuffle(plan)
        if all(a != b for a, b in zip(plan, plan[1:])):
            break
    print(f"\n[CALIBRATION] {len(plan)} captures: do each prompted movement once after GO, "
          f"the way you do it in live use")
    session = "C" + time.strftime("%m%d%H%M")
    caps, tries = [], {}
    queue = list(plan)
    while queue:
        label = queue.pop(0)
        print(f"  [{len(caps) + 1:2d}/{len(plan)}] {label:<11} {C.DESCRIPTION[label]}")
        print("        GO", end="", flush=True)
        got = next_capture(radar, seg, args.timeout)    # automatic, as live captures it
        if got is None:
            if label == "NO_GESTURE":
                print("  -> nothing triggered (fine)")
                plan.remove(label)
                continue
            tries[label] = tries.get(label, 0) + 1
            print("  -> no movement detected, again" if tries[label] <= 2 else "  -> skipped")
            if tries[label] <= 2:
                queue.insert(0, label)
            continue
        caps.append(dict(w=got["window"], label=label, session=session, ts=int(time.time())))
        print("  -> ok")

    if not caps:
        print("[WARN] no calibration captures -- using the base model")
        return
    # save first, so an interrupted fine-tune never loses the captures
    if not args.no_save_calibration:
        out = Path(args.data)
        out.mkdir(parents=True, exist_ok=True)
        for k, it in enumerate(caps, 1):
            np.save(out / f"{it['label']}_{session}_{it['ts']}_{k:03d}.npy", it["w"])
        print(f"[CALIBRATION] {len(caps)} captures saved to {out}/ as session {session}")

    import multiprocessing as mp
    import tempfile
    import torch
    tmp = tempfile.mkdtemp(prefix="gesture_cal_")
    cal_path = os.path.join(tmp, "cal.npz")
    np.savez(cal_path, w=np.stack([c["w"] for c in caps]),
             labels=np.array([c["label"] for c in caps]), session=session)
    print(f"[CALIBRATION] fine-tuning {len(model.models)} models in a separate process "
          f"(Ctrl+C skips it and keeps the base model) ...", flush=True)
    proc = mp.get_context("spawn").Process(
        target=_finetune_worker,
        args=(args.model, cal_path, args.data, tmp, CAL_EPOCHS, CAL_LR, CAL_REPEAT))
    t0 = time.time()
    proc.start()
    try:
        while proc.is_alive() and time.time() - t0 < CAL_TIMEOUT_S:
            fr = radar.get_frame()             # keep the background model current
            if fr is not None:
                seg.push(fr)
            sys.stdout.write(f"\r    {time.time() - t0:4.0f} s")
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n[CALIBRATION] skipped -- using the base model")
        proc.terminate()
        return
    if proc.is_alive():
        proc.terminate()
        print(f"\n[WARN] fine-tuning took over {CAL_TIMEOUT_S} s -- using the base model")
        return
    files = [os.path.join(tmp, f"ft{i}.pt") for i in range(len(model.models))]
    if proc.exitcode != 0 or not all(os.path.exists(f) for f in files):
        print(f"\n[WARN] fine-tuning failed (exit {proc.exitcode}) -- using the base model")
        return
    for m, f in zip(model.models, files):
        m.load_state_dict(torch.load(f, map_location=model.device, weights_only=True))
        m.eval()
    print(f"\n[CALIBRATION] done in {time.time() - t0:.0f} s")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host1", required=True)
    ap.add_argument("--host2", required=True)
    ap.add_argument("--model", default=os.path.join("models", "gesture_v3.json"))
    ap.add_argument("--calibrate", type=int, default=2,
                    help="captures per class for session calibration (0 = skip)")
    ap.add_argument("--data", default="gesture_dataset_v3",
                    help="stored dataset mixed into calibration, and where it is saved")
    ap.add_argument("--no-save-calibration", dest="no_save_calibration", action="store_true")
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument("--trigger-x", dest="trigger_x", type=float, default=None,
                    help=f"trigger multiple of the idle level (default from the model, {C.TRIGGER_X:g})")
    ap.add_argument("--confidence", type=float, default=None, help="override the confidence floor")
    ap.add_argument("--margin", type=float, default=None, help="override the margin floor")
    ap.add_argument("--log", default=None, help="save every live capture to this folder")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"{args.model} not found -- record with x7_gesture_record.py and "
                         f"train with x7_gesture_train.py first")
    model = C.GestureModel(args.model)
    th = model.meta["thresholds"]
    ev = model.meta["evaluation"]
    print("=" * 66)
    print(f" GESTURE LIVE   model {args.model} ({len(model.models)} models, "
          f"{model.meta['parameters']:,} parameters each, input {'+'.join(model.channels)})")
    print(f" classes {', '.join(model.classes)}")
    print(f" base model, held-out session: accuracy {ev['raw_accuracy'] * 100:.0f}%")
    print("=" * 66)
    log = Path(args.log) if args.log else None
    if log:
        log.mkdir(parents=True, exist_ok=True)

    radar = open_radar(args.host1, args.host2)
    seg = C.Segmenter(args.trigger_x or model.meta.get("trigger_x", C.TRIGGER_X))
    try:
        while seg.state == "WARMUP":
            fr = radar.get_frame()
            if fr is None:
                print("\n[ERROR] no frames -- is the radar connected?")
                return
            seg.push(fr)
            sys.stdout.write(f"\r  {seg.status}      ")
            sys.stdout.flush()
        report_range(radar)
        print(f"\n[INFO] idle level {seg.level:.0f}, trigger {seg.threshold:.0f}")

        if args.calibrate > 0:
            calibrate(radar, seg, model, args)
            th["confidence"] = max(th["confidence"], CAL_FLOOR)
        if args.confidence is not None:
            th["confidence"] = args.confidence
        if args.margin is not None:
            th["margin"] = args.margin
        print(f"\n[READY] floors: confidence {th['confidence']:.2f}, margin {th['margin']:.2f}. "
              f"Perform a gesture.\n")
        seg.reset_capture()

        stalls = 0
        while True:
            fr = radar.get_frame()
            if fr is None:
                stalls += 1
                if stalls in (2, 10) or stalls % 30 == 0:
                    print(f"\n[WARN] no frames for ~{stalls * 2} s -- check the radar")
                continue
            stalls = 0
            if seg.n % 200 == 0:
                check_lag(radar)
            got = seg.push(fr)
            if got is None:
                continue
            t0 = time.perf_counter()
            p = model.probs(got["window"])
            label, conf, why = model.decide(p)
            ms = (time.perf_counter() - t0) * 1000
            if label:
                print(f"  >>> {label:<9} ({conf * 100:4.1f}%)   [{ms:.0f} ms]")
            else:
                print(f"      ignored: {why}")
            if log:
                tag = label or ("NO_GESTURE" if why == "not a gesture" else "UNSURE")
                np.save(log / f"{tag}_live{time.strftime('%m%d')}_{int(time.time())}_"
                              f"{int(conf * 100):03d}.npy", got["window"])
    except KeyboardInterrupt:
        print("\n[INFO] stopped")
    finally:
        radar.close()


if __name__ == "__main__":
    main()
