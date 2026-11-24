"""
x7_live_combined.py
===================
Presence and gesture recognition on ONE dual-X7 board, in one window
(x7_gesture_gui.py: cabin panel + gesture animation).

Why the modes alternate instead of running at the same time
-----------------------------------------------------------
The two validated models need incompatible radar presets, and the board runs
one preset at a time (the FT4222 is exclusive-access: one process, one preset):

    presence  human_preset   5 FPS  IterationsPerFrame 3000  112 bins to 6.55 m
    gesture   default_preset 40 FPS IterationsPerFrame 6     34 bins

Presence needs the 3000-pulse integration (~+27 dB) to see a STILL person
breathe; gesture needs 40 FPS. Neither model has ever seen the other preset's
frames. So this script switches the radar between the two:

    PRESENCE --(cabin OCCUPIED)--> GESTURE --(no hand movement for a while)--> PRESENCE

    * presence -> gesture as soon as the cabin is OCCUPIED. Motion counts, so a
      person moving in switches within seconds; a still person takes the 30 s
      breathing window. EMPTY / UNKNOWN keep checking presence: nobody to
      gesture.
    * gesture -> presence after RECHECK_IDLE_S without any captured movement.
      While gestures are being made the cabin is trivially occupied.

Each switch restarts the acquisition process (a few seconds), and every
session starts clean: a new ZoneDetector (its 30 s FFT window must not span the
gap) and a new gesture Segmenter (3 s warm-up to learn the idle level). Only
presence auto-alignment and the gesture calibration carry over.

Gesture session calibration (--calibrate, default 2 per class) runs once, the
first time gesture mode starts -- see x7_gesture_live.py.

    --simulate   no radar: replays human_dataset_v3 sittings (alternating
                 occupied / empty) for presence and gesture_dataset_v4_pause
                 windows for gestures, through the real models and window.
                 Gesture windows are injected directly (the trigger is not
                 simulated). --speed sets the replay speed.

Usage:
    python x7_live_combined.py --host1 529 --host2 545
    python x7_live_combined.py --host1 529 --host2 545 --calibrate 0 --recheck-idle 90
    python x7_live_combined.py --simulate
"""

import os
if hasattr(os, "add_dll_directory") and os.path.isdir(r"C:\RR_02\bin"):
    os.add_dll_directory(r"C:\RR_02\bin")

import sys
import glob
import json
import time
import random
import argparse
import datetime
import collections

import numpy as np

import x7_gesture_core as C
import x7_presence_zone as Z
from x7_gesture_gui import GestureDisplay, NullDisplay, full_name

# --------------------------------------------------------------------------- #
# Switching policy
# --------------------------------------------------------------------------- #
RECHECK_IDLE_S = 60.0     # gesture -> presence after this long without movement
SIM_SITTING_S = 90.0      # --simulate: each recorded sitting lasts this long


def ready_for_gestures(st):
    """Presence -> gesture. st is ZoneDetector.state(). OCCUPIED includes the
    motion-only state during warm-up, so someone moving in switches quickly."""
    return st.get("status") == "OCCUPIED"


def should_recheck_presence(idle_s, session_s):
    """Gesture -> presence.
    idle_s     seconds since the last captured movement (accepted or not) --
               or since gestures became available, if there was none yet
    session_s  seconds since gestures became available in this session

    Every recheck makes gestures unavailable for ~10-40 s (radar restart, then
    motion or the 30 s breathing window, then restart + 3 s warm-up)."""
    return idle_s >= RECHECK_IDLE_S


# --------------------------------------------------------------------------- #
# Radar sources
# --------------------------------------------------------------------------- #
class LiveRadars:
    """Opens the real board in the requested preset, out of process."""
    live = True

    def __init__(self, host1, host2):
        self.host1, self.host2 = host1, host2

    def open(self, kind, attempts=3):
        from x7_acq_worker import RadarProcess, clear_stale_workers
        err = None
        for k in range(attempts):
            # the previous preset's worker must have released the FT4222
            clear_stale_workers(verbose=False)
            radar = RadarProcess(self.host1, self.host2, kind=kind)
            try:
                radar.wait_ready()
                return radar
            except RuntimeError as exc:
                err = exc
                radar.close()
                if k + 1 < attempts:
                    print(f"\n[WARN] {kind} preset did not start ({exc}); retrying")
                    time.sleep(3.0)
        raise RuntimeError(f"could not start the {kind} preset: {err}")

    def segmenter(self, trigger_x):
        return C.Segmenter(trigger_x)


class _SimPresenceRadar:
    def __init__(self, src):
        self.src = src

    def get_frame(self, timeout=1.0):
        time.sleep(1.0 / (Z.FPS * self.src.speed))
        return self.src.next_presence_frame()

    def close(self):
        pass


class _SimGestureRadar:
    def __init__(self, src):
        self.src = src

    def get_frame(self, timeout=2.0):
        time.sleep(1.0 / (C.FPS * self.src.speed))
        self.src.t += 1.0 / C.FPS                           # the cabin lives on
        return np.zeros((2, 2, 2, 34), np.complex64)

    def flush_queue(self):
        pass

    def close(self):
        pass


class _SimSegmenter:
    """Stands in for C.Segmenter: after the warm-up, hands out recorded gesture
    windows every few seconds, then goes quiet so a presence recheck happens."""

    def __init__(self, windows, per_session=4):
        self.windows = windows
        self.n = 0
        self.state = "WARMUP"
        self.level = self.threshold = 1.0
        self.left = per_session
        self.next_at = C.WARMUP_FRAMES + int(C.FPS * random.uniform(3, 6))
        self.truth = None

    @property
    def status(self):
        return f"warming up -- keep still ({(C.WARMUP_FRAMES - self.n) / C.FPS:.0f} s)"

    def push(self, frame):
        self.n += 1
        if self.state == "WARMUP":
            if self.n >= C.WARMUP_FRAMES:
                self.state = "LISTENING"
            return None
        if self.left and self.n >= self.next_at:
            self.left -= 1
            self.next_at = self.n + int(C.FPS * random.uniform(3, 6))
            path = random.choice(self.windows)
            name = os.path.basename(path)
            self.truth = next((g for g in C.GESTURES if name.startswith(g + "_")), name)
            return {"window": np.load(path)}
        return None

    def reset_capture(self, cooldown=0):
        pass

    def meter(self):
        return "simulated"


class SimRadars:
    """Recordings instead of a radar. Presence plays sittings of human_dataset_v3
    as one continuous timeline (occupied, empty, occupied, ...), so time keeps
    passing while gesture mode runs -- like a real cabin."""
    live = False

    def __init__(self, presence_dir, gesture_dir, speed, seed=0):
        self.speed = speed
        random.seed(seed)
        by = collections.defaultdict(list)
        for fp in sorted(glob.glob(os.path.join(presence_dir, "*", "*.npy"))):
            m = Z._NAME.match(os.path.basename(fp))
            if m:
                by[(int(m.group(1)), m.group(2), m.group(3))].append(fp)
        occ = [v for k, v in by.items() if k[0] > 0 and len(v) >= 6]
        emp = [v for k, v in by.items() if k[0] == 0 and len(v) >= 6]
        if not occ or not emp:
            raise SystemExit(f"--simulate needs occupied and empty sittings in {presence_dir}")
        random.shuffle(occ)
        random.shuffle(emp)
        self.playlist = [s for pair in zip(occ, emp) for s in pair]
        self.windows = sorted(glob.glob(os.path.join(gesture_dir, "*.npy")))
        if not self.windows:
            raise SystemExit(f"--simulate needs gesture windows in {gesture_dir}")
        self.i_sit, self.buf = -1, []
        self.t, self.t_end = 0.0, 0.0                       # simulated cabin clock

    def next_presence_frame(self):
        self.t += 1.0 / Z.FPS
        if self.t >= self.t_end or not self.buf:
            self.i_sit = (self.i_sit + 1) % len(self.playlist)
            files = self.playlist[self.i_sit]
            self.buf = list(np.concatenate([np.load(f) for f in files[:18]]))
            self.t_end = self.t + SIM_SITTING_S
            n = int(os.path.basename(files[0])[0])
            print(f"\n[SIM] t={self.t:4.0f} s  cabin now: {n} "
                  f"{'person' if n == 1 else 'people'} "
                  f"({os.path.basename(files[0]).split('_')[1]})")
        fr = self.buf.pop(0)
        self.buf.append(fr)                                 # loop the sitting
        return fr

    def open(self, kind):
        time.sleep(2.0 / self.speed)                        # a preset switch costs time
        return _SimPresenceRadar(self) if kind == "presence" else _SimGestureRadar(self)

    def segmenter(self, trigger_x):
        return _SimSegmenter(self.windows)


# --------------------------------------------------------------------------- #
# The two modes
# --------------------------------------------------------------------------- #
def run_presence(src, model, calib, args, ui, carry):
    """Presence preset until the cabin is OCCUPIED. Returns the last state."""
    ui.mode("switching", "starting the presence preset")
    print(f"\n[{time.strftime('%H:%M:%S')}] -> PRESENCE")
    radar = src.open("presence")
    det = Z.ZoneDetector(model, args.exit_s, calib=calib, auto_align=not args.no_auto_align)
    if carry.get("align_bins"):
        det.align_bins = carry["align_bins"]
        det._build_layout()
    ui.mode("presence", f"{det.K * Z.BLOCK / Z.FPS:.0f} s window, gestures resume when occupied")
    last = time.time()
    try:
        while True:
            fr = radar.get_frame(timeout=1.0)
            if fr is None:
                if time.time() - last > 2.0:
                    st = {"status": "SENSOR FAULT",
                          "detail": f"no frames for {time.time() - last:.0f} s — occupancy unknown"}
                    ui.presence(st)
                    sys.stdout.write(f"\r  SENSOR FAULT: {st['detail']}      ")
                    sys.stdout.flush()
                continue
            last = time.time()
            if fr.shape[-1] != model["n_bins"]:
                raise SystemExit(f"radar sends {fr.shape[-1]} bins, presence model expects "
                                 f"{model['n_bins']}")
            st = det.update(fr)
            if st is None:
                continue
            ui.presence(st)
            sys.stdout.write("\r  " + Z._fmt(st)[:150] + "      ")
            sys.stdout.flush()
            if ready_for_gestures(st):
                return st
    finally:
        carry["align_bins"] = det.align_bins
        radar.close()


def run_gesture(src, model, args, ui, carry):
    """Gesture preset until should_recheck_presence() says otherwise."""
    from x7_gesture_live import calibrate, parse_class_floors, CAL_FLOOR
    from x7_gesture_record import check_lag
    ui.mode("switching", "starting the gesture preset")
    print(f"\n[{time.strftime('%H:%M:%S')}] -> GESTURE")
    radar = src.open("gesture")
    seg = src.segmenter(args.trigger_x or model.meta.get("trigger_x", C.TRIGGER_X))
    th = model.meta["thresholds"]
    try:
        ui.mode("gesture", "warming up — keep still for 3 s")
        while seg.state == "WARMUP":
            fr = radar.get_frame()
            if fr is not None:
                seg.push(fr)
        if not carry.get("calibrated"):
            carry["calibrated"] = True
            if args.calibrate > 0 and src.live:
                ui.mode("gesture", "session calibration")
                calibrate(radar, seg, model, args, ui)
                th["confidence"] = max(th["confidence"], CAL_FLOOR)
            th["class_confidence"] = parse_class_floors(None, model.classes)
        ui.mode("gesture", "hold your hand still for a moment, then swipe")
        ui.status("ready — hold your hand still for a moment, then swipe")
        seg.reset_capture()
        n0 = n_act = seg.n
        stalls = 0
        while True:
            fr = radar.get_frame()
            if fr is None:
                stalls += 1
                if stalls in (2, 10) or stalls % 30 == 0:
                    ui.status(f"no radar frames for ~{stalls * 2} s — check the radar")
                continue
            stalls = 0
            if src.live and seg.n % 200 == 0:
                check_lag(radar)
            got = seg.push(fr)
            if got is not None:
                n_act = seg.n
                label, conf, why = model.decide(model.probs(got["window"]))
                truth = getattr(seg, "truth", None)
                tag = f"   (recorded as {truth})" if truth else ""
                if label:
                    print(f"\n  >>> {full_name(label):<14} ({conf * 100:4.1f}%){tag}")
                    ui.detected(label, conf)
                else:
                    print(f"\n      ignored: {why}{tag}")
                    ui.ignored(why, conf)
            # frame counts, not wall time: the same in --simulate at any --speed
            if should_recheck_presence((seg.n - n_act) / C.FPS, (seg.n - n0) / C.FPS):
                print(f"\n[INFO] no movement for {(seg.n - n_act) / C.FPS:.0f} s "
                      f"-- rechecking presence")
                return
    finally:
        radar.close()


def main():
    global RECHECK_IDLE_S
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host1")
    ap.add_argument("--host2")
    ap.add_argument("--model", default=os.path.join("models", "gesture_v4_pause.json"),
                    help="gesture model")
    ap.add_argument("--presence-model", dest="presence_model",
                    default=os.path.join("models", "presence_zone.json"))
    ap.add_argument("--calib", default=os.path.join("models", "presence_calib.json"),
                    help="presence calibration from x7_presence_setup.py")
    ap.add_argument("--no-auto-align", dest="no_auto_align", action="store_true")
    ap.add_argument("--exit-s", dest="exit_s", type=float, default=None,
                    help="presence hold time (default from the model)")
    ap.add_argument("--recheck-idle", dest="recheck_idle", type=float, default=RECHECK_IDLE_S,
                    help="gesture -> presence after this many seconds without movement")
    ap.add_argument("--start", choices=("presence", "gesture"), default="presence")
    ap.add_argument("--calibrate", type=int, default=2,
                    help="gesture captures per class for session calibration (0 = skip)")
    ap.add_argument("--data", default="gesture_dataset_v4_pause",
                    help="gesture dataset mixed into calibration, and where it is saved")
    ap.add_argument("--no-save-calibration", dest="no_save_calibration", action="store_true")
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument("--trigger-x", dest="trigger_x", type=float, default=None)
    ap.add_argument("--no-gui", dest="no_gui", action="store_true")
    ap.add_argument("--simulate", action="store_true", help="replay recordings, no radar")
    ap.add_argument("--speed", type=float, default=4.0, help="--simulate replay speed")
    ap.add_argument("--sim-presence", dest="sim_presence", default="human_dataset_v3")
    args = ap.parse_args()

    RECHECK_IDLE_S = args.recheck_idle
    if not args.simulate and not (args.host1 and args.host2):
        raise SystemExit("--host1 and --host2 are required (or use --simulate)")

    gmodel = C.GestureModel(args.model)
    pmodel = json.load(open(args.presence_model))
    calib = None
    if args.simulate:
        print("[SIM] presence uses the training geometry (recordings predate the current "
              "calibration)")
    elif args.calib and os.path.exists(args.calib):
        calib = json.load(open(args.calib))
        age = (datetime.datetime.now()
               - datetime.datetime.fromisoformat(calib["created"])).total_seconds() / 3600
        print(f"[INFO] presence calibration {args.calib} ({age:.1f} h old)")
    else:
        print("[WARN] no presence calibration -- using the training geometry. Run\n"
              "       python x7_presence_setup.py calibrate --host1 .. --host2 ..")

    print("=" * 66)
    print(f" PRESENCE + GESTURE   gesture {args.model}, presence {args.presence_model}")
    print(f" presence -> gesture when OCCUPIED; gesture -> presence after "
          f"{RECHECK_IDLE_S:.0f} s without movement")
    print("=" * 66)

    src = (SimRadars(args.sim_presence, args.data, args.speed) if args.simulate
           else LiveRadars(args.host1, args.host2))
    ui = NullDisplay() if args.no_gui else GestureDisplay("X7 Presence + Gesture")
    carry = {}
    mode = args.start
    try:
        while True:
            if mode == "presence":
                run_presence(src, pmodel, calib, args, ui, carry)
                mode = "gesture"
            else:
                run_gesture(src, gmodel, args, ui, carry)
                mode = "presence"
    except KeyboardInterrupt:
        print("\n[INFO] stopped")
    finally:
        ui.close()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
