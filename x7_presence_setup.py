"""
x7_presence_setup.py
====================
Per-setup calibration and a pre-session self-test for x7_presence_zone.

The zone model is invariant by construction to sensor gain, per-antenna phase
and day-to-day drift (measured Sep 25). What it is NOT invariant to is where the
rows sit relative to the radar: its front/back zones are fixed range gates, and
moving the radar 21 cm farther dropped front-row recall from 98% to 77%.

    calibrate   measures where each row actually is in THIS setup and shifts the
                trained zones to match; also measures the empty-room motion
                floor and the receiver noise reference used by the health monitor.
                ~5 min:  empty 60 s, then one person in each seat for 45 s.

    check       THE DAILY ROUTINE (~3.5 min). One pass -- empty, then one person
                in each seat -- scored live against the current calibration AND
                used to compute a fresh one. Unchanged setup: PASS, calibration
                kept. Moved setup: the fresh calibration is replayed over the
                same recordings, adopted if it passes, the old one backed up.

    selftest    the same steps without recalibrating (~3.5 min), or --full for
                the long version that also times occupied -> EMPTY clearing
                (~8 min). --steps reruns chosen steps, e.g. --steps 3 empty.

Seat numbering is the dataset's: seats 1, 2 = front row, 3, 4 = back row.
The sensors are fixed 15 cm apart for gesture work, so the back row stays a
"Y or ?" indicator and is reported, not required.

Both commands also run from recordings (--from-files DIR) so the procedure
itself can be tested without hardware.

Why not faster: every seat needs one full 30 s analysis window -- the time to
see a still person breathe. Seats cannot be calibrated in pairs: with one front
and one back occupant the back-row estimate locked onto front leakage (0.91 m
instead of 1.53 m), and two people in one row read anywhere from +2 to +21 cm off.

Usage:
    python x7_presence_setup.py check     --host1 529 --host2 545
    python x7_presence_setup.py calibrate --host1 529 --host2 545
    python x7_presence_setup.py selftest  --host1 529 --host2 545
    python x7_presence_setup.py selftest  --host1 529 --host2 545 --all-seats
    python x7_presence_setup.py calibrate --from-files human_dataset_v3
    python x7_presence_setup.py selftest  --from-files human_dataset_v3
"""

import os
import sys
import glob
import json
import time
import argparse
import datetime
import threading

import numpy as np

import x7_presence_zone as Z
from x7_physics import BIN_LENGTH_M

FRONT_SEATS = (1, 2)
BACK_SEATS = (3, 4)
MAX_SHIFT_BINS = 5          # beyond ~36 cm the trained model may not transfer
# Same-setup day-to-day spread of the row breathing peak (v2+v3): front daily
# medians -3.6..+5.2 cm, back -10.5..+1.1 cm. Offsets inside these bands are
# normal variation, not a moved setup, and must not shift the zones.
DEADBAND_FRONT_M = 0.05
SAME_MOVE_M = 0.10          # rows within this of each other moved together
CAL_MARGIN_M = 0.5          # wider peak search during calibration
MODEL = os.path.join("models", "presence_zone.json")
CALIB = os.path.join("models", "presence_calib.json")


# ─── frame sources ──────────────────────────────────────────────────────────

class LiveSource:
    """Radar in its own process; a thread keeps draining it so the detector
    sees a continuous stream even while the operator answers prompts."""

    live = True

    def __init__(self, host1, host2):
        from x7_acq_worker import RadarProcess, clear_stale_workers
        clear_stale_workers()
        self.radar = RadarProcess(host1, host2)
        self.radar.wait_ready()
        self.lock = threading.Lock()
        self.capture = None
        self.sink = None
        self.n = 0
        self.last = None
        self.stop = False
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        t0 = time.time()
        while self.n == 0:
            if time.time() - t0 > 20:
                raise SystemExit("SENSOR FAULT: radar started but sent no frames in 20 s")
            time.sleep(0.2)
        print(f"[INFO] radar streaming after {time.time() - t0:.1f} s")

    def _run(self):
        while not self.stop:
            try:
                fr = self.radar.get_frame(timeout=1.0)
            except Exception as exc:              # acquisition process died
                self.error = str(exc)
                return
            if fr is None:
                continue
            with self.lock:
                self.n += 1
                self.last = time.time()
                if self.capture is not None:
                    self.capture.append(fr)
                sink = self.sink
            if sink is not None:
                try:
                    sink(fr)
                except Exception as exc:
                    self.error = f"detector error: {exc}"
                    return

    def _wait(self, n_frames):
        start = self.n
        while self.n - start < n_frames:
            if self.error:
                raise SystemExit(f"SENSOR FAULT: {self.error}")
            if self.last and time.time() - self.last > 3:
                raise SystemExit("SENSOR FAULT: frames stopped")
            time.sleep(0.1)

    def capture_frames(self, scene, seconds):
        with self.lock:
            self.capture = []
        self._wait(int(seconds * Z.FPS))
        with self.lock:
            fr, self.capture = self.capture, None
        return np.asarray(fr)

    def run(self, scene, seconds):
        self._wait(int(seconds * Z.FPS))

    def record(self, scene, seconds):
        """Capture frames while any sink (detector) keeps processing them."""
        return self.capture_frames(scene, seconds)

    def countdown(self, seconds, msg):
        for c in range(int(seconds), 0, -1):
            sys.stdout.write(f"\r    {msg} {c:2d} s ")
            sys.stdout.flush()
            time.sleep(1.0)
        sys.stdout.write("\r" + " " * 60 + "\r")

    def close(self):
        self.stop = True
        self.radar.close()


class FileSource:
    """Recorded sessions stand in for the room: empty -> 0p files, seat k ->
    1p seats{k}, all -> 4p. For testing the procedure, not the setup."""

    live = False

    def __init__(self, data_dir):
        self.dir = data_dir
        self.sink = None
        self.pos = {}

    def _files(self, scene):
        pat = {"empty": "0_people/*empty*", "all": "4_people/*seats1234*"}.get(scene)
        if pat is None:
            pat = f"1_people/*seats{scene}_*"
        f = sorted(glob.glob(os.path.join(self.dir, pat)))
        if not f:
            raise SystemExit(f"no recordings for scene '{scene}' under {self.dir}")
        return f

    def _frames(self, scene, seconds):
        need = int(seconds * Z.FPS)
        files = self._files(scene)
        i = self.pos.get(scene, 0)
        out = []
        while sum(len(a) for a in out) < need:
            out.append(np.load(files[i % len(files)]))
            i += 1
        self.pos[scene] = i
        return np.concatenate(out)[:need]

    def capture_frames(self, scene, seconds):
        return self._frames(scene, seconds)

    def run(self, scene, seconds):
        for fr in self._frames(scene, seconds):
            self.sink(fr)

    def record(self, scene, seconds):
        fr = self._frames(scene, seconds)
        if self.sink is not None:
            for f in fr:
                self.sink(f)
        return fr

    def countdown(self, seconds, msg):
        pass

    def close(self):
        pass


def ask(src, msg):
    if src.live:
        input(f"\n>>> {msg}\n    press Enter when ready ")
    else:
        print(f"\n[replay] {msg}")


# ─── calibration ────────────────────────────────────────────────────────────

def _blocks(frames):
    return [frames[i * Z.BLOCK:(i + 1) * Z.BLOCK] for i in range(len(frames) // Z.BLOCK)]


def compute_calibration(model, empty, seats, source):
    """empty: (N, 2, 2, 2, bins) frames; seats: {seat: frames}. -> (calib, report)"""
    if "row_ref_m" not in model:
        raise SystemExit("model has no row_ref_m -- retrain with the current "
                         "x7_presence_zone.py")
    zones0 = {z: tuple(v) for z, v in model["zones_m"].items()}
    ref = model["row_ref_m"]
    rep, warn = [], []

    seat_pk = {}
    for k, fr in seats.items():
        row = "front" if k in FRONT_SEATS else "back"
        seat_pk[k] = [Z.row_peak(b, zones0[row], CAL_MARGIN_M) for b in _blocks(fr)]
    row_pk, shift = {}, {}
    for row, ss in (("front", FRONT_SEATS), ("back", BACK_SEATS)):
        v = [p for k in ss if k in seat_pk for p in seat_pk[k]]
        if not v:
            warn.append(f"no {row}-row seat captured: {row} zone left at the training position")
            row_pk[row], shift[row] = ref[row], 0
            continue
        row_pk[row] = float(np.median(v))
    # A moved radar shifts both rows by the same amount; the front row measures
    # it far more precisely (back-row scatter is ~1.4x larger), so the back row
    # follows the front unless the two clearly disagree (chairs moved).
    off = {r: row_pk[r] - ref[r] for r in row_pk}
    if "front" not in shift:
        shift["front"] = (0 if abs(off["front"]) < DEADBAND_FRONT_M
                          else int(round(off["front"] / BIN_LENGTH_M)))
    if "back" not in shift:
        shift["back"] = (shift["front"] if abs(off["back"] - off["front"]) <= SAME_MOVE_M
                         else int(round(off["back"] / BIN_LENGTH_M)))
    rep.append(f"row positions (breathing peak; offsets under {DEADBAND_FRONT_M * 100:.0f} cm "
               f"are normal day-to-day variation):")
    for row in ("front", "back"):
        rep.append(f"   {row:<5}  measured {row_pk[row]:.3f} m   training {ref[row]:.3f} m   "
                   f"-> zone shift {shift[row]:+d} bins ({shift[row] * BIN_LENGTH_M * 100:+.0f} cm)")
        if abs(shift[row]) > MAX_SHIFT_BINS:
            warn.append(f"{row} row is {shift[row] * BIN_LENGTH_M * 100:+.0f} cm from the "
                        f"training geometry: the model may not transfer -- record data here")
    for k in sorted(seat_pk):
        rep.append(f"   seat {k}: {np.median(seat_pk[k]):.3f} m  "
                   f"(spread {np.std(seat_pk[k]) * 100:.0f} cm over {len(seat_pk[k])} blocks)")
    if abs(shift["front"] - shift["back"]) > 2:
        rep.append("   rows shifted by different amounts: chairs moved relative to each "
                   "other, not just the radar")

    calib = {"zone_shift_bins": shift}
    zones = Z.zones_for(model, calib)
    layout = Z.make_layout(model["n_bins"], zones)

    mv = [Z.frame_motion(empty[j - 1], empty[j], layout) for j in range(1, len(empty))]
    thr_m = float(np.percentile(mv, 99.9) + 0.5)
    rep.append(f"motion threshold {thr_m:.2f} from the empty capture "
               f"(model default {model['motion_threshold']:.2f})")
    if thr_m > model["motion_threshold"] + 2.0:
        warn.append("strong motion during the empty capture (someone within ~2 m of the "
                    "radar?): the motion fast path will be less sensitive -- redo with "
                    "everyone at least 2.5 m away")

    lv = [10 * np.log10(Z.block_stats(b, layout)["ref"].reshape(2, 4).mean(1) + Z.EPS)
          for b in _blocks(empty)]
    noise = np.median(lv, axis=0).tolist()
    rep.append(f"receiver reference noise: sensor 1 {noise[0]:.1f} dB, sensor 2 {noise[1]:.1f} dB")

    # validation: the calibrated model on the captures it was calibrated from
    def windows(fr):
        st = [Z.block_stats(b, layout) for b in _blocks(fr)]
        K = model["window_blocks"]
        return [Z.window_features(st[i:i + K], layout) for i in range(len(st) - K + 1)]

    val = {}
    out = model["outputs"]
    thr = {o: (out[o]["threshold"] if out[o].get("usable", True)
               else out[o].get("y_only_threshold", np.inf)) for o in Z.OUTPUTS}
    for scene, fr in [("empty", empty)] + [(f"seat {k}", seats[k]) for k in sorted(seats)]:
        W = windows(fr)
        if not W:
            warn.append(f"{scene}: capture too short to validate")
            continue
        X = np.array(W)
        val[scene] = {o: float(np.mean(Z.predict_lr(out[o], X) >= thr[o])) for o in Z.OUTPUTS}
    rep.append("validation (fraction of windows flagged):")
    rep.append(f"   {'scene':<8}{'occupied':>10}{'front':>8}{'back':>8}")
    for scene, v in val.items():
        rep.append(f"   {scene:<8}{v['occupied'] * 100:9.0f}%{v['front'] * 100:7.0f}%"
                   f"{v['back'] * 100:7.0f}%")
    if "empty" in val and val["empty"]["occupied"] > 0:
        warn.append("the empty room is detected as occupied -- something breathes or moves "
                    "in the seat zones, or the setup differs too much from training")
    for k in seats:
        v = val.get(f"seat {k}")
        if v is None:
            continue
        if v["occupied"] < 1.0:
            warn.append(f"seat {k}: occupant not detected in every window "
                        f"({v['occupied'] * 100:.0f}%)")
        if k in FRONT_SEATS and v["front"] < 0.5:
            warn.append(f"seat {k}: front row not detected")
        if k in BACK_SEATS and v["front"] > 0.5:
            warn.append(f"seat {k}: back-seat occupant shows up as front row")

    calib.update({
        "version": 1,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "source": source, "model_created": model.get("created"),
        "row_peak_m": row_pk, "row_ref_m": ref,
        "seat_peak_m": {str(k): float(np.median(v)) for k, v in seat_pk.items()},
        "zones_m": {z: list(v) for z, v in zones.items()},
        "motion_threshold": thr_m, "noise_ref_db": noise,
        "validation": val, "warnings": warn})
    return calib, rep, warn


def cmd_calibrate(args):
    model = json.load(open(args.model))
    src = FileSource(args.from_files) if args.from_files else LiveSource(args.host1, args.host2)
    try:
        print("=" * 64)
        print(" CALIBRATION  --  seats 1, 2 = front row; 3, 4 = back row")
        print(" The operator stays in ONE place for the whole procedure, at least")
        print(" 2.5 m from the radar (or out of the room). Do not move the radar,")
        print(" chairs or yourself between steps.")
        print("=" * 64)
        ask(src, "EMPTY: everyone out of the seats and at least 2.5 m from the radar.")
        src.countdown(args.leave_s, "leave now")
        print(f"    recording empty room {args.empty_s:.0f} s ...")
        empty = src.capture_frames("empty", args.empty_s)
        seats = {}
        for k in args.seats:
            ask(src, f"SEAT {k}: ONE person sits in seat {k}, normally, and stays still. "
                     f"Everyone else stays away.")
            src.countdown(5, "settle")
            print(f"    recording seat {k} {args.seat_s:.0f} s ...")
            seats[k] = src.capture_frames(k, args.seat_s)
        calib, rep, warn = compute_calibration(
            model, empty, seats,
            f"files:{args.from_files}" if args.from_files else f"live:{args.host1},{args.host2}")
    finally:
        src.close()
    print("\n" + "\n".join(rep))
    if warn:
        print("\nWARNINGS:")
        for w in warn:
            print(f"   ! {w}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"\nsaved {args.out}" + ("" if not warn else "   (with warnings -- read them)"))
    print("next: python x7_presence_setup.py selftest ...")
    return 0 if not warn else 2


# ─── steps shared by selftest and check ────────────────────────────────────

QUICK = {"empty": 45, "seat": 35}          # one 30 s window + a few decisions
FULL = {"empty": 100, "seat": 45}


def _prompt(src, scene, leave_s):
    if scene == "empty":
        ask(src, "EMPTY: everyone out of the seats, at least 2.5 m from the radar.")
        src.countdown(leave_s, "leave now")
    elif scene == "all":
        ask(src, "ALL FOUR SEATS occupied.")
        src.countdown(5, "settle")
    else:
        ask(src, f"SEAT {scene}: exactly ONE person, in seat {scene}. Everyone else away.")
        src.countdown(5, "settle")


def run_steps(src, det, steps, leave_s, verb="observing"):
    """Walk the steps with the detector running on every frame.
    -> [(scene, seconds, frames, [(t, state), ...])]"""
    timeline, lock = [], threading.Lock()

    def sink(fr):
        st = det.update(fr)
        if st is not None:
            with lock:
                timeline.append((det.n, st))

    src.sink = sink
    out = []
    for i, (scene, dur) in enumerate(steps):
        _prompt(src, scene, leave_s)
        if i == 0:
            # A fresh detector needs one full window before it can decide;
            # hold the first scene until then so warm-up is not scored.
            need = (det.frames.maxlen - len(det.frames)) / Z.FPS
            if need > 0:
                print(f"    warming up {need:.0f} s (hold this position) ...")
                src.run(scene, need)
        with lock:
            n0 = det.n
        print(f"    {verb} {dur:.0f} s ...")
        fr = src.record(scene, dur)
        with lock:
            seg = [(round((n - n0) / Z.FPS, 1), st) for n, st in timeline if n > n0]
        out.append((scene, dur, fr, seg))
    src.sink = None
    return out


def replay_steps(model, calib, recorded, window_s, clear_by):
    """Score recorded steps again with another calibration (no radar needed)."""
    det = Z.ZoneDetector(model, calib=calib, auto_align=False)
    res = []
    for scene, dur, fr, _ in recorded:
        n0 = det.n
        seg = []
        for f in fr:
            st = det.update(f)
            if st is not None:
                seg.append((round((det.n - n0) / Z.FPS, 1), st))
        res.append(evaluate_step(scene, seg, clear_by, window_s))
    return res


def evaluate_step(scene, seg, clear_by, window_s):
    """seg: [(t_seconds_since_step_start, state)] -> result dict."""
    notes, ok = [], True
    warns = sorted({w for _, st in seg for w in st.get("warnings", [])})
    if warns:
        ok = False
        notes.append("health: " + " | ".join(warns))
    if not seg:
        return {"scene": str(scene), "pass": False, "notes": notes + ["no decisions"]}
    end = seg[-1][0]
    if scene == "empty":
        first = next((t for t, st in seg if st["status"] == "EMPTY"), None)
        final = [st for t, st in seg if t >= end - max(10.0, min(20.0, end / 3))]
        if first is None or first > clear_by:
            ok = False
            notes.append(f"EMPTY not reached within {clear_by:.0f} s")
        else:
            notes.append(f"EMPTY after {first:.0f} s")
        if not all(st["status"] == "EMPTY" for st in final):
            ok = False
            notes.append("not EMPTY to the end of the step")
        return {"scene": "empty", "pass": ok, "notes": notes}

    occ = [st for t, st in seg if t >= 10.0]              # 10 s to sit down
    miss = sum(1 for st in occ if st["status"] != "OCCUPIED")
    if miss:
        ok = False
        notes.append(f"MISSED OCCUPANT in {miss}/{len(occ)} decisions")
    else:
        notes.append(f"occupied in {len(occ)}/{len(occ)} decisions")
    # rows are judged once the analysis window holds only this step's data
    tail = [st for t, st in seg if t >= min(window_s, end - 5)
            and st["status"] == "OCCUPIED" and not st.get("warming")]
    front = np.mean([st["front"] for st in tail]) if tail else 0.0
    back = np.mean([st["back"] for st in tail]) if tail else 0.0
    want_front = scene == "all" or scene in FRONT_SEATS
    if not tail:
        ok = False
        notes.append("no settled decisions to judge the rows")
    elif want_front and front < 0.9:
        ok = False
        notes.append(f"front row shown on {front * 100:.0f}% (need 90%)")
    elif not want_front and front > 0.1:
        ok = False
        notes.append(f"front row wrongly shown on {front * 100:.0f}%")
    else:
        notes.append(f"front row {'Y' if want_front else '-'} ok")
    if scene == "all" or scene in BACK_SEATS:
        notes.append(f"back row Y on {back * 100:.0f}% (informational)")
    return {"scene": str(scene), "pass": ok, "notes": notes}


def _parse_steps(tokens, dur, all_seats=False, full=False):
    if tokens:
        steps = []
        for t in tokens:
            t = t.lower()
            if t == "empty":
                # after an occupied step, EMPTY takes window + hold (~75 s) by
                # design, so that step needs the long duration
                steps.append((t, FULL["empty"] if steps else dur["empty"]))
            elif t == "all":
                steps.append((t, dur["seat"]))
            elif t in ("1", "2", "3", "4"):
                steps.append((int(t), dur["seat"]))
            else:
                raise SystemExit(f"unknown step '{t}' (use empty, 1, 2, 3, 4, all)")
        return steps
    steps = [("empty", dur["empty"])] + [(k, dur["seat"]) for k in (3, 4, 1, 2)]
    if all_seats:
        steps.append(("all", dur["seat"]))
    if full:
        steps.append(("empty", dur["empty"]))       # times occupied -> EMPTY
    return steps


def _print_results(title, results):
    ok = all(r["pass"] for r in results)
    print("\n" + "=" * 64)
    print(f" {title} {'PASSED' if ok else 'FAILED'}")
    print("=" * 64)
    for r in results:
        print(f"  {'PASS' if r['pass'] else 'FAIL'}  {r['scene']:<8} " + "; ".join(r["notes"]))
    failed = [r["scene"] for r in results if not r["pass"]]
    if failed:
        print(f"\n  rerun just these: python x7_presence_setup.py selftest ... "
              f"--steps {' '.join(failed)}")
    return ok


def _save_report(args, kind, payload):
    os.makedirs(args.report_dir, exist_ok=True)
    path = os.path.join(args.report_dir, f"{kind}_{datetime.datetime.now():%Y%m%d_%H%M%S}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"report: {path}")


# ─── selftest ───────────────────────────────────────────────────────────────

def cmd_selftest(args):
    model = json.load(open(args.model))
    calib = json.load(open(args.calib)) if os.path.exists(args.calib) else None
    if calib is None:
        print(f"[WARN] no calibration at {args.calib}: testing the training geometry")
    det = Z.ZoneDetector(model, calib=calib)
    window = model["window_blocks"] * Z.BLOCK / Z.FPS
    clear_by = window + det.exit_s + 20
    dur = FULL if args.full else QUICK
    steps = _parse_steps(args.steps, dur, args.all_seats, args.full)
    src = FileSource(args.from_files) if args.from_files else LiveSource(args.host1, args.host2)
    print("=" * 64)
    print(f" SELF-TEST ({'full' if args.full else 'quick'})  window {window:.0f} s, "
          f"hold {det.exit_s:.0f} s, steps: " + ", ".join(str(sc) for sc, _ in steps))
    print(" The operator stays in ONE place, at least 2.5 m from the radar.")
    print("=" * 64)
    try:
        rec = run_steps(src, det, steps, args.leave_s)
    finally:
        src.close()
    results = [evaluate_step(sc, seg, clear_by, window) for sc, _, _, seg in rec]
    ok = _print_results("SELF-TEST", results)
    _save_report(args, "selftest", {
        "passed": ok, "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "mode": "full" if args.full else "quick", "model_created": model.get("created"),
        "calibration": args.calib if calib else None,
        "source": args.from_files or f"live:{args.host1},{args.host2}", "steps": results})
    return 0 if ok else 1


# ─── check: selftest + recalibrate-if-moved in one pass ─────────────────────

def cmd_check(args):
    model = json.load(open(args.model))
    old = json.load(open(args.calib)) if os.path.exists(args.calib) else None
    window = model["window_blocks"] * Z.BLOCK / Z.FPS
    det = Z.ZoneDetector(model, calib=old, auto_align=False)
    clear_by = window + det.exit_s + 20
    steps = [("empty", args.empty_s)] + [(k, args.seat_s) for k in (3, 4, 1, 2)]
    src = FileSource(args.from_files) if args.from_files else LiveSource(args.host1, args.host2)
    print("=" * 64)
    print(" CHECK  --  test the current setup and recalibrate only if it moved")
    print(f" steps: empty {args.empty_s:.0f} s, seats 3, 4, 1, 2 at {args.seat_s:.0f} s each")
    print(" The operator stays in ONE place, at least 2.5 m from the radar.")
    print("=" * 64)
    try:
        rec = run_steps(src, det, steps, args.leave_s, verb="recording")
    finally:
        src.close()

    now_res = [evaluate_step(sc, seg, clear_by, window) for sc, _, _, seg in rec]
    empty = rec[0][2]
    seats = {sc: fr for sc, _, fr, _ in rec[1:]}
    new, rep, warn = compute_calibration(
        model, empty, seats,
        f"files:{args.from_files}" if args.from_files else f"live:{args.host1},{args.host2}")
    print("\n" + "\n".join(rep))

    old_shift = (old or {}).get("zone_shift_bins")
    moved = old_shift != new["zone_shift_bins"]
    if old is None:
        print("\nno previous calibration -- using the one just measured")
    elif moved:
        print(f"\nSETUP CHANGED: zone shift {old_shift} -> {new['zone_shift_bins']}")
    else:
        print(f"\nsetup unchanged (zone shift {old_shift})")

    if old is not None and not moved:
        ok = _print_results("CHECK (current calibration)", now_res)
        verdict = "PASS" if ok else "FAIL"
        if not ok:
            print("  The geometry is unchanged, so a failure here is a disturbance or a real\n"
                  "  problem, not a moved setup. Rerun the failed steps.")
        results, adopted = now_res, False
    else:
        if old is not None:
            _print_results("CHECK with the OLD calibration", now_res)
        print("\nreplaying the same recordings with the new calibration ...")
        results = replay_steps(model, new, rec, window, clear_by)
        ok = _print_results("CHECK with the NEW calibration", results) and not warn
        for w in warn:
            print(f"   ! {w}")
        adopted = True
        if old is not None:
            backup = os.path.splitext(args.calib)[0] + "_prev.json"
            with open(backup, "w") as f:
                json.dump(old, f, indent=2)
            print(f"previous calibration kept as {backup}")
        with open(args.calib, "w") as f:
            json.dump(new, f, indent=2)
        print(f"new calibration saved to {args.calib}")
        verdict = ("RECALIBRATED" if ok else "RECALIBRATED -- BUT FAILED") if old else \
                  ("CALIBRATED" if ok else "CALIBRATED -- BUT FAILED")
    print(f"\n>>> {verdict}" + ("   -- go live" if ok else "   -- do not rely on it yet"))
    _save_report(args, "check", {
        "verdict": verdict, "passed": ok, "moved": bool(moved), "adopted": adopted,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "old_shift": old_shift, "new_shift": new["zone_shift_bins"],
        "steps_current": now_res, "steps_final": results, "warnings": warn})
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "calibrate", "selftest"):
        p = sub.add_parser(name)
        p.add_argument("--host1")
        p.add_argument("--host2")
        p.add_argument("--from-files", dest="from_files", default=None,
                       help="run the procedure on recordings instead of the radar")
        p.add_argument("--model", default=MODEL)
        p.add_argument("--leave-s", dest="leave_s", type=float, default=15,
                       help="countdown for everyone to clear the seats")
    k = sub.choices["check"]
    k.add_argument("--empty-s", dest="empty_s", type=float, default=QUICK["empty"])
    k.add_argument("--seat-s", dest="seat_s", type=float, default=QUICK["seat"])
    k.add_argument("--calib", default=CALIB)
    k.add_argument("--report-dir", dest="report_dir", default="selftest_reports")
    c = sub.choices["calibrate"]
    c.add_argument("--empty-s", dest="empty_s", type=float, default=40)
    c.add_argument("--seat-s", dest="seat_s", type=float, default=30)
    c.add_argument("--seats", type=int, nargs="+", default=[1, 2, 3, 4])
    c.add_argument("--out", default=CALIB)
    t = sub.choices["selftest"]
    t.add_argument("--calib", default=CALIB)
    t.add_argument("--full", action="store_true",
                   help="long version: 100 s empties, 45 s seats, and the occupied -> EMPTY "
                        "clearing test at the end")
    t.add_argument("--all-seats", dest="all_seats", action="store_true",
                   help="also test with all four seats occupied (needs four people)")
    t.add_argument("--steps", nargs="+", default=None,
                   help="run only these steps, e.g. --steps 3 empty")
    t.add_argument("--report-dir", dest="report_dir", default="selftest_reports")
    args = ap.parse_args()
    if not args.from_files and not (args.host1 and args.host2):
        ap.error("--host1 and --host2 are required unless --from-files is given")
    sys.exit({"check": cmd_check, "calibrate": cmd_calibrate,
              "selftest": cmd_selftest}[args.cmd](args))


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
