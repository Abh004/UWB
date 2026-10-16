"""
test_presence_setup.py
======================
Offline checks that calibration and the setup-change monitor do their job,
using the v3 recordings with simulated setup changes.

    A  radar moved: detection with the OLD calibration vs after RECALIBRATING
    B  auto-alignment: old calibration, moved radar, front row occupied -> the
       monitor must notice and realign the zones
    C  no false alarms: unchanged setup for 10 min must raise no warnings
    D  sensor fault: one sensor 12 dB down -> warning, and never EMPTY

A radar move is simulated by shifting every physical range bin and scaling
amplitude by R^-2 (R^-4 in power). Real moves also change angles and
multipath, so these results are optimistic.

Usage:  python test_presence_setup.py
"""

import glob
import json
import os
import sys

import numpy as np

import x7_presence_zone as Z
import x7_presence_setup as S
from x7_physics import BIN_LENGTH_M, RANGE_OFFSET_M, FIRST_PHYSICAL_BIN as FB

DATA = "human_dataset_v3"
RNG = RANGE_OFFSET_M + np.arange(112) * BIN_LENGTH_M
failures = []


def check(cond, msg):
    print(f"   [{'ok' if cond else 'FAIL'}] {msg}")
    if not cond:
        failures.append(msg)


def move(a, d):
    k = int(round(d / BIN_LENGTH_M))
    b = a.copy()
    phys = np.roll(b[..., FB:], k, axis=-1)
    # a target now at range R was at R - d: amplitude scales by ((R - d) / R)^2
    R_new = np.clip(RNG[FB:], 0.05, None)
    R_old = np.clip(RNG[FB:] - d, 0.05, None)
    b[..., FB:] = phys * (R_old / R_new) ** 2
    return b


def frames(pattern, fn=lambda a: a, limit=None):
    fs = sorted(glob.glob(os.path.join(DATA, pattern)))[:limit]
    return np.concatenate([fn(np.load(f)) for f in fs])


def evaluate(model, calib, fn):
    lay = Z.make_layout(model["n_bins"], Z.zones_for(model, calib))
    trials, _ = Z.load_trials(DATA)
    st = [Z.block_stats(fn(np.load(t["path"])), lay) for t in trials]
    X, Y, _, _ = Z.build_windows(trials, st, model["window_blocks"], lay)
    out = {}
    for i, o in ((0, "occupied"), (1, "front")):
        p = Z.predict_lr(model["outputs"][o], X) >= model["outputs"][o]["threshold"]
        out[o] = (p[Y[:, i] == 1].mean(), p[Y[:, i] == 0].mean())
    return out


def calibrate(model, fn):
    empty = frames("0_people/*", fn, 6)
    seats = {k: frames(f"1_people/*seats{k}_*", fn, 5) for k in (1, 2, 3, 4)}
    cal, _, _ = S.compute_calibration(model, empty, seats, "test")
    return cal


def stream(det, arr):
    last, det.first_align_s = None, None
    for fr in arr:
        st = det.update(fr)
        if st is not None:
            last = st
            if det.align_bins and det.first_align_s is None:
                det.first_align_s = det.now
    return last


def main():
    model = json.load(open(S.MODEL))
    base = calibrate(model, lambda a: a)
    print(f"baseline calibration: shift {base['zone_shift_bins']}")

    print("\nA. radar moved -- old calibration vs recalibrated")
    for d in (0.21, -0.14):
        fn = lambda a, d=d: move(a, d)
        old = evaluate(model, base, fn)
        cal = calibrate(model, fn)
        new = evaluate(model, cal, fn)
        print(f"   radar {d * 100:+.0f} cm: recalibration shift {cal['zone_shift_bins']}")
        for o in ("occupied", "front"):
            print(f"      {o:<9} old calib recall {old[o][0] * 100:5.1f}% FA {old[o][1] * 100:5.1f}%"
                  f"   ->   recalibrated recall {new[o][0] * 100:5.1f}% FA {new[o][1] * 100:5.1f}%")
        # expected shift is relative to TRAINING: this setup's own offset plus the move
        expect = (base["row_peak_m"]["front"] - model["row_ref_m"]["front"] + d) / BIN_LENGTH_M
        # +-1.5 bins: a body is an extended target and R^-4 tilts its range
        # profile toward whichever end got closer, so a large move is over-read
        # by ~7-10 cm. Zones are 0.7 m wide; the detection checks below are
        # the real requirement.
        check(abs(cal["zone_shift_bins"]["front"] - expect) <= 1.5,
              f"calibration measures the {d * 100:+.0f} cm move "
              f"(shift {cal['zone_shift_bins']['front']}, expected {expect:.1f} bins)")
        check(new["front"][0] >= 0.95 and new["front"][1] <= 0.05,
              f"after recalibrating ({d * 100:+.0f} cm) front recall >= 95%, FA <= 5%")

    print("\nB. auto-alignment: old calibration, radar moved +21 cm, front row occupied")
    front = frames("*_people/*seats1*", lambda a: move(a, 0.21))
    det = Z.ZoneDetector(model, calib=base)
    st = stream(det, np.concatenate([front] * 2))
    t = det.first_align_s
    print(f"   aligned {det.align_bins * BIN_LENGTH_M * 100:+.0f} cm, first realignment after "
          f"{t / 60:.1f} min of front-row occupancy" if t else "   never realigned")
    print(f"   warnings {st.get('warnings')}")
    check(abs(det.align_bins - 3) <= 1, f"monitor realigns the zones by ~+3 bins / 21 cm "
                                         f"(got {det.align_bins})")
    check(t is not None and t <= 6 * 60, "realignment within 6 min of front-row occupancy")
    check(any("auto-aligned" in w for w in st.get("warnings", [])), "operator is told to recalibrate")

    print("\nC. unchanged setup, 10 min with the front row occupied -> no warnings")
    det = Z.ZoneDetector(model, calib=base)
    same = frames("*_people/*seats1*")
    reps = int(np.ceil(10 * 60 * Z.FPS / len(same)))
    st = stream(det, np.concatenate([same] * reps))
    print(f"   aligned {det.align_bins} bins, warnings {st.get('warnings')}")
    check(det.align_bins == 0 and not st.get("warnings"), "no false setup-change alarm")

    print("\nD. sensor 1 signal 12 dB down, then the room empties")
    def weak(a):
        b = a.copy(); b[:, 0] *= 0.25; return b
    det = Z.ZoneDetector(model, calib=base)
    st = stream(det, np.concatenate([frames("1_people/*seats1_*", weak), frames("0_people/*", weak)]))
    print(f"   final state: {st['status']}  {st.get('detail', '')}")
    check(any("signal level" in w for w in st.get("warnings", [])), "sensor fault is reported")
    check(st["status"] != "EMPTY", "a faulty sensor never reports EMPTY")

    print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED"))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
