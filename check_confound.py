"""
check_confound.py
=================
Gate a presence dataset BEFORE training on it.

Why this exists
---------------
The v2 dataset recorded one occupancy level per sitting: 4p on 09-11, 3p on
09-15, 2p on 09-16, 1p on 09-21. Occupancy and session are therefore perfectly
aliased, and a model trained on it scored 99.7% against a 40% baseline -- a
number that cannot be interpreted, because nothing in the data distinguishes
"counts people" from "recognises the day".

A caveat that matters, and that a naive reading of this script would miss:
range bins beyond the seating area are NOT empty. v2 was recorded in a ROOM
with car-like seating, not in a car, and that room has static structure at
4.26 m and 5.33 m -- walls or furniture. Moving energy past 2.5 m is 14-21% of
the total at every occupancy level and does NOT scale with occupant count
(1p 16.7%, 2p 20.8%, 3p 14.5%, 4p 16.7%), so it is mostly environment and
residual clutter rather than an occupancy cue.

Two further facts keep the far-region result ambiguous rather than damning:
far-region energy is flat against elapsed time inside a sitting (|r| <= 0.11,
all n.s.), so it is not equipment drift; and bins behind the antenna, which
see fixed TX->RX coupling, vary 10x LESS between days than between consecutive
trials on the same day, so the instrument is stable. Above-chance prediction
from non-seating bins is therefore expected and is not by itself proof of a
confound.

The hard gate is therefore test 1. Test 2 is diagnostic context, not a verdict.

  1. SITTING BALANCE (GATE). Every occupancy level must appear in more than
     one sitting. If a level lives in exactly one sitting, label and session
     are aliased and no train/test split can separate them.

  2. REGION PREDICTABILITY (diagnostic). How well do non-seating regions
     predict the label? In a cabin, expect some signal from multipath. Read it
     alongside test 3.

  3. WITHIN-SITTING TIME DRIFT (diagnostic). Correlation between elapsed time
     in a sitting and far-region energy. Strong correlation means the hardware
     is drifting and the block structure of the recording will leak; near zero
     means what test 2 sees is more likely real multipath.

Usage:
    python check_confound.py                       # human_dataset_v2
    python check_confound.py --data human_dataset_v2 human_dataset_v3
"""

import os
import re
import glob
import argparse
from collections import Counter, defaultdict

import numpy as np

from x7_physics import BIN_LENGTH_M, RANGE_OFFSET_M, FIRST_PHYSICAL_BIN

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
except ImportError:
    raise SystemExit("needs scikit-learn:  pip install scikit-learn")


# Seating geometry. Front row ~0.5-0.9 m, back row ~1.3-1.9 m. Nothing past
# 2.5 m is a seat, and bins below FIRST_PHYSICAL_BIN are behind the antenna.
SEATING_MAX_M = 2.5


def parse(path):
    """-> (n_people, scenario_tag, sitting_id, timestamp) or None."""
    b = os.path.basename(path)
    m = re.match(r"(\d+)people_([A-Za-z0-9]+)_(.+?)_(\d{10})_", b)
    if not m:
        return None
    n, tag, subj, ts = m.groups()
    # Recorder writes "S01-09211425"; older files have no sitting id, so fall
    # back to the calendar day, which is what the v2 sittings actually were.
    sit = subj.split("-", 1)[1] if "-" in subj else "day" + \
        __import__("datetime").datetime.fromtimestamp(int(ts)).strftime("%m%d")
    return int(n), tag, sit, int(ts)


def region_features(files, sl):
    X = []
    for f in files:
        a = np.load(f)
        mti = np.abs(a - a.mean(axis=0, keepdims=True))[:, :, :, :, sl]
        raw = np.abs(a)[:, :, :, :, sl]
        X.append([mti.mean(), mti.std(), np.median(mti),
                  mti.mean(axis=(0, 1, 2, 3)).max(), raw.mean()])
    return np.array(X)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", default=["human_dataset_v2"],
                    help="one or more dataset folders; pass several to check "
                         "whether a new bridging sitting de-aliases the old data")
    args = ap.parse_args()

    files = sorted(f for d in args.data
                   for f in glob.glob(os.path.join(d, "*", "*.npy")))
    parsed = [(f, p) for f in files if (p := parse(f))]
    if not parsed:
        raise SystemExit(f"no parseable recordings under {args.data}")

    files = [f for f, _ in parsed]
    lab = np.array([p[0] for _, p in parsed])
    sit = np.array([p[2] for _, p in parsed])

    n_bins = np.load(files[0], mmap_mode="r").shape[-1]
    rng = RANGE_OFFSET_M + np.arange(n_bins) * BIN_LENGTH_M
    first_empty = int(np.searchsorted(rng, SEATING_MAX_M))

    classes = sorted(set(lab.tolist()))
    chance = np.bincount(lab).max() / len(lab)

    print("=" * 66)
    print(f" {len(files)} recordings   classes {classes}   "
          f"majority baseline {chance*100:.1f}%")
    print(f" {n_bins} bins   {rng[0]:+.2f} m .. {rng[-1]:+.2f} m")
    print("=" * 66)

    # ---- test 1: is every class present in every sitting? ----------------
    print("\n1. SITTING BALANCE  (every class should appear in every sitting)\n")
    sittings = sorted(set(sit.tolist()))
    grid = defaultdict(int)
    for c, s in zip(lab, sit):
        grid[(s, c)] += 1
    print("   sitting     " + "".join(f"{c}p".rjust(8) for c in classes))
    for s in sittings:
        row = "".join(str(grid[(s, c)] or "-").rjust(8) for c in classes)
        missing = sum(1 for c in classes if not grid[(s, c)])
        print(f"   {s:<12}{row}   {'' if not missing else f'<- {missing} class(es) absent'}")

    aliased = [c for c in classes
               if len({s for s in sittings if grid[(s, c)]}) == 1]
    if aliased:
        print(f"\n   FAIL: class(es) {aliased} appear in exactly one sitting each.")
        print("         Label and session are aliased; no split can separate them.")
    else:
        print("\n   PASS: every class spans more than one sitting.")

    # ---- test 2: how much do non-seating regions predict? ----------------
    print("\n2. REGION PREDICTABILITY  (diagnostic -- cabin multipath is real)\n")
    regions = {
        f"behind antenna (bins 0-{FIRST_PHYSICAL_BIN-1})": slice(0, FIRST_PHYSICAL_BIN),
        f"beyond seating > {SEATING_MAX_M:.1f} m (bins {first_empty}+)":
            slice(first_empty, n_bins),
        f"far field > 5 m": slice(int(np.searchsorted(rng, 5.0)), n_bins),
    }
    worst = 0.0
    for name, sl in regions.items():
        X = region_features(files, sl)
        acc = cross_val_score(RandomForestClassifier(200, random_state=0),
                              X, lab, cv=5).mean()
        worst = max(worst, acc)
        flag = "  <-- above chance" if acc > chance + 0.10 else ""
        print(f"   {name:<40} {acc*100:5.1f}%{flag}")

    X = np.array([[np.abs(np.load(f)).mean()] for f in files])
    acc1 = cross_val_score(RandomForestClassifier(200, random_state=0),
                           X, lab, cv=5).mean()
    print(f"\n   {'single scalar (whole-frame mean |x|)':<40} {acc1*100:5.1f}%")
    for c in classes:
        v = X[lab == c, 0]
        print(f"      {c}p  mean {v.mean():.4f}  sd {v.std():.4f}  "
              f"range {v.min():.4f}-{v.max():.4f}")

    # ---- test 3: does far-region energy track time inside a sitting? -----
    print("\n3. WITHIN-SITTING TIME DRIFT  (diagnostic)\n")
    far = slice(first_empty, n_bins)
    max_r = 0.0
    for s_ in sittings:
        idx = [i for i in range(len(files)) if sit[i] == s_]
        if len(idx) < 10:
            continue
        idx.sort(key=lambda i: parsed[i][1][3])
        t = np.array([parsed[i][1][3] for i in idx], float)
        t = (t - t[0]) / 60.0
        e = np.array([np.abs(np.load(files[i]))[:, :, :, :, far].mean()
                      for i in idx])
        r = float(np.corrcoef(t, e)[0, 1]) if t.std() > 0 else 0.0
        max_r = max(max_r, abs(r))
        note = "drifting" if abs(r) > 0.6 else (
            "some trend" if abs(r) > 0.3 else "flat")
        print(f"   {s_:<12} n={len(idx):<5} span {t.max():5.0f} min   "
              f"r(time, far energy) = {r:+.2f}   {note}")

    print("\n" + "=" * 66)
    ok = not aliased
    if ok:
        print(" VERDICT: usable. Every class spans more than one sitting, so a")
        print("          session-aware split can test generalisation honestly.")
        if max_r > 0.6:
            print(" NOTE:    strong within-sitting drift -- prefer a split that")
            print("          holds out whole sittings, not random recordings.")
    else:
        print(" VERDICT: NOT INTERPRETABLE -- accuracy from this dataset cannot")
        print("          be trusted or refuted.")
        print(f"          class(es) {aliased} appear in exactly one sitting each,")
        print("          so 'counts occupants' and 'recognises the session' make")
        print("          identical predictions on every split.")
        print(f"          (Non-seating regions reach {worst*100:.1f}% vs "
              f"{chance*100:.1f}% chance, but in a")
        print("           metal cabin that is partly real multipath, so it is")
        print(f"           not proof of a fault. Max within-sitting drift "
              f"|r|={max_r:.2f}.)")
        print("")
        print("          Cheapest fix: record ONE bridging sitting containing")
        print("          every occupancy level, then retest. Full re-record is")
        print("          only needed if the bridge shows the old data does not")
        print("          generalise.")
    print("=" * 66)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
