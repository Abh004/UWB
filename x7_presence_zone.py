"""
x7_presence_zone.py
===================
Lightweight occupancy for the dual X7: occupied/empty + front/back row.

Design
------
A fixed physics front end turns radar frames into 11 features; three logistic
regressions (12 parameters each) turn those into probabilities; an asymmetric
state machine turns probabilities into decisions. No neural network, no GPU,
numpy only, well under 1 MFLOP/s at 5 FPS.

Why so small: the v2 dataset has ~800 recordings but only ~16 independent
sit-downs, and every higher-capacity model tried (count CNN, seat CNN) scored
~100% by memorising the sit-down and chance on a new one. Absolute phase is a
fingerprint of exactly where a body landed (1 cm = ~90 deg round trip at
lambda = 41 mm). These features never see absolute phase or absolute
amplitude, and 36 parameters leave no room to memorise.

Front end, per 10 s block (50 frames at 5 FPS)
    1. quadratic detrend over time per channel   (MTI + drift removal; drift
       in a 10 s window otherwise lands in the 0.2 Hz breathing bins)
    2. FFT over time
    3. breathing band 0.2-0.5 Hz, noise band 0.8-2.5 Hz
    4. thermal/coupling reference: bins behind the antenna (no occupant can
       be there, in a room or a car)
A window is K blocks. Per zone (front, back) the breathing-band snapshots of
all 8 virtual channels x zone bins are whitened by each element's own
noise-band power, and:
    eig     log(top eigenvalue / Marchenko-Pastur edge)   structure above noise
    ncomp   log(1 + #eigenvalues above the edge)
    snr     log(max breathing power / reference floor)    CFAR-like detection
    br      log(max breathing power / own noise band)     amplitude-free ratio
    motion  log(noise-band power / reference floor)       fidgeting, moving
plus
    coherence  |<front component, back component>|^2      leak rejection: a
               front occupant spilling into back bins is coherent with it

Safety policy: a missed occupant is worse than a false alarm, so
    - thresholds are set for >= 99% out-of-fold recall
    - "occupied" enters immediately (probability or per-frame motion) and is
      held EXIT_S seconds (default 60) after the last positive evidence
    - warm-up reports UNKNOWN, a stalled sensor reports SENSOR FAULT;
      neither ever reports EMPTY

Validation is leave-one-sit-down-out. Only one empty "sit-down" exists in v2,
so it is split into two chronological halves; empty-room false-alarm figures
therefore rest on a single session and are weak evidence.

Usage
    python x7_presence_zone.py train                         # human_dataset_v2
    python x7_presence_zone.py train --windows 1 2 3 6
    python x7_presence_zone.py replay --runs 16 --run-len 20
    python x7_presence_zone.py live --host1 529 --host2 545
"""

import os
import re
import sys
import glob
import json
import time
import argparse
import datetime
import collections

import numpy as np

from x7_physics import BIN_LENGTH_M, RANGE_OFFSET_M, FIRST_PHYSICAL_BIN

FPS = 5.0
BLOCK = 50                       # frames per block = 10 s, matches recordings
HOP = 5                          # live decision every 5 frames = 1 s
BREATH_BAND = (0.2, 0.5)         # Hz
NOISE_BAND = (0.8, 2.5)          # Hz
ZONES = {"front": (0.35, 1.05), "back": (1.25, 2.0)}   # metres from radar
REF_BINS = FIRST_PHYSICAL_BIN - 4  # bins [0, REF_BINS): behind the antenna
OUTPUTS = ("occupied", "front", "back")
FEAT_KINDS = ("eig", "ncomp", "snr", "br", "motion")
EXIT_S = 60.0
TARGET_RECALL = 0.99
MAX_USABLE_FA = 0.5              # above this at the recall target an output is noise
Y_ONLY_FA = 0.01                 # an unreliable output may still say "Y" at this FA
EPS = 1e-30

# Setup-change monitoring. Numbers measured on v2 + v3 (Sep 25):
#   front-row breathing peak, same setup:  10 s scatter sd ~9 cm, daily medians
#   within -3.6..+5.2 cm; a simulated radar move of 7/14/21 cm shows up as
#   +7.6/+14.8/+21.9 cm. The back row is too noisy (sd 12.7 cm) to monitor.
#   receiver reference noise level: sd 0.9 dB, max 3.3 dB across all days.
#   Static-room fingerprints and wall-reflector lags were tried and could not
#   tell a moved radar from an unchanged one.
ALIGN_TOL_M = 0.10               # front-row drift that counts as a setup change
ALIGN_MIN_SAMPLES = 18           # 10 s blocks with a front occupant (~3 min)
ALIGN_MAX_M = 0.30               # auto-alignment never moves zones further than this
NOISE_TOL_DB = 6.0               # reference noise vs calibration
NOISE_HOLD = 30                  # consecutive decisions (s) out of tolerance to flag
PEAK_MARGIN_M = 0.30             # search margin around a zone for its breathing peak
MIN_PEAK_RANGE_M = 0.35          # antenna near-field; no occupant can sit closer


# ─── front end ──────────────────────────────────────────────────────────────

def make_layout(n_bins, zones=None):
    zones = zones or ZONES
    rng = RANGE_OFFSET_M + np.arange(n_bins) * BIN_LENGTH_M
    zi = {z: np.where((rng >= lo) & (rng < hi))[0] for z, (lo, hi) in zones.items()}
    for z, idx in zi.items():
        if len(idx) == 0:
            raise ValueError(f"zone {z} {zones[z]} contains no range bins")
    return {"zones": zi, "ref": np.arange(REF_BINS), "n_bins": n_bins,
            "zone_m": {z: list(v) for z, v in zones.items()}}


def feature_names(layout):
    return [f"{z}_{k}" for z in layout["zones"] for k in FEAT_KINDS] + ["coherence"]


_DETREND = {}


def _detrender(n):
    if n not in _DETREND:
        t = np.arange(n) / FPS
        V = np.vstack([np.ones(n), t, t ** 2]).T
        _DETREND[n] = np.eye(n) - V @ np.linalg.pinv(V)
    return _DETREND[n]


def block_stats(block, layout):
    """block: (50, 2 sensors, 2 tx, 2 rx, bins) complex -> per-block statistics."""
    n = block.shape[0]
    x = block.reshape(n, 8, -1).astype(np.complex128)
    x = np.tensordot(_detrender(n), x, axes=(1, 0))
    X = np.fft.fft(x, axis=0)
    f = np.abs(np.fft.fftfreq(n, 1.0 / FPS))
    br = (f >= BREATH_BAND[0]) & (f <= BREATH_BAND[1])
    nz = (f >= NOISE_BAND[0]) & (f <= NOISE_BAND[1])
    ref = np.abs(X[:, :, layout["ref"]]) ** 2
    out = {"ref": ref.mean(axis=(0, 2))}                       # (8,)
    for z, idx in layout["zones"].items():
        Xz = X[:, :, idx]                                      # (n, 8, nb)
        S = Xz[br].reshape(int(br.sum()), -1)                  # snapshots (nbr, p)
        pn = (np.abs(Xz[nz]) ** 2).mean(0).reshape(-1)         # (p,)
        out[z] = (S, pn)
    return out


def _top_eig(Sw):
    """Eigenvalues (descending) and top eigenvector of Sw^H Sw / n, cheaply."""
    n, p = Sw.shape
    if n <= p:
        G = Sw @ Sw.conj().T / n
        w, U = np.linalg.eigh(G)
        v = Sw.conj().T @ U[:, -1]
        v = v / (np.linalg.norm(v) + EPS)
    else:
        w, V = np.linalg.eigh(Sw.conj().T @ Sw / n)
        v = V[:, -1]
    return np.clip(w[::-1].real, 0, None), v


def window_features(stats, layout):
    """stats: list of K block_stats -> feature vector (11,)."""
    ref = np.mean([s["ref"] for s in stats], axis=0) + EPS
    feats, comp = [], {}
    for z, idx in layout["zones"].items():
        nb = len(idx)
        ch = np.repeat(np.arange(8), nb)
        S = np.concatenate([s[z][0] for s in stats])
        pn = np.mean([s[z][1] for s in stats], axis=0) + EPS
        Sw = S / np.sqrt(pn)
        n, p = S.shape
        w, v = _top_eig(Sw)
        edge = (1.0 + np.sqrt(p / n)) ** 2
        pb = (np.abs(S) ** 2).mean(0)
        feats += [np.log(w[0] / edge + EPS),
                  np.log1p(float((w > edge).sum())),
                  np.log((pb / ref[ch]).max() + EPS),
                  np.log((pb / pn).max() + EPS),
                  np.log((pn / ref[ch]).mean() + EPS)]
        comp[z] = Sw @ v
    zs = list(layout["zones"])
    a, b = comp[zs[0]], comp[zs[1]]
    coh = np.abs(np.vdot(a, b)) ** 2 / ((np.vdot(a, a).real * np.vdot(b, b).real) + EPS)
    feats.append(float(coh))
    return np.array(feats, dtype=np.float64)


def row_peak(block, zone_m, margin=PEAK_MARGIN_M):
    """Range (m) of the strongest breathing-band return near a zone.

    Breathing-band power per range bin, per sensor, parabolic sub-bin peak,
    mean of the two sensors. This is the quantity that moves with the radar.
    """
    n = block.shape[0]
    x = np.tensordot(_detrender(n), block.reshape(n, 2, 4, -1).astype(np.complex128),
                     axes=(1, 0))
    X = np.fft.fft(x, axis=0)
    f = np.abs(np.fft.fftfreq(n, 1.0 / FPS))
    br = (f >= BREATH_BAND[0]) & (f <= BREATH_BAND[1])
    P = (np.abs(X[br]) ** 2).mean(axis=(0, 2))                  # (2 sensors, bins)
    rng = RANGE_OFFSET_M + np.arange(P.shape[-1]) * BIN_LENGTH_M
    lo, hi = zone_m
    idx = np.where((rng >= max(lo - margin, MIN_PEAK_RANGE_M)) & (rng < hi + margin))[0]
    out = []
    for sn in range(P.shape[0]):
        q = P[sn, idx]
        i = int(np.argmax(q))
        off = 0.0
        if 0 < i < len(q) - 1:
            y0, y1, y2 = np.log(q[i - 1:i + 2] + EPS)
            d = y0 - 2 * y1 + y2
            if d != 0:
                off = float(np.clip(0.5 * (y0 - y2) / d, -0.5, 0.5))
        out.append(rng[idx[0]] + (i + off) * BIN_LENGTH_M)
    return float(np.mean(out))


def zones_for(model, calib=None, extra_bins=0):
    """Trained zones, shifted by whole range bins to where calibration found
    the rows. Widths never change, so features stay comparable to training."""
    sh = (calib or {}).get("zone_shift_bins", {})
    out = {}
    for z, (lo, hi) in model["zones_m"].items():
        k = sh.get(z, 0) + extra_bins
        out[z] = (lo + k * BIN_LENGTH_M, hi + k * BIN_LENGTH_M)
    return out


def frame_motion(prev, cur, layout):
    """Per-frame fast path: frame-difference power in the zones vs the reference."""
    d = (cur - prev).reshape(8, -1)
    zi = np.concatenate(list(layout["zones"].values()))
    num = (np.abs(d[:, zi]) ** 2).mean()
    den = (np.abs(d[:, layout["ref"]]) ** 2).mean() + EPS
    return float(np.log(num / den + EPS))


# ─── dataset ────────────────────────────────────────────────────────────────

_NAME = re.compile(r"(\d+)people_([A-Za-z0-9]+)_(.+?)_(\d{10})_")


def labels_from_tag(n, tag):
    seats = [int(c) for c in tag[5:]] if tag.startswith("seats") else []
    return np.array([n > 0, any(s in (1, 2) for s in seats),
                     any(s in (3, 4) for s in seats)], dtype=int)


def load_trials(data_dir):
    """data_dir: one folder or a list of folders (e.g. v2 + v3)."""
    dirs = [data_dir] if isinstance(data_dir, str) else list(data_dir)
    trials = []
    for fp in sorted(f for d in dirs for f in glob.glob(os.path.join(d, "*", "*.npy"))):
        m = _NAME.match(os.path.basename(fp))
        if m:
            trials.append(dict(path=fp, n=int(m.group(1)), tag=m.group(2),
                               subj=m.group(3), ts=int(m.group(4))))
    if not trials:
        raise SystemExit(f"no recordings under {data_dir}")
    # sit-down = one scenario in one sitting. Deliberately conservative: gaps
    # inside a block do not split it, because occupants may never have stood
    # up, and a split would leak the same body fingerprint across a fold.
    trials.sort(key=lambda t: (t["subj"], t["tag"], t["ts"]))
    keys = {}
    for t in trials:
        t["group"] = keys.setdefault((t["subj"], t["tag"]), len(keys))
        t["day"] = datetime.datetime.fromtimestamp(t["ts"]).strftime("%m-%d")
    gid = len(keys) - 1
    # a lone empty sit-down cannot sit on both sides of a fold: split it
    empty = sorted({t["group"] for t in trials if t["n"] == 0})
    split_note = None
    if len(empty) == 1:
        g = empty[0]
        members = sorted([t for t in trials if t["group"] == g], key=lambda t: t["ts"])
        for t in members[len(members) // 2:]:
            t["group"] = gid + 1
        split_note = ("only one empty sit-down: split into two chronological "
                      "halves, so empty-room figures rest on a single session")
    return trials, split_note


def build_windows(trials, stats, K, layout):
    X, Y, G, D = [], [], [], []
    by = collections.defaultdict(list)
    for i, t in enumerate(trials):
        by[t["group"]].append(i)
    stride = max(1, K // 2)
    for g, idx in by.items():
        idx.sort(key=lambda i: trials[i]["ts"])
        for s in range(0, len(idx) - K + 1, stride):
            w = idx[s:s + K]
            X.append(window_features([stats[i] for i in w], layout))
            t = trials[w[0]]
            Y.append(labels_from_tag(t["n"], t["tag"]))
            G.append(g)
            D.append(t["day"])
    return np.array(X), np.array(Y), np.array(G), np.array(D)


# ─── model ──────────────────────────────────────────────────────────────────

def fit_lr(X, y):
    from sklearn.linear_model import LogisticRegression
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    lr = LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000)
    lr.fit((X - mu) / sd, y)
    return {"mean": mu.tolist(), "std": sd.tolist(),
            "w": lr.coef_[0].tolist(), "b": float(lr.intercept_[0])}


def predict_lr(p, X):
    z = (np.atleast_2d(X) - np.array(p["mean"])) / np.array(p["std"])
    return 1.0 / (1.0 + np.exp(-(z @ np.array(p["w"]) + p["b"])))


def loso(X, Y, G):
    oof = np.full(Y.shape, np.nan)
    for g in np.unique(G):
        tr, te = G != g, G == g
        for o in range(Y.shape[1]):
            if len(np.unique(Y[tr, o])) < 2:
                continue
            oof[te, o] = predict_lr(fit_lr(X[tr], Y[tr, o]), X[te])
    return oof


def recall_threshold(p, y, G, target=TARGET_RECALL):
    from sklearn.metrics import roc_auc_score
    ok = ~np.isnan(p)
    p, y, G = p[ok], y[ok], G[ok]
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    thr = float(np.quantile(pos, 1.0 - target))
    worst = min(((p[(G == g)] >= thr).mean(), g) for g in np.unique(G[y == 1]))
    return {"auc": float(roc_auc_score(y, p)), "threshold": thr,
            "recall": float((pos >= thr).mean()),
            "false_alarm": float((neg >= thr).mean()) if len(neg) else float("nan"),
            "worst_sitdown_recall": float(worst[0]), "n_pos": int(len(pos)),
            "n_neg": int(len(neg))}


# ─── train ──────────────────────────────────────────────────────────────────

def cmd_train(args):
    trials, split_note = load_trials(args.data)
    n_bins = np.load(trials[0]["path"], mmap_mode="r").shape[-1]
    layout = make_layout(n_bins)
    names = feature_names(layout)
    groups = len({t["group"] for t in trials})
    print(f"{len(trials)} recordings, {groups} sit-downs, {n_bins} bins")
    for z, idx in layout["zones"].items():
        print(f"   zone {z:<6} {ZONES[z][0]:.2f}-{ZONES[z][1]:.2f} m  "
              f"bins {idx[0]}-{idx[-1]}  channels {8 * len(idx)}")
    if split_note:
        print(f"   NOTE: {split_note}")

    stats = [block_stats(np.load(t["path"]), layout) for t in trials]

    # Where a single occupant's breathing peaks, per row, in the training
    # geometry. Calibration measures the same thing in a new setup and shifts
    # the zones by the difference.
    refs = {"front": [], "back": []}
    for t in trials:
        if t["n"] == 1 and t["tag"].startswith("seats"):
            seat = int(t["tag"][5:])
            row = "front" if seat in (1, 2) else "back"
            refs[row].append(row_peak(np.load(t["path"]), ZONES[row]))
    row_ref = {r: float(np.median(v)) for r, v in refs.items() if v}
    print("   single-occupant breathing peak (reference for calibration): "
          + "  ".join(f"{r} {v:.3f} m" for r, v in row_ref.items()))

    print(f"\nthresholds for {TARGET_RECALL:.0%} recall.  LOSO = leave-one-sit-down-out,"
          f" LODO = leave-one-DAY-out")
    print("LODO is the cross-session test and sets the threshold wherever it exists")
    print(f"{'window':>7}  {'output':<9}{'split':<6}{'AUC':>6}{'recall':>8}"
          f"{'false alarm':>13}{'worst sit-down':>16}")
    results = {}
    for K in args.windows:
        X, Y, G, D = build_windows(trials, stats, K, layout)
        oof_g, oof_d = loso(X, Y, G), loso(X, Y, D)
        met = {}
        for i, o in enumerate(OUTPUTS):
            mg = recall_threshold(oof_g[:, i], Y[:, i], G)
            md = recall_threshold(oof_d[:, i], Y[:, i], G)
            met[o] = dict(md or mg)
            met[o]["split"] = "LODO" if md else "LOSO"
            for tag, m in (("LOSO", mg), ("LODO", md)):
                if m is None:
                    print(f"{K * BLOCK / FPS:6.0f}s  {o:<9}{tag:<6}   n/a   "
                          f"(some held-out day leaves a class with no training data)")
                    continue
                print(f"{K * BLOCK / FPS:6.0f}s  {o:<9}{tag:<6}{m['auc']:6.3f}"
                      f"{m['recall']*100:7.1f}%{m['false_alarm']*100:12.1f}%"
                      f"{m['worst_sitdown_recall']*100:15.1f}%")
        results[K] = {"X": X, "Y": Y, "G": G, "metrics": met, "oof_d": oof_d, "oof_g": oof_g}
    # An output that cannot reach the recall target without flagging most
    # negatives is not a detector. It is kept for display only, never allowed
    # to drive "occupied", and ignored when choosing the window.
    usable = {o: min(results[k]["metrics"][o]["false_alarm"] for k in results)
              < MAX_USABLE_FA for o in OUTPUTS}
    for o in OUTPUTS:
        if not usable[o]:
            print(f"\n{o}: false alarm >= {MAX_USABLE_FA:.0%} at {TARGET_RECALL:.0%} recall "
                  f"for every window -> marked UNRELIABLE (display only)")
    crit = [o for o in OUTPUTS if usable[o]] or list(OUTPUTS)
    K = args.window or min(results, key=lambda k: (
        round(sum(results[k]["metrics"][o]["false_alarm"] for o in crit), 3), k))
    print(f"\nselected window {K * BLOCK / FPS:.0f} s"
          + ("" if args.window else f" (lowest false alarm on {', '.join(crit)})"))

    model = {"caveats": []}
    if not args.no_control:
        ctrl = make_layout(n_bins, {"front": (2.6, 3.2), "back": (3.2, 3.9)})
        cstats = [block_stats(np.load(t["path"]), ctrl) for t in trials]
        Xc, Yc, Gc, _ = build_windows(trials, cstats, K, ctrl)
        m = recall_threshold(loso(Xc, Yc, Gc)[:, 0], Yc[:, 0], Gc)
        print(f"\nCONTROL, same pipeline on 2.6-3.9 m (no seat there): "
              f"occupied AUC {m['auc']:.3f}")
        model["control_auc"] = m["auc"]
        occ_lodo = results[K]["metrics"]["occupied"]["split"] == "LODO"
        if m["auc"] > 0.75 and occ_lodo:
            # Empty trials exist on more than one day, so occupied has a
            # cross-day test. Occupants shadow the walls behind them and
            # reflect into far bins, so a no-seat region responds to people
            # too: the control cannot separate "people" from "session" here.
            print("   the no-seat region also responds to occupants (bodies shadow and "
                  "reflect into\n   far bins). Not diagnostic for occupied-vs-empty; "
                  "the leave-one-day-out\n   result above is the evidence that counts.")
            model["caveats"].append(
                f"control AUC {m['auc']:.3f}: the no-seat region responds to occupants "
                f"themselves; occupied is validated leave-one-day-out instead")
        elif m["auc"] > 0.75:
            print("   WARNING: the region with no seats also predicts 'occupied', and "
                  "empty trials\n   exist on one day only, so session and occupancy "
                  "cannot be separated.\n   Empty-room figures above are optimistic.")
            model["caveats"].append(
                f"control: non-seat region predicts occupied at AUC {m['auc']:.3f} and "
                f"empty trials come from one day only; empty-room figures are optimistic")
        else:
            print("   no leakage through the non-seat region")

    X, Y, G = results[K]["X"], results[K]["Y"], results[K]["G"]
    model = {**model, "version": 1, "created": datetime.datetime.now().isoformat(timespec="seconds"),
             "data": args.data, "fps": FPS, "block": BLOCK, "window_blocks": K,
             "breath_band": BREATH_BAND, "noise_band": NOISE_BAND,
             "zones_m": layout["zone_m"], "ref_bins": REF_BINS, "n_bins": n_bins,
             "features": names, "exit_s": args.exit_s, "row_ref_m": row_ref,
             "outputs": {}, "evaluation": {}}
    for i, o in enumerate(OUTPUTS):
        p = fit_lr(X, Y[:, i])
        p["threshold"] = results[K]["metrics"][o]["threshold"]
        p["usable"] = bool(usable[o])
        p["validated_by"] = results[K]["metrics"][o]["split"]
        if not usable[o]:
            oo = results[K]["oof_d"][:, i]
            if np.isnan(oo).all():
                oo = results[K]["oof_g"][:, i]
            ok = ~np.isnan(oo)
            neg, pos = oo[ok & (Y[:, i] == 0)], oo[ok & (Y[:, i] == 1)]
            yt = float(np.quantile(neg, 1.0 - Y_ONLY_FA))
            p["y_only_threshold"] = yt
            p["y_only_recall"] = float((pos >= yt).mean())
            print(f"{o}: 'Y only' at {Y_ONLY_FA:.0%} false alarm -> threshold {yt:.3f}, "
                  f"recall {p['y_only_recall']*100:.1f}% (shown as Y or ?, never '-')")
        model["outputs"][o] = p
    model["evaluation"] = {f"{k * BLOCK / FPS:.0f}s": v["metrics"] for k, v in results.items()}

    # per-frame motion fast path: threshold from empty-room frames
    em, oc = [], []
    for t in trials:
        a = np.load(t["path"])
        v = [frame_motion(a[j - 1], a[j], layout) for j in range(1, len(a))]
        (em if t["n"] == 0 else oc).extend(v)
    thr_m = float(np.percentile(em, 99.9) + 0.5)
    model["motion_threshold"] = thr_m
    print(f"\nmotion fast path: threshold {thr_m:.2f} (empty 99.9th pct + margin); "
          f"trips on {np.mean(np.array(oc) > thr_m)*100:.1f}% of occupied frames")
    if split_note:
        model["caveats"].append(split_note)
    model["caveats"].append("thresholds chosen on the same out-of-fold predictions "
                            "they are reported on: recall/false-alarm are optimistic")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(model, f, indent=2)
    n_par = sum(len(model["outputs"][o]["w"]) + 1 for o in OUTPUTS)
    print(f"saved {args.out}   ({n_par} learned parameters)")


# ─── runtime ────────────────────────────────────────────────────────────────

class ZoneDetector:
    """Frame-in, decision-out. Identical front end to training."""

    def __init__(self, model, exit_s=None, calib=None, auto_align=True):
        self.m = model
        self.K = model["window_blocks"]
        self.exit_s = exit_s if exit_s is not None else model["exit_s"]
        self.calib = calib
        self.auto_align = auto_align
        self.align_bins = 0
        self.align_samples = collections.deque(maxlen=30)
        ref = (calib or {}).get("row_peak_m") or model.get("row_ref_m") or {}
        self.expected_front = ref.get("front")
        self.noise_ref_db = (calib or {}).get("noise_ref_db")
        self.noise_bad = 0
        self.warnings = {}                  # key -> (text, blocks_empty)
        self.motion_thr = (calib or {}).get("motion_threshold", model["motion_threshold"])
        if calib and calib.get("model_created") not in (None, model.get("created")):
            self.warnings["calib_model"] = (
                "calibration was made with a different model -- recalibrate", True)
        self._build_layout()
        self.frames = collections.deque(maxlen=self.K * BLOCK)
        self.prev = None
        self.n = 0
        self.last_on = {o: -np.inf for o in OUTPUTS}
        self.probs = {o: float("nan") for o in OUTPUTS}
        self.motion = False
        self.motion_val = float("nan")
        self.last_src = {"motion": -np.inf, "p_occupied": -np.inf,
                         "front": -np.inf, "back": -np.inf}

    def _build_layout(self):
        self.zones = zones_for(self.m, self.calib, self.align_bins)
        self.layout = make_layout(self.m["n_bins"], self.zones)

    @property
    def now(self):
        return self.n / FPS

    def _health(self, stats):
        """Noise-level check every decision; front-row geometry every 10 s."""
        if self.noise_ref_db is not None:
            lvl = 10 * np.log10(stats[-1]["ref"].reshape(2, 4).mean(1) + EPS)
            dev = lvl - np.array(self.noise_ref_db)
            if np.abs(dev).max() > NOISE_TOL_DB:
                self.noise_bad += 1
            else:
                self.noise_bad = 0
                self.warnings.pop("noise", None)
            if self.noise_bad >= NOISE_HOLD:
                sn = int(np.argmax(np.abs(dev))) + 1
                self.warnings["noise"] = (f"sensor {sn} signal level {dev[sn - 1]:+.1f} dB vs "
                                          f"calibration -- check the radar", True)
        if self.expected_front is None or self.n % BLOCK:
            return
        if self.probs["front"] < self.m["outputs"]["front"]["threshold"]:
            return                                        # nobody to measure
        arr = np.asarray(self.frames)[-BLOCK:]
        expect = self.expected_front + self.align_bins * BIN_LENGTH_M
        self.align_samples.append(row_peak(arr, self.zones["front"]) - expect)
        if len(self.align_samples) < ALIGN_MIN_SAMPLES:
            return
        est = float(np.median(self.align_samples))
        if abs(est) <= ALIGN_TOL_M:
            self.warnings.pop("setup", None)
            return
        # toward zero: the peak estimate runs slightly past the true move, so
        # rounding would over-correct; a residual under ALIGN_TOL_M is harmless
        k = int(np.trunc(est / BIN_LENGTH_M))
        total = (self.align_bins + k) * BIN_LENGTH_M
        if self.auto_align and k and abs(total) <= ALIGN_MAX_M:
            self.align_bins += k
            self._build_layout()
            self.align_samples.clear()
            self.warnings.pop("setup", None)
            self.warnings["aligned"] = (f"front row moved {total * 100:+.0f} cm since "
                                        f"calibration: zones auto-aligned -- recalibrate "
                                        f"when convenient", False)
        else:
            self.warnings["setup"] = (f"front row sits {est * 100:+.0f} cm from calibration "
                                      f"-- SETUP CHANGED, recalibrate", True)

    def update(self, frame):
        self.n += 1
        if self.prev is not None:
            self.motion_val = frame_motion(self.prev, frame, self.layout)
            self.motion = self.motion_val > self.motion_thr
            if self.motion:
                self.last_on["occupied"] = self.now
                self.last_src["motion"] = self.now
        self.prev = frame
        self.frames.append(frame)
        if self.n % HOP:
            return None
        if len(self.frames) < self.frames.maxlen:
            return self.state()                     # UNKNOWN, with countdown
        arr = np.asarray(self.frames)
        stats = [block_stats(arr[i * BLOCK:(i + 1) * BLOCK], self.layout) for i in range(self.K)]
        f = window_features(stats, self.layout)
        for o in OUTPUTS:
            p = self.m["outputs"][o]
            self.probs[o] = float(predict_lr(p, f)[0])
            thr = p["threshold"] if p.get("usable", True) else p.get("y_only_threshold", np.inf)
            if self.probs[o] >= thr:
                self.last_on[o] = self.now
                self.last_src["p_occupied" if o == "occupied" else o] = self.now
        # a detected row is an occupied cabin -- but only a row detector that
        # met the recall target without flagging most negatives
        for o in ("front", "back"):
            if self.m["outputs"][o].get("usable", True):
                self.last_on["occupied"] = max(self.last_on["occupied"], self.last_on[o])
        self._health(stats)
        return self.state()

    def state(self):
        if len(self.frames) < self.frames.maxlen:
            left = (self.frames.maxlen - len(self.frames)) / FPS
            if (self.now - self.last_on["occupied"]) < self.exit_s:
                # motion already proves someone is there; rows need the full
                # window, and EMPTY can never be claimed during warm-up
                return {"status": "OCCUPIED", "warming": True, "left": left,
                        "occupied": True, "front": False, "back": False,
                        "probs": dict(self.probs), "motion": self.motion,
                        "motion_val": self.motion_val, "motion_thr": self.motion_thr,
                        "warnings": [w for w, _ in self.warnings.values()],
                        "usable": {o: self.m["outputs"][o].get("usable", True)
                                   for o in OUTPUTS}}
            return {"status": "UNKNOWN", "detail": f"warming up, {left:.0f} s"}
        s = {o: (self.now - self.last_on[o]) < self.exit_s for o in OUTPUTS}
        src = {k: v for k, v in self.last_src.items()
               if k in ("motion", "p_occupied") or self.m["outputs"][k].get("usable", True)}
        why = max(src, key=src.get)
        warn = [w for w, _ in self.warnings.values()]
        blocking = any(b for _, b in self.warnings.values())
        status = "OCCUPIED" if s["occupied"] else "EMPTY"
        if status == "EMPTY" and blocking:
            # never claim an empty cabin when the setup may have changed
            return {"status": "UNKNOWN", "detail": "; ".join(warn), "warnings": warn}
        return {"status": status, **s, "warnings": warn,
                "align_cm": self.align_bins * BIN_LENGTH_M * 100,
                "probs": dict(self.probs), "motion": self.motion,
                "motion_val": self.motion_val,
                "motion_thr": self.m["motion_threshold"],
                "held_by": why, "held_ago": self.now - src[why],
                "clears_in": max(0.0, self.exit_s - (self.now - self.last_on["occupied"])),
                "usable": {o: self.m["outputs"][o].get("usable", True) for o in OUTPUTS}}


def _fmt(st):
    if st["status"] == "UNKNOWN":
        return f"UNKNOWN ({st['detail']})"
    if st.get("warming"):
        return (f"OCCUPIED (motion)  front ?  back ?  | rows in {st['left']:.0f} s"
                f"  | motion {st['motion_val']:4.1f}/{st['motion_thr']:.1f}")
    p = st["probs"]
    u = st.get("usable", {})
    # an unreliable row can confirm an occupant but never claim the seat is empty
    row = lambda o: (("Y" if st[o] else "?") if not u.get(o, True)
                     else ("Y" if st[o] else "-"))
    line = (f"{st['status']:<8} front {row('front')}  back {row('back')}"
            f"  | p occ {p['occupied']:.2f} front "
            f"{p['front']:.2f} back {p['back']:.2f}"
            f"  | motion {st['motion_val']:4.1f}/{st['motion_thr']:.1f}")
    if st["status"] == "OCCUPIED" and "held_by" in st:
        names = {"motion": "motion", "p_occupied": "p occ", "front": "front row",
                 "back": "back row"}
        line += (f"  | held by {names[st['held_by']]} {st['held_ago']:.0f}s ago, "
                 f"clears in {st['clears_in']:.0f}s if nothing new")
    if st.get("warnings"):
        line += "  | ! " + "; ".join(st["warnings"])
    return line


def cmd_replay(args):
    """Stream recordings through the detector as RUNS of one sit-down, so the
    state machine sees real transitions (occupied -> empty -> occupied)."""
    model = json.load(open(args.model))
    by = collections.defaultdict(list)
    for fp in sorted(glob.glob(args.files)):
        m = _NAME.match(os.path.basename(fp))
        if m:
            by[(int(m.group(1)), m.group(2))].append(fp)
    occ = [k for k in by if k[0] > 0]
    emp = [k for k in by if k[0] == 0]
    rng = np.random.default_rng(args.seed)
    plan, prev_empty = [], True
    for _ in range(args.runs):
        # alternate so every run is a transition; empty only if we have any
        k = (emp[rng.integers(len(emp))] if (not prev_empty and emp and rng.random() < 0.5)
             else occ[rng.integers(len(occ))])
        prev_empty = k[0] == 0
        files = by[k]
        s0 = rng.integers(0, max(1, len(files) - args.run_len + 1))
        plan.append((k, files[s0:s0 + args.run_len]))
    det = ZoneDetector(model, args.exit_s)
    miss = false = total = 0
    detect_lat, clear_lat = [], []
    for (n, tag), files in plan:
        t0 = det.now
        first_ok = None
        for fp in files:
            for fr in np.load(fp):
                st = det.update(fr)
                if not st or st["status"] == "UNKNOWN":
                    continue
                total += 1
                o = st["occupied"]
                miss += int(n > 0 and not o)
                false += int(n == 0 and o)
                if first_ok is None and (o == (n > 0)):
                    first_ok = det.now - t0
        span = det.now - t0
        (detect_lat if n > 0 else clear_lat).append(first_ok if first_ok is not None else np.inf)
        print(f"t={t0:7.0f}s  {n}p {tag:<9} {span:5.0f}s   correct after "
              + (f"{first_ok:5.0f} s" if first_ok is not None else "  never"))
    if total:
        dl = np.array(detect_lat); cl = np.array(clear_lat)
        print(f"\n{total} decisions over {det.now/60:.0f} min")
        print(f"   missed-occupant  {miss/total*100:5.1f}% of decisions")
        print(f"   false-occupied   {false/total*100:5.1f}% of decisions")
        if len(dl):
            print(f"   time to report OCCUPIED after occupants arrive: median "
                  f"{np.median(dl):.0f} s, max {dl.max():.0f} s")
        if len(cl):
            print(f"   time to report EMPTY after everyone leaves:    median "
                  f"{np.median(cl):.0f} s, max {cl.max():.0f} s "
                  f"(window {det.K*BLOCK/FPS:.0f} s + hold {det.exit_s:.0f} s)")
        print("   replaying training data is optimistic: this checks the state "
              "machine and latencies, not generalisation")


def cmd_live(args):
    from x7_acq_worker import RadarProcess, clear_stale_workers
    model = json.load(open(args.model))
    calib = None
    if args.calib and os.path.exists(args.calib):
        calib = json.load(open(args.calib))
        age = (datetime.datetime.now()
               - datetime.datetime.fromisoformat(calib["created"])).total_seconds() / 3600
        print(f"[INFO] calibration {args.calib} ({age:.1f} h old): zone shift "
              + ", ".join(f"{z} {k * BIN_LENGTH_M * 100:+.0f} cm"
                          for z, k in calib["zone_shift_bins"].items()))
    else:
        print("[WARN] no calibration found -- using the training geometry. Run\n"
              "       python x7_presence_setup.py calibrate --host1 .. --host2 ..")
    det = ZoneDetector(model, args.exit_s, calib=calib, auto_align=not args.no_auto_align)
    clear_stale_workers()
    radar = RadarProcess(args.host1, args.host2)
    radar.wait_ready()
    print(f"[INFO] window {det.K * BLOCK / FPS:.0f} s, hold {det.exit_s:.0f} s, "
          f"thresholds for {TARGET_RECALL:.0%} recall")
    t_start = time.time()
    first_frame = None          # the SDK needs a few seconds before it streams
    last_frame = None
    n_frames = 0
    rate = collections.deque(maxlen=25)

    def show(msg):
        sys.stdout.write("\r" + msg + " " * 12)
        sys.stdout.flush()

    try:
        while True:
            if args.duration and time.time() - t_start > args.duration:
                print(f"\n[INFO] --duration {args.duration:.0f} s reached, "
                      f"{n_frames} frames received")
                break
            fr = radar.get_frame(timeout=1.0)
            now = time.time()
            if fr is None:
                if first_frame is None:
                    wait = now - t_start
                    if wait > args.startup_timeout:
                        show(f"SENSOR FAULT: radar started but no frames after "
                             f"{wait:.0f} s -- occupancy unknown")
                    else:
                        show(f"waiting for radar stream... {wait:.0f} s")
                elif now - last_frame > 2.0:
                    show(f"SENSOR FAULT: no frames for {now - last_frame:.0f} s "
                         f"-- occupancy unknown")
                continue
            if first_frame is None:
                first_frame = now
                print(f"\n[INFO] first frame after {now - t_start:.1f} s")
            last_frame = now
            n_frames += 1
            rate.append(now)
            if fr.shape[-1] != model["n_bins"]:
                raise SystemExit(f"radar sends {fr.shape[-1]} bins, model expects "
                                 f"{model['n_bins']}")
            st = det.update(fr)
            if st:
                fps = (len(rate) - 1) / (rate[-1] - rate[0]) if len(rate) > 1 else 0.0
                show(f"{_fmt(st)}  | {fps:3.1f} fps")
    except KeyboardInterrupt:
        print("\n[INFO] stopped")
    finally:
        radar.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", nargs="+", default=["human_dataset_v2"],
                   help="one or more dataset folders")
    t.add_argument("--out", default=os.path.join("models", "presence_zone.json"))
    t.add_argument("--windows", type=int, nargs="+", default=[1, 2, 3, 6],
                   help="window lengths to evaluate, in 10 s blocks")
    t.add_argument("--window", type=int, default=None,
                   help="force this window (blocks) instead of auto-selecting")
    t.add_argument("--exit-s", dest="exit_s", type=float, default=EXIT_S)
    t.add_argument("--no-control", action="store_true")
    r = sub.add_parser("replay")
    r.add_argument("--model", default=os.path.join("models", "presence_zone.json"))
    r.add_argument("--files", default=os.path.join("human_dataset_v2", "*", "*.npy"))
    r.add_argument("--runs", type=int, default=16, help="number of sit-down runs")
    r.add_argument("--run-len", dest="run_len", type=int, default=20,
                   help="consecutive 10 s files per run (20 = ~200 s)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--exit-s", dest="exit_s", type=float, default=None)
    lv = sub.add_parser("live")
    lv.add_argument("--host1", required=True)
    lv.add_argument("--host2", required=True)
    lv.add_argument("--model", default=os.path.join("models", "presence_zone.json"))
    lv.add_argument("--exit-s", dest="exit_s", type=float, default=None)
    lv.add_argument("--calib", default=os.path.join("models", "presence_calib.json"),
                    help="calibration file from x7_presence_setup.py calibrate")
    lv.add_argument("--no-auto-align", dest="no_auto_align", action="store_true",
                    help="report a moved setup but do not shift the zones")
    lv.add_argument("--duration", type=float, default=0,
                    help="stop cleanly after this many seconds (0 = run until Ctrl+C)")
    lv.add_argument("--startup-timeout", dest="startup_timeout", type=float, default=20,
                    help="seconds to wait for the first frame before reporting a fault")
    args = ap.parse_args()
    {"train": cmd_train, "replay": cmd_replay, "live": cmd_live}[args.cmd](args)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
