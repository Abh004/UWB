"""
x7_gesture_train.py
===================
Train the compact gesture model on windows recorded with x7_gesture_record.py,
evaluated ONLY on held-out sessions.

The old training script scored a random per-file split: on the 4-subject corpus
that reported 87.4% while a new person got 73.7%. Here:
  * >= 2 sessions  -> leave-one-session-out (the number that predicts live use)
  * 1 session      -> time-ordered 70/30 split, clearly labelled OPTIMISTIC
The confidence and margin floors are then chosen on the held-out predictions to
reach --target-precision, counting a NO_GESTURE movement reported as a gesture
as a false prediction.

Usage:
    python x7_gesture_train.py                                  # gesture_dataset_v3
    python x7_gesture_train.py --data gesture_dataset_v3 --target-precision 0.95
    python x7_gesture_train.py --raw-corpus DIR --out models/_corpus_check.json   # design check
"""

import os
import re
import glob
import json
import random
import argparse
import datetime
import collections

import numpy as np
import torch
import torch.nn as nn

import x7_gesture_core as C

_NAME = re.compile(r"^(" + "|".join(C.GESTURES) + r")_(.+?)_(\d{10})_(\d+)\.npy$")
_CORPUS = {"G01_SWIPE_LR": "SWIPE_LR", "G02_SWIPE_RL": "SWIPE_RL", "G03_SWIPE_UD": "SWIPE_UD",
           "G04_SWIPE_DU": "SWIPE_DU", "G05_PUSH_IN": "PUSH_IN"}


def load_windows(dirs):
    items = []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.npy"))):
            m = _NAME.match(os.path.basename(f))
            if not m:
                continue
            w = np.load(f)
            if w.shape != (C.WINDOW, 2, 2, 2, C.N_BINS):
                print(f"[WARN] skipping {f}: shape {w.shape}")
                continue
            items.append(dict(w=w, label=m.group(1), session=m.group(2), ts=int(m.group(3))))
    return items


def load_raw_corpus(root):
    """Old 100-frame corpus recordings -> 64-frame windows centred on the peak of
    the same moving energy the live trigger uses. Session = subject."""
    items = []
    for folder, label in _CORPUS.items():
        for f in sorted(glob.glob(os.path.join(root, folder, "*.npy"))):
            a = np.load(f)
            bg = np.median(a.real, 0) + 1j * np.median(a.imag, 0)
            e = (np.abs((a - bg)[..., C.PHYS]) ** 2).mean(axis=(1, 2, 3, 4))
            pk = int(np.argmax(np.convolve(e, np.ones(5), "same")))
            s = max(0, min(len(a) - C.WINDOW, pk - C.WINDOW // 2))
            b = os.path.basename(f)
            items.append(dict(w=a[s:s + C.WINDOW], label=label, session="S" + b.split("_")[0],
                              ts=int(re.search(r"_(\d{10})", b).group(1))))
    return items


def augment(w, rng):
    """Raw-window augmentation: timing and pace vary between people and days."""
    shift = rng.integers(-6, 7)
    idx = np.clip(np.arange(C.WINDOW) - shift, 0, C.WINDOW - 1)
    rate = rng.uniform(0.8, 1.25)
    idx = np.clip(((idx - C.WINDOW // 2) * rate + C.WINDOW // 2).astype(int), 0, C.WINDOW - 1)
    w = w[idx]
    k = rng.integers(-1, 2)                                 # standoff +-7 cm
    if k:
        w = np.roll(w, k, axis=-1)
    sigma = 0.05 * np.std(w - w.mean(0, keepdims=True))
    return w + (rng.normal(0, sigma, w.shape) + 1j * rng.normal(0, sigma, w.shape)).astype(np.complex64)


def fit(items, classes, channels, epochs, seed, device, init=None, lr=2e-3):
    """Train from scratch, or fine-tune from `init` (live session calibration)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    model = C.build_model(C.n_channels(channels), len(classes)).to(device)
    if init is not None:
        model.load_state_dict(init.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    y_all = np.array([classes.index(it["label"]) for it in items])
    w_cls = np.bincount(y_all, minlength=len(classes)).astype(float)
    w_cls = torch.tensor(w_cls.sum() / (len(classes) * np.maximum(w_cls, 1)), dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=w_cls.to(device), label_smoothing=0.05)
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(items))
        for j in range(0, len(order), 32):
            b = order[j:j + 32]
            x = torch.from_numpy(np.stack([C.features(augment(items[i]["w"], rng), channels)
                                           for i in b])).to(device)
            y = torch.from_numpy(y_all[b]).to(device)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
        sch.step()
    return model


@torch.no_grad()
def predict(models, items, channels, device):
    x = torch.from_numpy(np.stack([C.features(it["w"], channels) for it in items])).to(device)
    return sum(torch.softmax(m.eval()(x), 1) for m in models).cpu().numpy() / len(models)


MIN_RECALL = 0.15


def choose_thresholds(P, y, classes, target):
    """Maximise gesture recall subject to precision >= target. If the target
    cannot be reached, maximise PRECISION among floors that still report at
    least MIN_RECALL of the gestures. (The first version fell back to
    maximising recall, which picked the lowest floor, 0.30, and let the
    low-confidence wrong predictions through.)"""
    ng = classes.index("NO_GESTURE") if "NO_GESTURE" in classes else None
    pred = P.argmax(1)
    srt = np.sort(P, 1)
    conf, margin = srt[:, -1], srt[:, -1] - srt[:, -2]
    is_g = (y != ng) if ng is not None else np.ones(len(y), bool)
    best = None
    for c in np.arange(0.30, 0.96, 0.05):
        for mg in np.arange(0.0, 0.51, 0.05):
            rep = (conf >= c) & (margin >= mg) & ((pred != ng) if ng is not None else True)
            if not rep.any():
                continue
            prec = np.mean(pred[rep] == y[rep])
            rec = np.sum(rep & (pred == y) & is_g) / max(1, is_g.sum())
            if prec >= target:
                cand = (2, rec, prec, -c, -mg)
            elif rec >= MIN_RECALL:
                cand = (1, prec, rec, -c, -mg)
            else:
                cand = (0, rec, prec, -c, -mg)
            if best is None or cand > best[0]:
                best = (cand, float(c), float(mg), float(prec), float(rec))
    _, c, mg, prec, rec = best
    return {"confidence": round(c, 2), "margin": round(mg, 2),
            "precision": prec, "recall": rec, "target_met": prec >= target}


def floor_table(P, y, classes):
    ng = classes.index("NO_GESTURE") if "NO_GESTURE" in classes else None
    pred = P.argmax(1)
    conf = P.max(1)
    is_g = (y != ng) if ng is not None else np.ones(len(y), bool)
    print("   confidence floor -> precision of what is shown / share of gestures shown correctly")
    for c in (0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
        rep = (conf >= c) & ((pred != ng) if ng is not None else True)
        prec = np.mean(pred[rep] == y[rep]) * 100 if rep.any() else float("nan")
        rec = np.sum(rep & (pred == y) & is_g) / max(1, is_g.sum()) * 100
        print(f"      {c:.2f}: precision {prec:5.1f}%   recall {rec:5.1f}%   (shown {rep.sum()})")


def report(P, y, classes, th, title):
    ng = classes.index("NO_GESTURE") if "NO_GESTURE" in classes else None
    pred = P.argmax(1)
    srt = np.sort(P, 1)
    ok = (srt[:, -1] >= th["confidence"]) & (srt[:, -1] - srt[:, -2] >= th["margin"])
    shown = np.where(ok & ((pred != ng) if ng is not None else True), pred, -1)
    print(f"\n{title}")
    print(f"   raw accuracy (argmax, no rejection): {np.mean(pred == y) * 100:5.1f}%")
    print(f"   with floors conf >= {th['confidence']:.2f}, margin >= {th['margin']:.2f}:"
          f" precision {th['precision'] * 100:5.1f}%, gesture recall {th['recall'] * 100:5.1f}%")
    print("   rows = truth; last column = nothing reported")
    print("   " + " " * 12 + "".join(f"{c[:8]:>9}" for c in classes) + "    (none)")
    for i, c in enumerate(classes):
        r = [(shown[y == i] == j).sum() for j in range(len(classes))] + [(shown[y == i] == -1).sum()]
        print(f"   {c:<12}" + "".join(f"{v:9d}" for v in r))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", default=["gesture_dataset_v4_pause"])
    ap.add_argument("--raw-corpus", dest="raw_corpus", default=None,
                    help="also/only use old 100-frame corpus recordings (dir with G0x_* folders)")
    ap.add_argument("--pretrain-corpus", dest="pretrain_corpus", default=None,
                    help="add old corpus recordings to the TRAINING side of every fold "
                         "(they are never held out or scored)")
    # angle + amplitude. On the old corpus (a different mounting) angle alone
    # transferred best, but on gesture_dataset_v3 (current mounting,
    # leave-one-session-out): angle 40.6%, angle+amp 46.1%, +time reversal
    # 53.9%; angle+amp+doppler 51.6% / 45.3%. Amplitude carries which board the
    # hand reaches first -- the strongest left/right cue at this mounting.
    ap.add_argument("--channels", default="angle+amp", choices=list(C.CHANNEL_SETS))
    ap.add_argument("--no-reverse", dest="reverse", action="store_false",
                    help="disable time-reversal augmentation (on by default)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--target-precision", dest="target", type=float, default=0.95)
    ap.add_argument("--out", default=os.path.join("models", "gesture_v4_pause.json"))
    args = ap.parse_args()

    items = [] if args.raw_corpus and args.data == ["gesture_dataset_v4_pause"] and \
        not os.path.isdir("gesture_dataset_v4_pause") else load_windows(args.data)
    if args.raw_corpus:
        items += load_raw_corpus(args.raw_corpus)
    if not items:
        raise SystemExit(f"no recordings in {args.data}")
    extra = load_raw_corpus(args.pretrain_corpus) if args.pretrain_corpus else []
    if extra:
        print(f"[INFO] {len(extra)} corpus windows added to training only")
    channels = C.CHANNEL_SETS[args.channels]
    classes = [g for g in C.GESTURES if any(it["label"] == g for it in items)]
    sessions = sorted({it["session"] for it in items})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnt = collections.Counter((it["session"], it["label"]) for it in items)
    print(f"{len(items)} windows, {len(sessions)} session(s), classes {classes}, "
          f"channels {channels}, device {device}")
    print("   session     " + "".join(f"{c[:8]:>9}" for c in classes))
    for s in sessions:
        print(f"   {s:<12}" + "".join(f"{cnt[(s, c)]:9d}" for c in classes))
    missing = [g for g in C.GESTURES if g not in classes]
    if missing:
        print(f"[WARN] no recordings of {missing}")

    y = np.array([classes.index(it["label"]) for it in items])
    P = np.zeros((len(items), len(classes)))
    if len(sessions) >= 2:
        mode = "leave-one-session-out"
        for s in sessions:
            te = [i for i, it in enumerate(items) if it["session"] == s]
            tr = [i for i, it in enumerate(items) if it["session"] != s]
            tr_items = [items[i] for i in tr]
            tr_items += (C.time_reversed(tr_items) if args.reverse else []) + extra
            ms = [fit(tr_items, classes, channels, args.epochs, sd, device)
                  for sd in range(args.seeds)]
            P[te] = predict(ms, [items[i] for i in te], channels, device)
            acc = np.mean(P[te].argmax(1) == y[te])
            print(f"   held out {s:<12} raw accuracy {acc * 100:5.1f}%  (n={len(te)})", flush=True)
    else:
        mode = "time-ordered 70/30 within ONE session -- OPTIMISTIC, record a second session"
        order = sorted(range(len(items)), key=lambda i: items[i]["ts"])
        k = int(0.7 * len(order))
        tr, te = order[:k], order[k:]
        tr_items = [items[i] for i in tr]
        tr_items += C.time_reversed(tr_items) if args.reverse else []
        ms = [fit(tr_items, classes, channels, args.epochs, sd, device)
              for sd in range(args.seeds)]
        P[te] = predict(ms, [items[i] for i in te], channels, device)
        keep = np.zeros(len(items), bool)
        keep[te] = True
        P, y_eval = P[keep], y[keep]
        y = y_eval

    th = choose_thresholds(P, y, classes, args.target)
    report(P, y, classes, th, f"EVALUATION ({mode})")
    floor_table(P, y, classes)
    np.savez(os.path.splitext(args.out)[0] + "_heldout.npz", P=P, y=y, classes=classes)
    if not th["target_met"]:
        print(f"[WARN] target precision {args.target:.0%} not reachable; floors give the best "
              f"precision found. More data (or another session) is needed.")

    print(f"\ntraining the final {args.seeds}-model ensemble on all data ...", flush=True)
    base, _ = os.path.splitext(args.out)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    weights = []
    for sd in range(args.seeds):
        m = fit(items + (C.time_reversed(items) if args.reverse else []) + extra,
                classes, channels, args.epochs, sd, device)
        p = f"{os.path.basename(base)}_seed{sd}.pt"
        torch.save(m.state_dict(), os.path.join(os.path.dirname(args.out) or ".", p))
        weights.append(p)
    n_par = sum(p.numel() for p in m.parameters())
    meta = {"created": datetime.datetime.now().isoformat(timespec="seconds"),
            "classes": classes, "channels": channels, "weights": weights,
            "thresholds": {"confidence": th["confidence"], "margin": th["margin"]},
            "evaluation": {"mode": mode, "precision": th["precision"], "recall": th["recall"],
                           "raw_accuracy": float(np.mean(P.argmax(1) == y)),
                           "target_precision": args.target},
            "sessions": sessions, "counts": {c: int((np.array([it["label"] for it in items]) == c).sum())
                                             for c in classes},
            "trigger_x": C.TRIGGER_X, "parameters": int(n_par), "data": args.data,
            "pretrain_corpus": args.pretrain_corpus, "reverse": bool(args.reverse)}
    json.dump(meta, open(args.out, "w"), indent=2)
    print(f"saved {args.out}  ({args.seeds} models x {n_par:,} parameters)")


if __name__ == "__main__":
    main()
