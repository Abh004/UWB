"""
x7_2s_train_presence_seats.py
=============================
Occupancy by SEAT, not by count -- a workaround for day-aliased data.

Why
---
Every occupancy level in human_dataset_v2 was recorded on its own day
(4p 09-11, 3p 09-15, 2p 09-16, 1p 09-21, 0p 09-22). A count classifier can
therefore score ~100% by recognising the day, and no split of that data can
tell the two apart.

Seat occupancy is different. Within the 2p day, seat 1 is occupied in
seats12/13/14 and empty in seats23/24/34 -- same sitting, both labels. So each
seat's label varies WITHIN the 1p, 2p and 3p days, and with per-day class
re-weighting (each seat 50/50 inside each day) the day carries zero
information about any seat label. The count is then the sum of the four seat
decisions.

That also makes a test possible that a count classifier cannot pass: train on
the 1p/2p/3p days only, then evaluate on the 0p and 4p days -- unseen days AND
unseen counts. Recognising the day cannot produce "4" on a day never seen.

The 0p and 4p days are never trained on: every seat has one constant label all
day there, which would reintroduce the alias.

Folds (leave-one-day-out):
    train {1p,2p} days      -> test 3p day
    train {1p,3p} days      -> test 2p day
    train {2p,3p} days      -> test 1p day
    train {1p,2p,3p} days   -> test 0p day and 4p day   (unseen counts)

Plus a control: the same pipeline fed ONLY bins beyond 2.5 m (no seat there).

Usage:
    python x7_2s_train_presence_seats.py
    python x7_2s_train_presence_seats.py --data human_dataset_v2 --epochs 30 --seeds 3
"""

import os
import re
import glob
import json
import argparse
import datetime

import numpy as np
import torch
import torch.nn as nn

from x7_2s_train_presence import (extract_presence_features, PresenceCNN_X7_Dual,
                                  FIRST_BIN)
from x7_physics import BIN_LENGTH_M, RANGE_OFFSET_M

N_SEATS = 4


def parse(path):
    b = os.path.basename(path)
    m = re.match(r"(\d+)people_([A-Za-z0-9]+)_(.+?)_(\d{10})_", b)
    if not m:
        return None
    n, tag, _, ts = m.groups()
    seats = np.zeros(N_SEATS, np.float32)
    if tag.startswith("seats"):
        for ch in tag[5:]:
            seats[int(ch) - 1] = 1.0
    if int(seats.sum()) != int(n):
        raise ValueError(f"{b}: tag {tag} disagrees with count {n}")
    day = datetime.datetime.fromtimestamp(int(ts)).strftime("%m-%d")
    return int(n), seats, day


def load(data_dir, device):
    files = sorted(glob.glob(os.path.join(data_dir, "*", "*.npy")))
    meta = [(f, parse(f)) for f in files]
    meta = [(f, p) for f, p in meta if p]
    feats = []
    for i in range(0, len(meta), 64):
        chunk = np.stack([np.load(f) for f, _ in meta[i:i + 64]])
        with torch.no_grad():
            x = extract_presence_features(torch.from_numpy(chunk).to(device))
        feats.append(x.float().cpu())
    X = torch.cat(feats)                                    # (N, 10, 50, 92)
    count = np.array([p[0] for _, p in meta])
    seats = np.stack([p[1] for _, p in meta])
    day = np.array([p[2] for _, p in meta])
    return X, count, seats, day


def day_balanced_weights(seats, day):
    """Per-sample, per-seat weights making every seat 50/50 inside every day."""
    w = np.zeros_like(seats)
    for d in np.unique(day):
        m = day == d
        for s in range(N_SEATS):
            y = seats[m, s]
            pos = y.sum()
            neg = len(y) - pos
            if pos == 0 or neg == 0:        # constant within day: unusable
                continue
            w[m, s] = np.where(y == 1, 0.5 / pos, 0.5 / neg) * len(y)
    return w


def train_fold(X, seats, w, idx, device, epochs, seed, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = PresenceCNN_X7_Dual(n_classes=N_SEATS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    Xt, Yt, Wt = X[idx], torch.from_numpy(seats[idx]), torch.from_numpy(w[idx])
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(idx))
        for j in range(0, len(idx), 32):
            b = perm[j:j + 32]
            xb = Xt[b].to(device)
            # small time-shift jitter: the window start is arbitrary
            xb = torch.roll(xb, shifts=int(np.random.randint(0, 50)), dims=2)
            loss = (bce(model(xb), Yt[b].to(device)) * Wt[b].to(device)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return model


@torch.no_grad()
def predict(model, X, idx, device):
    model.eval()
    out = []
    for j in range(0, len(idx), 128):
        out.append(torch.sigmoid(model(X[idx[j:j + 128]].to(device))).cpu())
    return torch.cat(out).numpy()


def mask_far_only(X):
    """Zero every bin up to 2.5 m: nothing that can hold a seat survives."""
    rng = RANGE_OFFSET_M + (np.arange(X.shape[-1]) + FIRST_BIN) * BIN_LENGTH_M
    keep = torch.from_numpy(rng > 2.5)
    Z = torch.zeros_like(X)
    Z[..., keep] = X[..., keep]
    return Z


def run(X, count, seats, day, day_of, device, epochs, seeds, label):
    folds = [
        ([day_of[1], day_of[2]], [day_of[3]]),
        ([day_of[1], day_of[3]], [day_of[2]]),
        ([day_of[2], day_of[3]], [day_of[1]]),
        ([day_of[1], day_of[2], day_of[3]], [day_of[0], day_of[4]]),
    ]
    w = day_balanced_weights(seats, day)
    print(f"\n{'=' * 72}\n {label}\n{'=' * 72}")
    print(f" {'trained on':<24}{'tested on':<10}{'n':>5}{'seat acc':>10}"
          f"{'count acc':>11}   predicted counts")
    summary = []
    for tr_days, te_days in folds:
        tr = np.where(np.isin(day, tr_days))[0]
        probs = []
        for sd in range(seeds):
            m = train_fold(X, seats, w, tr, device, epochs, sd)
            probs.append({d: predict(m, X, np.where(day == d)[0], device)
                          for d in te_days})
        for d in te_days:
            te = np.where(day == d)[0]
            p = np.mean([pr[d] for pr in probs], axis=0)
            hard = (p > 0.5).astype(int)
            seat_acc = (hard == seats[te]).mean()
            cnt = hard.sum(1)
            cnt_acc = (cnt == count[te]).mean()
            hist = np.bincount(cnt, minlength=5)
            trn = "+".join(f"{count[day == t][0]}p" for t in tr_days)
            print(f" {trn + ' days':<24}{str(count[te][0]) + 'p day':<10}{len(te):>5}"
                  f"{seat_acc*100:>9.1f}%{cnt_acc*100:>10.1f}%   "
                  + " ".join(f"{k}:{v}" for k, v in enumerate(hist)))
            summary.append((count[te][0], seat_acc, cnt_acc))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="human_dataset_v2")
    ap.add_argument("--out", default="models")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--no-control", action="store_true",
                    help="skip the far-field-only control run")
    ap.add_argument("--cpu", action="store_true",
                    help="never touch CUDA. Needed when the NVIDIA driver is "
                         "down: torch.cuda.is_available() itself segfaults then")
    args = ap.parse_args()

    if args.cpu:
        device = torch.device("cpu")
        torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, count, seats, day = load(args.data, device)

    day_of = {}
    for d in np.unique(day):
        c = np.unique(count[day == d])
        if len(c) != 1:
            raise SystemExit(f"day {d} mixes counts {c}: this dataset is already "
                             "interleaved, use a normal session-held-out split")
        day_of[int(c[0])] = d
    missing = [c for c in range(5) if c not in day_of]
    if missing:
        raise SystemExit(f"need one day per count 0-4, missing {missing}")

    print(f"{len(X)} recordings on device {device}")
    for c in range(5):
        print(f"   {c}p  day {day_of[c]}  n={int((count == c).sum())}")

    real = run(X, count, seats, day, day_of, device, args.epochs, args.seeds,
               "SEAT MODEL, full features")
    if not args.no_control:
        run(mask_far_only(X), count, seats, day, day_of, device, args.epochs,
            args.seeds, "CONTROL: only bins beyond 2.5 m (no seat can be there)")

    # final checkpoint: all de-aliased days (1p/2p/3p) -- never the 0p/4p days
    tr = np.where(np.isin(day, [day_of[1], day_of[2], day_of[3]]))[0]
    w = day_balanced_weights(seats, day)
    m = train_fold(X, seats, w, tr, device, args.epochs, seed=0)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "presence_seats.pth")
    torch.save(m.state_dict(), path)
    with open(os.path.join(args.out, "presence_seats.json"), "w") as f:
        json.dump({"type": "seats", "n_seats": N_SEATS, "threshold": 0.5,
                   "trained_on_days": [day_of[1], day_of[2], day_of[3]],
                   "held_out_results": [
                       {"count": int(c), "seat_acc": float(s), "count_acc": float(a)}
                       for c, s, a in real]}, f, indent=2)
    print(f"\nsaved {path}  (trained on the 1p/2p/3p days only)")


if __name__ == "__main__":
    main()
