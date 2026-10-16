"""
x7_presence_bridge_eval.py
==========================
Train on one dataset, test on a DIFFERENT DAY -- the only test that separates
"detects people" from "recognises the session".

v2 (Sep 11-22) has one occupancy level per day. v3 (Sep 25) has every level on
one day, in one setup. Anything trained on v2 that memorised days or
sit-downs has nothing to recognise in v3.

Evaluated, all trained on --train only:
  1. zone model      models/presence_zone.json (fitted on v2; thresholds from v2)
  2. count CNN       5-class, retrained here on v2
  3. count CNN       same, fed ONLY bins beyond 3.55 m (no seat) -- control
  4. seat CNN        models/presence_seats.pth (per-seat, day-balanced)
  5. zone control    zone features from 2.6-3.9 m (no seat), fitted on v2

Usage:
    python x7_presence_bridge_eval.py
    python x7_presence_bridge_eval.py --train human_dataset_v2 --test human_dataset_v3
"""

import os
import re
import glob
import json
import argparse
import collections

import numpy as np

import x7_presence_zone as Z

_NAME = re.compile(r"(\d+)people_([A-Za-z0-9]+)_")


def files_labels(d):
    out = []
    for f in sorted(glob.glob(os.path.join(d, "*", "*.npy"))):
        m = _NAME.match(os.path.basename(f))
        if m:
            out.append((f, int(m.group(1)), m.group(2)))
    return out


def conf_matrix(y, p, k=5):
    cm = np.zeros((k, k), int)
    for a, b in zip(y, p):
        cm[a, b] += 1
    return cm


def balanced_acc(cm):
    r = [cm[i, i] / cm[i].sum() for i in range(len(cm)) if cm[i].sum()]
    return float(np.mean(r))


# ─── 1 + 5: zone model ──────────────────────────────────────────────────────

def eval_zone(args):
    model = json.load(open(args.zone_model))
    K = model["window_blocks"]
    zones = {z: tuple(v) for z, v in model["zones_m"].items()}
    trials, _ = Z.load_trials(args.test)
    layout = Z.make_layout(model["n_bins"], zones)
    stats = [Z.block_stats(np.load(t["path"]), layout) for t in trials]
    X, Y, G, _ = Z.build_windows(trials, stats, K, layout)
    tags = {}
    for t in trials:
        tags[t["group"]] = (t["n"], t["tag"])
    from sklearn.metrics import roc_auc_score
    print(f"\n{'=' * 72}\n 1. ZONE MODEL  (fitted on v2, {K * Z.BLOCK / Z.FPS:.0f} s window, "
          f"v2 thresholds)  -> tested on {args.test}\n{'=' * 72}")
    print(f" {'output':<9}{'v2 status':<14}{'AUC':>7}{'recall':>9}{'false alarm':>13}"
          f"{'n pos/neg':>12}")
    res = {}
    for i, o in enumerate(Z.OUTPUTS):
        p = model["outputs"][o]
        pr = Z.predict_lr(p, X)
        y = Y[:, i]
        hit = pr >= p["threshold"]
        auc = roc_auc_score(y, pr) if len(np.unique(y)) == 2 else float("nan")
        rec = hit[y == 1].mean() if (y == 1).any() else float("nan")
        fa = hit[y == 0].mean() if (y == 0).any() else float("nan")
        res[o] = dict(auc=auc, recall=rec, fa=fa, pr=pr)
        status = "usable" if p.get("usable", True) else "UNRELIABLE"
        print(f" {o:<9}{status:<14}{auc:7.3f}{rec*100:8.1f}%{fa*100:12.1f}%"
              f"{int((y == 1).sum()):>6}/{int((y == 0).sum())}")
    print("\n per scenario: fraction of windows flagged  (truth in brackets)")
    print(f" {'scenario':<14}" + "".join(f"{o:>12}" for o in Z.OUTPUTS))
    for g in sorted(set(G), key=lambda g: tags[g]):
        m = G == g
        n, tag = tags[g]
        truth = Z.labels_from_tag(n, tag)
        cells = "".join(
            f"{(res[o]['pr'][m] >= model['outputs'][o]['threshold']).mean()*100:8.0f}%"
            f"[{'Y' if truth[i] else '-'}]" for i, o in enumerate(Z.OUTPUTS))
        print(f" {str(n) + 'p ' + tag:<14}{cells}")

    # control: same pipeline on a region with no seats, fitted on v2
    ctrl = Z.make_layout(model["n_bins"], {"front": (2.6, 3.2), "back": (3.2, 3.9)})
    tr_trials, _ = Z.load_trials(args.train)
    trs = [Z.block_stats(np.load(t["path"]), ctrl) for t in tr_trials]
    Xc, Yc, _, _ = Z.build_windows(tr_trials, trs, K, ctrl)
    pc = Z.fit_lr(Xc, Yc[:, 0])
    tes = [Z.block_stats(np.load(t["path"]), ctrl) for t in trials]
    Xt, Yt, _, _ = Z.build_windows(trials, tes, K, ctrl)
    auc_c = roc_auc_score(Yt[:, 0], Z.predict_lr(pc, Xt))
    print(f"\n 5. ZONE CONTROL: no-seat region 2.6-3.9 m, fitted on v2, occupied AUC on "
          f"v3 = {auc_c:.3f}")
    print("    For occupied-vs-empty this is not diagnostic: occupants shadow the walls and\n"
          "    reflect into far bins, so the region responds to people as well as to a\n"
          "    session. The model generalising to a new day's empty trials is the test.")
    res["control_auc"] = auc_c
    return res


# ─── 2-4: CNNs ──────────────────────────────────────────────────────────────

def eval_cnns(args):
    import torch
    import torch.nn as nn
    from x7_2s_train_presence import (extract_presence_features, PresenceCNN_X7_Dual,
                                      FIRST_BIN)
    from x7_physics import BIN_LENGTH_M, RANGE_OFFSET_M
    dev = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

    def feats(items):
        out = []
        for i in range(0, len(items), 64):
            a = np.stack([np.load(f) for f, _, _ in items[i:i + 64]])
            with torch.no_grad():
                out.append(extract_presence_features(torch.from_numpy(a).to(dev)).float().cpu())
        return torch.cat(out)

    tr, te = files_labels(args.train), files_labels(args.test)
    Xtr, Xte = feats(tr), feats(te)
    ytr = torch.tensor([n for _, n, _ in tr])
    yte = np.array([n for _, n, _ in te])
    rng = RANGE_OFFSET_M + (np.arange(Xtr.shape[-1]) + FIRST_BIN) * BIN_LENGTH_M
    far = torch.from_numpy(rng > 3.55)

    def mask(X):
        Zm = torch.zeros_like(X)
        Zm[..., far] = X[..., far]
        return Zm

    def train_count(X, y, seed):
        torch.manual_seed(seed)
        m = PresenceCNN_X7_Dual(n_classes=5).to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        w = 1.0 / torch.bincount(y, minlength=5).float().clamp(min=1)
        lossf = nn.CrossEntropyLoss(weight=(w / w.sum() * 5).to(dev), label_smoothing=0.05)
        for _ in range(args.epochs):
            m.train()
            perm = torch.randperm(len(X))
            for j in range(0, len(X), 32):
                b = perm[j:j + 32]
                opt.zero_grad()
                lossf(m(X[b].to(dev)), y[b].to(dev)).backward()
                opt.step()
            sch.step()
        return m

    @torch.no_grad()
    def logits(m, X):
        m.eval()
        return torch.cat([m(X[j:j + 128].to(dev)).cpu() for j in range(0, len(X), 128)])

    for name, fx in (("2. COUNT CNN, full features", lambda X: X),
                     ("3. COUNT CNN, ONLY bins > 3.55 m (control)", mask)):
        L = sum(torch.softmax(logits(train_count(fx(Xtr), ytr, s), fx(Xte)), 1)
                for s in range(args.seeds))
        pred = L.argmax(1).numpy()
        cm = conf_matrix(yte, pred)
        print(f"\n{'=' * 72}\n {name}  (trained on v2, {args.seeds} seeds)\n{'=' * 72}")
        print(f" v3 accuracy {np.mean(pred == yte)*100:.1f}%   balanced {balanced_acc(cm)*100:.1f}%"
              f"   (chance 20%)")
        print("        pred " + " ".join(f"{k}p" for k in range(5)) + "     (rows = truth)")
        for k in range(5):
            if cm[k].sum():
                print(f"   {k}p  " + " ".join(f"{v:3d}" for v in cm[k]))

    # seat CNN
    if os.path.exists(args.seat_model):
        m = PresenceCNN_X7_Dual(n_classes=4).to(dev)
        m.load_state_dict(torch.load(args.seat_model, map_location=dev, weights_only=True))
        p = torch.sigmoid(logits(m, Xte)).numpy() > 0.5
        seats = np.zeros((len(te), 4), int)
        for i, (_, n, tag) in enumerate(te):
            for ch in (tag[5:] if tag.startswith("seats") else ""):
                seats[i, int(ch) - 1] = 1
        cnt = p.sum(1)
        print(f"\n{'=' * 72}\n 4. SEAT CNN  ({args.seat_model}, trained on v2 1p-3p days)\n{'=' * 72}")
        print(f" per-seat accuracy " + "  ".join(f"s{i+1} {(p[:, i] == seats[:, i]).mean()*100:.0f}%"
                                              for i in range(4)))
        print(f" count accuracy {np.mean(cnt == yte)*100:.1f}%   predicted counts "
              f"{dict(sorted(collections.Counter(cnt.tolist()).items()))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="human_dataset_v2")
    ap.add_argument("--test", default="human_dataset_v3")
    ap.add_argument("--zone-model", dest="zone_model",
                    default=os.path.join("models", "presence_zone.json"))
    ap.add_argument("--seat-model", dest="seat_model",
                    default=os.path.join("models", "presence_seats.pth"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-cnn", action="store_true")
    args = ap.parse_args()
    eval_zone(args)
    if not args.no_cnn:
        eval_cnns(args)


if __name__ == "__main__":
    main()
