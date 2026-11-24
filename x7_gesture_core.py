"""
x7_gesture_core.py
==================
One implementation of triggering, windowing, features and the model, shared by
the recorder, the trainer and live detection -- so the model is trained on
exactly what live detection feeds it.

Why a new core (measured 2026-09-28/29, see documentation/2026-09-28.md)
-------------------------------------------------------------------------
* Each X7 board is a 3-element LINE array: its TX and RX baselines are
  parallel, so it measures angle along ONE axis only (the two cross virtual
  elements moved 0.03 rad while a single baseline swung 0.6-1.0 rad). The old
  "horizontal" and "vertical" AoA channels were the same measurement. Here the
  four single-baseline pairs are averaged coherently into one angle phasor.
* The deployed ensemble scored 7% on captures from the current mounting --
  systematically remapped (mirroring sensor 1 alone lifted it to 46%). The fix
  is data from the current mounting, which this pipeline records.
* The old trigger summed RAW energy over all 34 bins, 20 of which are static
  coupling behind the antenna: a gesture rose only 7.8x (5th pct 3.2x) above
  the quiet level. Moving energy in the hand's bins, relative to the noise
  floor behind the antenna, rises 88x (5th pct 13x).
* Only the physical bins (20-33, 0.05-0.98 m) are used; the hand sits at
  0.33-0.90 m. On the old corpus angle alone transferred best; at the CURRENT
  mounting amplitude adds the strongest left/right cue (which board the hand
  reaches first), so the default input is angle + amplitude
  (x7_gesture_train.py --channels; see the comparison there).

Timing is unchanged from x7_2s_live.py: 40 FPS, 64-frame window, peak-driven
capture (DECAY_FRAMES / POST_PEAK_FRAMES), same cooldowns.
"""

import collections
import json
import os

import numpy as np

FPS = 40
WINDOW = 64                       # frames fed to the model (1.6 s)
BUFFER = 200                      # ring buffer; must exceed WINDOW + MAX_GATHER
N_BINS = 34                       # bins delivered by default_preset.json
PHYS = np.arange(20, 34)          # hand region; 0-19 are behind the antenna
REF = np.arange(0, 16)            # noise-floor reference (behind the antenna)

WARMUP_FRAMES = 120               # 3 s: learn the background and the idle level
BG_TAU_S = 2.0                    # background (static clutter) time constant
TRIGGER_X = 6.0                   # trigger at this multiple of the idle level
REARM_RATIO = 0.5                 # re-arm below this fraction of the trigger
DECAY_FRAMES = 6                  # consecutive falling frames confirming the peak
POST_PEAK_FRAMES = 40             # frames needed after the peak
MAX_GATHER_FRAMES = 110           # hard stop for a noisy plateau
COOLDOWN_FRAMES = 20              # silence after a capture
IDLE_WINDOW_FRAMES = 1200         # 30 s of history for the idle level
IDLE_PERCENTILE = 20              # the quietest 20% of that history
IDLE_UPDATE_FRAMES = 40           # re-estimate once a second

GESTURES = ["SWIPE_LR", "SWIPE_RL", "SWIPE_UD", "SWIPE_DU", "PUSH_IN", "NO_GESTURE"]
DESCRIPTION = {
    "SWIPE_LR": "hand still on your LEFT, pause, then swipe LEFT -> RIGHT",
    "SWIPE_RL": "hand still on your RIGHT, pause, then swipe RIGHT -> LEFT",
    "SWIPE_UD": "swipe UP -> DOWN",
    "SWIPE_DU": "swipe DOWN -> UP",
    "PUSH_IN": "PUSH toward the radar",
    "NO_GESTURE": "NOT a gesture: reach for something, shift, walk past, "
                  "move your hand in and out -- anything but the five gestures",
}
# Time reversal: frames played backwards turn a left->right swipe into a
# right->left one (and up->down into down->up), so every swipe also yields a
# valid example of its opposite. PUSH_IN reversed is a pull, not a class, and
# NO_GESTURE reversed is still ambiguous, so neither is mirrored. Measured on
# the current mounting (leave-one-session-out): 46.1% -> 53.9%.
# Where the hand waits before a manually armed capture. Moving into this
# position is itself a movement (for SWIPE_RL the hand travels left -> right
# to get there), so it happens BEFORE Enter, while nothing is captured.
START_HINT = {"SWIPE_LR": "put your hand on your LEFT", "SWIPE_RL": "put your hand on your RIGHT",
              "SWIPE_UD": "put your hand UP", "SWIPE_DU": "put your hand DOWN",
              "PUSH_IN": "put your hand back, ~0.5 m from the radar",
              "NO_GESTURE": "get ready for a non-gesture movement"}

REVERSE_OF = {"SWIPE_LR": "SWIPE_RL", "SWIPE_RL": "SWIPE_LR",
              "SWIPE_UD": "SWIPE_DU", "SWIPE_DU": "SWIPE_UD"}


def time_reversed(items):
    return [dict(it, w=np.ascontiguousarray(it["w"][::-1]), label=REVERSE_OF[it["label"]])
            for it in items if it["label"] in REVERSE_OF]


CHANNEL_SETS = {"angle": ["angle"], "angle+amp": ["amp", "angle"],
                "angle+amp+doppler": ["amp", "angle", "doppler"]}
EPS = 1e-12


# ─── triggering and windowing ───────────────────────────────────────────────

class Segmenter:
    """Frame in, gesture window out.

    push(frame) returns None, or a dict with the 64-frame raw window
    (64, 2, 2, 2, 34) complex64 once a movement has been captured.
    """

    def __init__(self, trigger_x=TRIGGER_X):
        self.trigger_x = trigger_x
        self.alpha = 1.0 / (BG_TAU_S * FPS)
        self.bg = None
        self.frames = collections.deque(maxlen=BUFFER)
        self.energy = collections.deque(maxlen=BUFFER)      # per-frame moving energy
        self.ref = collections.deque(maxlen=400)
        self.q5 = collections.deque(maxlen=5)
        self.idle = []
        self.n = 0
        self.level0 = None         # idle level measured in the warm-up
        self.level = None          # idle level of the 5-frame trigger energy
        self.threshold = None
        self.state = "WARMUP"
        self.armed = True
        self.cooldown = 0
        self.peak = -1.0
        self.since_peak = 0
        self.falling = 0
        self.gathered = 0
        self.value = 0.0
        # The idle level keeps being re-estimated from the quietest part of the
        # last 30 s. A warm-up spoiled by movement (sitting down, adjusting)
        # otherwise fixes the trigger too high for the whole run, and smaller
        # gestures stop clearing it.
        self.recent = collections.deque(maxlen=IDLE_WINDOW_FRAMES)

    @property
    def status(self):
        if self.state == "WARMUP":
            return f"warming up -- keep still ({(WARMUP_FRAMES - self.n) / FPS:.0f} s)"
        return self.state

    def push(self, frame):
        f = np.asarray(frame, dtype=np.complex64)
        self.n += 1
        if self.bg is None:
            self.bg = f.copy()
        m = f - self.bg
        # static clutter adapts slowly; a moving hand does not get absorbed
        self.bg = (1 - self.alpha) * self.bg + self.alpha * f
        e = float((np.abs(m[..., PHYS]) ** 2).mean())
        self.ref.append(float((np.abs(m[..., REF]) ** 2).mean()))
        q = e / (np.median(self.ref) + EPS)
        self.frames.append(f)
        self.energy.append(q)
        self.q5.append(q)
        self.value = float(sum(self.q5))
        if len(self.q5) == 5:
            self.recent.append(self.value)
        if self.level is not None and self.n % IDLE_UPDATE_FRAMES == 0 and len(self.recent) >= 200:
            # only ever LOWER it: frequent gestures must not push the trigger up
            self.level = min(self.level0, float(np.percentile(self.recent, IDLE_PERCENTILE)))
            self.threshold = self.trigger_x * self.level

        if self.state == "WARMUP":
            if self.n > 20:                    # let the background settle first
                self.idle.append(self.value)
            if self.n >= WARMUP_FRAMES:
                self.level = self.level0 = float(np.percentile(self.idle, IDLE_PERCENTILE))
                self.threshold = self.trigger_x * self.level
                self.state = "LISTENING"
            return None
        if self.cooldown > 0:
            self.cooldown -= 1
            return None
        if not self.armed and self.value < REARM_RATIO * self.threshold:
            self.armed = True
        if self.state == "LISTENING":
            if self.armed and self.value > self.threshold and len(self.frames) >= WINDOW:
                self.state = "GATHERING"
                self.peak, self.since_peak, self.falling, self.gathered = self.value, 0, 0, 0
            return None

        # GATHERING: follow the energy until the movement has peaked
        self.gathered += 1
        if self.value > self.peak:
            self.peak, self.since_peak, self.falling = self.value, 0, 0
        else:
            self.since_peak += 1
            self.falling += 1
        if not ((self.falling >= DECAY_FRAMES and self.since_peak >= POST_PEAK_FRAMES)
                or self.gathered >= MAX_GATHER_FRAMES):
            return None
        buf = np.asarray(self.frames)
        en = np.asarray(self.energy)
        span0 = max(0, len(buf) - (self.gathered + WINDOW // 2))
        pk = span0 + int(np.argmax(en[span0:]))
        s = max(0, min(len(buf) - WINDOW, pk - WINDOW // 2))
        out = {"window": buf[s:s + WINDOW].copy(), "peak_energy": self.peak / self.level,
               "gathered": self.gathered}
        self.state = "LISTENING"
        self.armed = False
        self.cooldown = COOLDOWN_FRAMES
        return out

    def meter(self):
        """One-line trigger meter: current movement energy vs the trigger."""
        if self.level is None:
            return self.status
        return (f"energy {self.value / self.level:5.1f}x idle / trigger {self.trigger_x:g}x"
                f"{'  (capturing)' if self.state == 'GATHERING' else ''}")

    def reset_capture(self, cooldown=COOLDOWN_FRAMES):
        """After a pause (e.g. an operator prompt): drop stale frames, keep the
        learned background and idle level. cooldown=0 arms at once (manual
        trigger: the operator has just pressed Enter with the hand still)."""
        self.state = "LISTENING" if self.level is not None else "WARMUP"
        self.cooldown = cooldown
        self.armed = True
        self.q5.clear()


def wait_for_enter(radar, seg, label, keypress):
    """Manual trigger: keep reading frames (the background stays current) but
    capture nothing until Enter. Returns "enter", or "x" if x was pressed."""
    import sys
    import time
    print(f"        {START_HINT[label]}, hold still, then press ENTER and do it")
    last_show = 0.0
    while True:
        k = keypress()
        if k in ("\r", "\n"):
            seg.reset_capture(cooldown=0)
            return "enter"
        if k == "x":
            return "x"
        fr = radar.get_frame(timeout=0.5) if hasattr(radar, "_proc") else radar.get_frame()
        if fr is not None:
            seg.push(fr)
        now = time.time()
        if now - last_show > 0.25:
            lvl = f"movement {seg.value / seg.level:5.1f}x idle" if seg.level else seg.status
            sys.stdout.write(f"\r        waiting for ENTER (nothing is captured yet)   {lvl}      ")
            sys.stdout.flush()
            last_show = now


# ─── features ───────────────────────────────────────────────────────────────

def features(w, channels=("amp", "angle")):
    """(64, 2, 2, 2, 34) complex -> (2 sensors, C, 14 bins, 64 frames) float32.

    angle   : coherent average of the four single-baseline pairs along the
              board's one axis (TX0RX0-TX1RX0, TX0RX1-TX1RX1, TX0RX0-TX0RX1,
              TX1RX0-TX1RX1), as a unit phasor (Re, Im)
    amp     : MTI amplitude, z-scored within the window
    doppler : inter-frame phasor of TX0RX0 (Re, Im); aliases above 0.41 m/s
    """
    x = np.asarray(w)[..., PHYS]
    x = x - x.mean(axis=0, keepdims=True)                 # MTI over the window
    out = []
    for s in range(x.shape[1]):
        v = x[:, s]                                       # (T, tx, rx, bins)
        chans = []
        for c in channels:
            if c == "amp":
                a = np.abs(v).mean(axis=(1, 2))
                chans.append((a - a.mean()) / (a.std() + EPS))
            elif c == "angle":
                z = (v[:, 0, 0] * np.conj(v[:, 1, 0]) + v[:, 0, 1] * np.conj(v[:, 1, 1])
                     + v[:, 0, 0] * np.conj(v[:, 0, 1]) + v[:, 1, 0] * np.conj(v[:, 1, 1]))
                z = z / (np.abs(z) + EPS)
                chans += [z.real, z.imag]
            elif c == "doppler":
                d = np.zeros_like(v[:, 0, 0])
                d[1:] = v[1:, 0, 0] * np.conj(v[:-1, 0, 0])
                d = d / (np.abs(d) + EPS)
                chans += [d.real, d.imag]
            else:
                raise ValueError(f"unknown channel {c}")
        out.append(np.stack(chans, 0).transpose(0, 2, 1))   # (C, bins, T)
    return np.stack(out, 0).astype(np.float32)


def n_channels(channels):
    return sum(2 if c in ("angle", "doppler") else 1 for c in channels)


# ─── model ──────────────────────────────────────────────────────────────────

def build_model(n_ch, n_classes):
    """Compact two-branch CNN (~114 k parameters, vs 11.6 M before): one
    branch per board, because the boards are mounted differently."""
    import torch.nn as nn

    def branch():
        return nn.Sequential(
            nn.Conv2d(n_ch, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(True),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 8)), nn.Flatten())

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.b1, self.b2 = branch(), branch()
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * 64 * 8, 64),
                                      nn.ReLU(True), nn.Dropout(0.2),
                                      nn.Linear(64, n_classes))

        def forward(self, x):                            # (B, 2, C, bins, T)
            import torch
            return self.head(torch.cat([self.b1(x[:, 0]), self.b2(x[:, 1])], 1))

    return Net()


# ─── model bundle ───────────────────────────────────────────────────────────

class GestureModel:
    """Ensemble + metadata written by x7_gesture_train.py."""

    def __init__(self, meta_path, device=None):
        import torch
        self.meta = json.load(open(meta_path))
        self.classes = self.meta["classes"]
        self.channels = self.meta["channels"]
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base = os.path.dirname(meta_path)
        self.models = []
        for p in self.meta["weights"]:
            m = build_model(n_channels(self.channels), len(self.classes)).to(self.device)
            m.load_state_dict(torch.load(os.path.join(base, p), map_location=self.device,
                                         weights_only=True))
            m.eval()
            self.models.append(m)

    def probs(self, window):
        import torch
        import torch.nn.functional as F
        x = torch.from_numpy(features(window, self.channels)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            p = sum(F.softmax(m(x), 1)[0] for m in self.models) / len(self.models)
        return p.cpu().numpy()

    def decide(self, p):
        """-> (label or None, confidence, reason)"""
        i = int(np.argmax(p))
        srt = np.sort(p)
        conf, margin = float(srt[-1]), float(srt[-1] - srt[-2])
        name = self.classes[i]
        if name == "NO_GESTURE":
            return None, conf, "not a gesture"
        th = self.meta["thresholds"]
        floor = th.get("class_confidence", {}).get(name, th["confidence"])
        if conf < floor or margin < th["margin"]:
            return None, conf, (f"low confidence ({name} {conf * 100:.0f}%, "
                                f"margin {margin * 100:.0f}%)")
        return name, conf, ""
