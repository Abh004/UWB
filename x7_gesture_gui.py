"""
x7_gesture_gui.py
=================
Window that shows the recognised gesture as an animation plus its full name
("SWIPE_LR" -> "Left to Right"), the confidence, and a short history -- and,
when driven by x7_live_combined.py, a cabin panel with the presence result
(occupied / empty, front and back row) and which mode the radar is in.

The window runs in a SEPARATE PROCESS (python x7_gesture_gui.py --child) fed
with JSON lines on stdin. The radar SDK's callback threads hold the GIL while
they wait for frames (see x7_gesture_live.py), so a 60 fps Tk animation in the
radar process would stutter -- and could delay frame reads. A plain subprocess
(not multiprocessing) also avoids re-importing torch and the radar modules in
the child, and the window closes itself when the parent exits (stdin EOF).

This module is stdlib-only on purpose: the child must start fast.

    python x7_gesture_live.py --host1 529 --host2 545 --gui     live, with window
    python x7_live_combined.py --host1 529 --host2 545          presence + gesture
    python x7_gesture_gui.py --demo                             no radar, cycles gestures
"""

import sys
import json
import math
import time
import threading
import subprocess
import queue as _queue
from pathlib import Path

FULL_NAME = {
    "SWIPE_LR": "Left to Right",
    "SWIPE_RL": "Right to Left",
    "SWIPE_UD": "Up to Down",
    "SWIPE_DU": "Down to Up",
    "PUSH_IN": "Push In",
    "NO_GESTURE": "No Gesture",
}
KIND = {"SWIPE_LR": "SWIPE", "SWIPE_RL": "SWIPE", "SWIPE_UD": "SWIPE",
        "SWIPE_DU": "SWIPE", "PUSH_IN": "PUSH", "NO_GESTURE": ""}
COLOR = {
    "SWIPE_LR": "#4cc9f0", "SWIPE_RL": "#f72585", "SWIPE_UD": "#80ed99",
    "SWIPE_DU": "#ffd166", "PUSH_IN": "#b388ff", "NO_GESTURE": "#8892a6",
}


def full_name(label):
    """'SWIPE_LR' -> 'Left to Right'; unknown labels fall back to a tidied form."""
    return FULL_NAME.get(label) or label.replace("_", " ").title()


def presence_summary(st):
    """x7_presence_zone.ZoneDetector.state() -> the small JSON-safe dict the
    window draws. Rows: "yes" / "no" / "unknown". A row the model marks unusable
    can confirm an occupant but never claim the seat is empty, so its "no" is
    shown as "unknown" -- the same rule as the terminal's "Y or ?"."""
    status = st.get("status", "UNKNOWN")
    usable = st.get("usable", {})
    warming = bool(st.get("warming"))

    def row(o):
        if status not in ("OCCUPIED", "EMPTY") or warming or o not in st:
            return "unknown"
        if st[o]:
            return "yes"
        return "no" if usable.get(o, True) else "unknown"

    if warming:
        detail = f"seen moving — rows known in {st.get('left', 0):.0f} s"
    elif status == "OCCUPIED" and "clears_in" in st:
        names = {"motion": "motion", "p_occupied": "breathing", "front": "front row",
                 "back": "back row"}
        detail = (f"held by {names.get(st.get('held_by'), st.get('held_by'))} "
                  f"{st.get('held_ago', 0):.0f} s ago")
    else:
        detail = st.get("detail", "")
    return {"status": status, "front": row("front"), "back": row("back"),
            "detail": detail, "warming": warming,
            "warnings": [str(w) for w in st.get("warnings", [])]}


# --------------------------------------------------------------------------- #
# Parent side: a tiny handle the radar loop calls. Never raises, never blocks
# for long -- a closed window must not stop recognition.
# --------------------------------------------------------------------------- #
class GestureDisplay:
    def __init__(self, title="X7 Gesture"):
        self._p = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--child", "--title", title],
            stdin=subprocess.PIPE, text=True, encoding="utf-8")

    def _send(self, **msg):
        if self._p is None or self._p.poll() is not None:
            self._p = None                     # window was closed: go quiet
            return
        try:
            self._p.stdin.write(json.dumps(msg) + "\n")
            self._p.stdin.flush()
        except (OSError, ValueError):
            self._p = None

    def status(self, text):
        self._send(type="status", text=text)

    def prompt(self, label, step, total):
        """Calibration: loop the animation of the gesture the user should do next."""
        self._send(type="prompt", label=label, step=step, total=total)

    def detected(self, label, conf):
        self._send(type="detected", label=label, conf=float(conf))

    def ignored(self, why, conf=0.0):
        self._send(type="ignored", why=why, conf=float(conf))

    def mode(self, mode, detail=""):
        """mode: "presence" | "gesture" | "switching" -- which preset the radar runs."""
        self._send(type="mode", mode=mode, detail=detail)

    def presence(self, st):
        """A ZoneDetector.state() dict, or {"status": "SENSOR FAULT", "detail": ...}."""
        self._send(type="presence", **presence_summary(st))

    def close(self):
        if self._p is not None:
            try:
                self._p.stdin.close()          # EOF -> the window closes itself
                self._p.wait(timeout=2)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                self._p.kill()
            self._p = None


class NullDisplay:
    """Same interface, does nothing -- used when --gui is off."""
    def status(self, text): pass
    def prompt(self, label, step, total): pass
    def detected(self, label, conf): pass
    def ignored(self, why, conf=0.0): pass
    def mode(self, mode, detail=""): pass
    def presence(self, st): pass
    def close(self): pass


# --------------------------------------------------------------------------- #
# Child side: the Tk window.
# --------------------------------------------------------------------------- #
BG = "#0f1420"
PANEL = "#171e2e"
TEXT = "#e8ecf4"
DIM = "#6b7590"
CYCLE_S = 1.5          # one play of a gesture: pause, swipe, hold
PLAYS = 2              # a detection plays this many times, then rests at the end
STALE_S = 5.0          # the name dims after this long without a new detection
GREEN, AMBER, RED = "#80ed99", "#ffb703", "#ef476f"
STATUS_COLOR = {"OCCUPIED": GREEN, "EMPTY": "#8892a6", "UNKNOWN": AMBER,
                "SENSOR FAULT": RED}
MODE_TEXT = {"presence": ("PRESENCE CHECK", "#b388ff"),
             "gesture": ("GESTURE RECOGNITION", "#4cc9f0"),
             "switching": ("SWITCHING RADAR PRESET", AMBER)}


def _mix(c1, c2, t):
    """Blend two #rrggbb colours (Tk has no alpha)."""
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def _ease(t):
    t = min(max(t, 0.0), 1.0)
    return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def _phase(t):
    """Cycle time 0..1 -> movement progress 0..1: hold at the start (the pause
    the v4_pause model was trained on), swipe, then hold at the end."""
    return _ease((t - 0.22) / 0.40)


class App:
    def __init__(self, title):
        import tkinter as tk
        self.tk = tk
        self.root = tk.Tk()
        self.root.title(title)
        self.root.configure(bg=BG)
        self.root.geometry("780x600")
        self.root.minsize(560, 460)

        # mode bar and cabin panel: packed only once presence/mode messages
        # arrive, so x7_gesture_live.py --gui keeps the gesture-only layout
        self.bar = tk.Label(self.root, text="", fg=BG, bg=DIM, anchor="w", padx=16,
                            font=("Segoe UI", 11, "bold"))
        self.left = tk.Frame(self.root, bg=BG, width=270)
        self.left.pack_propagate(False)
        self.right = tk.Frame(self.root, bg=BG)
        self.right.pack(side="right", fill="both", expand=True)
        tk.Label(self.left, text="CABIN", fg=DIM, bg=BG,
                 font=("Segoe UI", 12, "bold")).pack(pady=(18, 0))
        self.p_status = tk.Label(self.left, text="—", fg=TEXT, bg=BG,
                                 font=("Segoe UI", 26, "bold"))
        self.p_status.pack()
        self.p_cv = tk.Canvas(self.left, bg=PANEL, highlightthickness=0, height=270)
        self.p_cv.pack(fill="x", padx=(20, 4), pady=(10, 6))
        self.p_detail = tk.Label(self.left, text="", fg=DIM, bg=BG, wraplength=240,
                                 font=("Segoe UI", 10), justify="center")
        self.p_detail.pack()
        self.p_fresh = tk.Label(self.left, text="", fg=DIM, bg=BG, wraplength=240,
                                font=("Segoe UI", 10, "italic"), justify="center")
        self.p_fresh.pack(pady=(4, 0))
        self.p_warn = tk.Label(self.left, text="", fg=RED, bg=BG, wraplength=240,
                               font=("Segoe UI", 10), justify="center")
        self.p_warn.pack(pady=(4, 0))
        self.combined = False
        self.sys_mode = None     # presence | gesture | switching (combined only)
        self.pres = None         # last presence summary
        self.t_pres = 0.0
        self.t_hand = 0.0        # last capture of any kind (detected or ignored)

        self.head = tk.Label(self.right, text="WAITING FOR RADAR", fg=DIM, bg=BG,
                             font=("Segoe UI", 12, "bold"))
        self.head.pack(pady=(18, 0))
        self.name = tk.Label(self.right, text="—", fg=TEXT, bg=BG,
                             font=("Segoe UI", 46, "bold"))
        self.name.pack()
        self.conf = tk.Canvas(self.right, height=34, bg=BG, highlightthickness=0)
        self.conf.pack(fill="x", padx=180)
        self.cv = tk.Canvas(self.right, bg=PANEL, highlightthickness=0)
        self.cv.pack(fill="both", expand=True, padx=24, pady=12)
        self.hist = tk.Label(self.right, text="", fg=DIM, bg=BG, font=("Segoe UI", 11))
        self.hist.pack()
        self.stat = tk.Label(self.right, text="", fg=DIM, bg=BG, font=("Segoe UI", 11))
        self.stat.pack(pady=(2, 12))

        self.q = _queue.Queue()
        self.label = None        # gesture being animated
        self.t_start = 0.0
        self.loop = False        # calibration prompt: loop forever
        self.mode = "idle"       # idle | detected | prompt
        self.t_detect = 0.0
        self.history = []

    # -- messages ----------------------------------------------------------- #
    def feed_stdin(self):
        for line in sys.stdin:
            try:
                self.q.put(json.loads(line))
            except json.JSONDecodeError:
                pass
        self.q.put({"type": "quit"})

    def poll(self):
        try:
            while True:
                self.handle(self.q.get_nowait())
        except _queue.Empty:
            pass
        self.root.after(30, self.poll)

    def handle(self, m):
        t = m.get("type")
        if t == "quit":
            self.root.destroy()
        elif t == "status":
            self.stat.config(text=m["text"], fg=DIM)
            if self.mode == "prompt":           # calibration finished / moved on
                self.mode, self.label = "idle", None
                self.head.config(text="WAITING FOR A GESTURE", fg=DIM)
                self.name.config(text="—", fg=TEXT)
                self.conf.delete("all")
        elif t == "prompt":
            self.mode, self.loop = "prompt", True
            self.label, self.t_start = m["label"], time.time()
            self.head.config(text=f"CALIBRATION {m['step']}/{m['total']} — PLEASE PERFORM",
                             fg="#ffb703")
            self.name.config(text=full_name(m["label"]), fg=COLOR.get(m["label"], TEXT))
            self.conf.delete("all")
            self.stat.config(text="do it once when the terminal shows GO — capture is automatic",
                             fg=DIM)
        elif t == "detected":
            self.mode, self.loop = "detected", False
            self.label, self.t_start = m["label"], time.time()
            self.t_detect = self.t_hand = time.time()
            kind = KIND.get(m["label"], "")
            self.head.config(text=f"DETECTED {kind}".strip(), fg=DIM)
            self.name.config(text=full_name(m["label"]), fg=COLOR.get(m["label"], TEXT))
            self.draw_conf(m["conf"], COLOR.get(m["label"], TEXT))
            self.history.insert(0, f"{time.strftime('%H:%M:%S')}  {full_name(m['label'])} "
                                   f"{m['conf'] * 100:.0f}%")
            del self.history[5:]
            self.hist.config(text="     ·     ".join(self.history))
            self.stat.config(text="", fg=DIM)
        elif t == "ignored":
            self.t_hand = time.time()
            self.show_ignored(m["why"], m.get("conf", 0.0))
        elif t == "mode":
            self.go_combined()
            self.sys_mode = m["mode"]
            text, color = MODE_TEXT.get(m["mode"], (m["mode"].upper(), DIM))
            self.bar.config(text=f"●  {text}" + (f"   —   {m['detail']}" if m.get("detail") else ""),
                            bg=color)
            if m["mode"] != "gesture" and self.mode == "prompt":
                self.mode, self.label = "idle", None
            self.draw_cabin()
        elif t == "presence":
            self.go_combined()
            self.pres, self.t_pres = m, time.time()
            self.draw_cabin()

    def go_combined(self):
        if self.combined:
            return
        self.combined = True
        self.root.geometry("1060x640")
        self.bar.pack(side="top", fill="x", before=self.right)
        self.left.pack(side="left", fill="y", before=self.right)
        self.tick_cabin()

    def tick_cabin(self):
        """Keep the 'last checked' age current between messages."""
        self.draw_cabin()
        self.root.after(500, self.tick_cabin)

    def draw_cabin(self):
        """Top view: the radar at the front, then the front and back row."""
        pr = self.pres or {"status": "UNKNOWN", "front": "unknown", "back": "unknown",
                           "detail": "no presence result yet", "warnings": []}
        live = self.sys_mode == "presence"
        fade = 1.0 if live or self.pres is None else 0.55        # an old result looks old
        sc = STATUS_COLOR.get(pr["status"], TEXT)
        self.p_status.config(text=pr["status"].title(), fg=_mix(BG, sc, fade))
        detail = pr["detail"]
        if pr.get("warming") and not live and self.pres is not None:
            # the countdown stopped when the radar left the presence preset
            detail = "occupied (seen moving) — rows not measured this time"
        self.p_detail.config(text=detail)
        self.p_warn.config(text="\n".join(pr["warnings"]))
        if self.pres is None:
            fresh = ""
        elif live:
            fresh = "live"
        else:
            age = time.time() - self.t_pres
            fresh = f"last checked {int(age // 60)}:{int(age % 60):02d} ago"
            if self.t_hand > self.t_pres:
                fresh += f"\nhand movement {time.time() - self.t_hand:.0f} s ago"
        self.p_fresh.config(text=fresh)

        cv = self.p_cv
        cv.delete("all")
        W, H = max(cv.winfo_width(), 200), max(cv.winfo_height(), 200)
        cv.create_rectangle(W / 2 - 34, 14, W / 2 + 34, 36, fill=_mix(PANEL, DIM, 0.5),
                            outline="")
        cv.create_text(W / 2, 25, text="RADAR", fill=TEXT, font=("Segoe UI", 8, "bold"))
        for i, (row, name) in enumerate((("front", "Front row"), ("back", "Back row"))):
            y0 = 58 + i * 105
            v = pr[row]
            if v == "yes":
                fill, outline, word, dash = _mix(PANEL, GREEN, 0.75 * fade), "", "occupied", ()
            elif v == "no":
                fill, outline, word, dash = PANEL, _mix(PANEL, DIM, 0.8), "empty", ()
            else:
                fill, outline, word, dash = PANEL, _mix(PANEL, AMBER, 0.7 * fade), "?", (4, 3)
            cv.create_rectangle(18, y0, W - 18, y0 + 84, fill=fill, outline=outline,
                                width=2, dash=dash)
            ink = BG if v == "yes" else (TEXT if v == "no" else AMBER)
            ink = _mix(PANEL, ink, fade) if v != "yes" else ink
            cv.create_text(W / 2, y0 + 28, text=name, fill=ink, font=("Segoe UI", 12, "bold"))
            cv.create_text(W / 2, y0 + 56, text=word, fill=ink, font=("Segoe UI", 11))

    def show_ignored(self, why, conf):
        """A capture the model rejected (NO_GESTURE or below the confidence /
        margin floors). Default: a short grey note in the status line; the last
        accepted gesture and its animation stay on screen untouched."""
        text = "not a gesture" if why == "not a gesture" else f"not recognised — {why}"
        self.stat.config(text=text, fg="#a0566a")

    def draw_conf(self, conf, color):
        c = self.conf
        c.delete("all")
        w = max(c.winfo_width(), 200)
        c.create_rectangle(0, 4, w, 12, fill=PANEL, outline="")
        c.create_rectangle(0, 4, w * conf, 12, fill=color, outline="")
        c.create_text(w / 2, 25, text=f"{conf * 100:.0f}% confidence",
                      fill=DIM, font=("Segoe UI", 10, "bold"))

    # -- animation ---------------------------------------------------------- #
    def hand(self, x, y, s, color):
        """Open palm facing the viewer (= facing the radar)."""
        cv, tag = self.cv, "anim"
        for dx, ln in ((-17, 32), (-6, 40), (5, 38), (16, 30)):
            cv.create_line(x + dx * s, y - 14 * s, x + dx * s, y - (14 + ln) * s,
                           width=8.5 * s, fill=color, capstyle="round", tags=tag)
        cv.create_line(x - 22 * s, y + 4 * s, x - 36 * s, y - 14 * s,
                       width=10 * s, fill=color, capstyle="round", tags=tag)
        cv.create_oval(x - 24 * s, y - 22 * s, x + 24 * s, y + 26 * s,
                       fill=color, outline="", tags=tag)

    def frame(self):
        cv = self.cv
        cv.delete("anim")
        W, H = max(cv.winfo_width(), 100), max(cv.winfo_height(), 100)
        lab = self.label
        if lab is None or lab not in FULL_NAME or lab == "NO_GESTURE":
            cv.create_text(W / 2, H / 2, text="waiting for a gesture…", fill=DIM,
                           font=("Segoe UI", 14), tags="anim")
            self.root.after(16, self.frame)
            return

        if self.combined and self.sys_mode in ("presence", "switching"):
            cv.create_text(W / 2, H / 2, fill=DIM, font=("Segoe UI", 14), tags="anim",
                           text="gestures paused — the radar is checking presence"
                           if self.sys_mode == "presence" else "switching the radar preset…")
            self.root.after(16, self.frame)
            return

        el = time.time() - self.t_start
        if self.loop or el < CYCLE_S * PLAYS:
            u = (el % CYCLE_S) / CYCLE_S
            fade = 1.0 if u < 0.85 else 1 - (u - 0.85) / 0.15 * 0.6   # soft restart
        else:
            u, fade = 0.75, 1.0                                       # rest at the end
        p = _phase(u)
        color = COLOR.get(lab, TEXT)
        if self.mode == "detected" and time.time() - self.t_detect > STALE_S:
            fade *= 0.45
            self.name.config(fg=_mix(color, BG, 0.5))
        hc = _mix(PANEL, color, fade)
        s = min(W, H) / 300

        if lab == "PUSH_IN":
            cx, cy = W / 2, H / 2 + 10 * s
            for k in range(3):                     # rings contracting toward the radar
                r = (1 - ((p + k / 3) % 1.0)) * 120 * s + 20 * s
                cv.create_oval(cx - r, cy - r, cx + r, cy + r, width=2,
                               outline=_mix(PANEL, color, 0.35 * fade), tags="anim")
            self.hand(cx, cy, s * (0.65 + 0.85 * p), hc)
            cv.create_text(cx, H - 18 * s, text="toward the radar", fill=DIM,
                           font=("Segoe UI", 11), tags="anim")
        else:
            m = 70 * s
            a, b = {"SWIPE_LR": ((m, H / 2), (W - m, H / 2)),
                    "SWIPE_RL": ((W - m, H / 2), (m, H / 2)),
                    "SWIPE_UD": ((W / 2, m), (W / 2, H - m)),
                    "SWIPE_DU": ((W / 2, H - m), (W / 2, m))}[lab]
            cv.create_line(*a, *b, width=6 * s, fill=_mix(PANEL, color, 0.18),
                           arrow="last", arrowshape=(22 * s, 26 * s, 10 * s), tags="anim")
            for k in range(10, 0, -1):             # fading trail behind the hand
                q = max(p - k * 0.035, 0.0)
                tx, ty = a[0] + (b[0] - a[0]) * q, a[1] + (b[1] - a[1]) * q
                r = (14 - k) * s
                cv.create_oval(tx - r, ty - r, tx + r, ty + r, outline="",
                               fill=_mix(PANEL, color, fade * (1 - k / 11) * 0.5), tags="anim")
            x, y = a[0] + (b[0] - a[0]) * p, a[1] + (b[1] - a[1]) * p
            self.hand(x, y + 10 * s, s * 0.9, hc)
        self.root.after(16, self.frame)

    def run(self):
        threading.Thread(target=self.feed_stdin, daemon=True).start()
        self.root.after(30, self.poll)
        self.root.after(16, self.frame)
        self.root.mainloop()


# --------------------------------------------------------------------------- #
def _demo():
    """No radar: drive the real (separate-process) window with fake events."""
    ui = GestureDisplay("X7 Gesture — demo")
    try:
        ui.status("demo mode — no radar")
        time.sleep(1.5)
        ui.prompt("SWIPE_LR", 1, 2)
        time.sleep(3.5)
        ui.prompt("PUSH_IN", 2, 2)
        time.sleep(3.5)
        ui.status("calibration done — demo detections follow")
        seq = [("SWIPE_LR", .93), ("SWIPE_RL", .88), (None, "low confidence (SWIPE_UD 41%, margin 6%)"),
               ("SWIPE_UD", .61), ("SWIPE_DU", .57), ("PUSH_IN", .97), (None, "not a gesture")]
        while True:
            for label, v in seq:
                if label:
                    ui.detected(label, v)
                else:
                    ui.ignored(v)
                time.sleep(3.6)
                if ui._p is None:
                    return
    except KeyboardInterrupt:
        pass
    finally:
        ui.close()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="show the window without a radar")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--title", default="X7 Gesture", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.child:
        App(a.title).run()
    elif a.demo:
        _demo()
    else:
        ap.print_help()
