"""
x7_2s_record_presence.py
========================
Dual-radar human presence / occupancy recorder.

Records 5-second dual-sensor windows for a chosen occupancy count AND a chosen
seating arrangement, then loops through every requested scenario automatically.

Scenario ORDER matters as much as the data. The v2 dataset recorded one
occupancy level per day, which made "how many people" perfectly predictable
from the day's noise floor: a classifier fed only bins beyond 3.55 m -- where
no seat exists -- still scored 99.7%. It had learnt the date, not the
occupants. This recorder therefore interleaves: a session runs in ROUNDS, and
every round visits every scenario once in reshuffled order, so drift in gain,
temperature or clutter spreads across all classes instead of encoding them.

Range: the stock human_preset.json already reaches 6.55 m. The bin count and
bin length are DETECTED from the radar at run time rather than hardcoded --
earlier revisions assumed a fixed count and silently zero-padded when the radar
disagreed, and assumed a bin length that was wrong by a factor of 11.

Usage:
    python x7_2s_record_presence.py --host1 529 --host2 545 --seats 4
"""

import os
import sys
import json
import tempfile
import argparse
import time
import threading
import queue
import itertools
import random
from pathlib import Path
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import x7_paths
x7_paths.setup_radar_imports()

from radar_direct_callback import RadarDirectCallback

# ── CONFIG ──────────────────────────────────────────────────────────────
# Frame rate is a TRADE, not a free parameter.
#
# Breathing is 0.20-0.33 Hz, so Nyquist needs only ~0.66 Hz -- 10 FPS is about
# 15x more than the signal requires. At 3 m the return is ~25 dB weaker than at
# 0.72 m (R^4), so SNR is the binding constraint, and surplus frame rate is
# exactly what you spend to buy integration back.
#
# Lowering FPS also lengthens the observation window for the same frame count.
# At 10 FPS a 50-frame window holds barely ONE breath cycle (1.0-1.7), which is
# too few to establish periodicity; at 5 FPS it holds 2.0-3.3.
#
# 5 FPS keeps Nyquist at 2.5 Hz, above heart rate (1.0-1.7 Hz), so respiration
# and pulse both stay accessible.
DEFAULT_FPS = 5
DEFAULT_SECONDS = 10.0

# Integration is scaled to hold the frame-time budget constant as FPS drops:
#     IterationsPerFrame = BASE_ITERATIONS * (BASE_FPS / fps)
BASE_FPS = 10
BASE_ITERATIONS = 700

SYNC_TOLERANCE = 0.1          # informational only; pairing is by sequence number

# Measured from the SDK (probe_range.py), not assumed. Imported so there is a
# single source of truth.
from x7_physics import BIN_LENGTH_M, RANGE_OFFSET_M

DEFAULT_PRESET = x7_paths.preset("human_preset.json")

# Measured on 2026-09-10: the stock preset returns 112 bins spanning
# -1.38 m .. 6.55 m. This is the EXPECTED value only; the recorder allocates
# from whatever the radar actually sends (see _ensure_alloc) and warns on
# mismatch. Raising MframesPerPulse to extend range does NOT work -- every
# preset shipped with the SDK uses 7 and the radar silently refuses others.
RANGE_BINS = 112

# x7_2s_live_presence imports this name.
RADAR_FPS = DEFAULT_FPS


# ── RADAR SOURCE ────────────────────────────────────────────────────────
class X7DualRadar:
    def __init__(self, host1, host2, preset_json_path=None, fps=DEFAULT_FPS):
        self.fps = fps
        self.frame_queue = queue.Queue(maxsize=400)
        self.range_bins = None            # detected from the first frame
        self._alloc_lock = threading.Lock()
        self.virtual_frame1 = None
        self.virtual_frame2 = None
        self.last_tx_data1 = [False, False]
        self.last_tx_data2 = [False, False]

        if preset_json_path is None:
            preset_json_path = DEFAULT_PRESET

        self.host1, self.host2 = host1, host2
        self.buffer_lock = threading.Lock()
        self.buffer1 = {}                 # seq -> (frame, timestamp)
        self.buffer2 = {}
        self.last_pair_skew = 0

        self.runner1 = RadarDirectCallback()
        self.runner2 = RadarDirectCallback()
        self.preset1_path = self._create_port_preset(preset_json_path, host1)
        self.preset2_path = self._create_port_preset(preset_json_path, host2)

        self._thread1 = threading.Thread(
            target=self.runner1.run_with_callback_preset,
            args=(self._frame_callback1, self.preset1_path), daemon=True)
        self._thread2 = threading.Thread(
            target=self.runner2.run_with_callback_preset,
            args=(self._frame_callback2, self.preset2_path), daemon=True)
        self._thread1.start()
        print(f"[INFO] Sensor 1 (Host {host1}) stream started.")
        self._thread2.start()
        print(f"[INFO] Sensor 2 (Host {host2}) stream started.")

    def _create_port_preset(self, base_preset, host_id):
        with open(base_preset, 'r') as f:
            preset = json.load(f)
        preset.setdefault("ConnectionParameters", {})
        preset["ConnectionParameters"]["HostInterfaceId"] = str(host_id)
        x7_paths.fix_firmware_path(preset)
        # FPS and integration are set together so they cannot drift apart.
        rs = preset.setdefault("RadarSource", {})
        rs["FPS"] = f"{float(self.fps):.1f}"
        rs["IterationsPerFrame"] = str(int(BASE_ITERATIONS * BASE_FPS / self.fps))
        tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        json.dump(preset, tmp)
        tmp.close()
        return tmp.name

    def _ensure_alloc(self, n_bins):
        """Size the virtual frames from the first frame the radar actually sends."""
        with self._alloc_lock:
            if self.range_bins is None:
                self.range_bins = int(n_bins)
                self.virtual_frame1 = np.zeros((2, 2, self.range_bins), dtype=np.complex64)
                self.virtual_frame2 = np.zeros((2, 2, self.range_bins), dtype=np.complex64)
                far = RANGE_OFFSET_M + (self.range_bins - 1) * BIN_LENGTH_M
                print(f"\n[INFO] Radar returned {self.range_bins} range bins -> "
                      f"{RANGE_OFFSET_M:+.2f} m .. {far:+.2f} m")
                if self.range_bins != RANGE_BINS:
                    print(f"[WARN] Expected {RANGE_BINS} bins. Recording is still "
                          f"valid, but tell whoever trains on it.")

    def _process_assembled_frame(self, frame, seq_num, sensor_idx, timestamp):
        with self.buffer_lock:
            (self.buffer1 if sensor_idx == 1 else self.buffer2)[seq_num] = (frame, timestamp)
            matched = None
            for off in range(-10, 11):
                t = seq_num + off
                if t in self.buffer1 and t in self.buffer2:
                    matched = t
                    self.last_pair_skew = abs(self.buffer1[t][1] - self.buffer2[t][1])
                    break
            if matched is None:
                return
            combined = np.stack([self.buffer1[matched][0], self.buffer2[matched][0]], axis=0)
            for k in [k for k in self.buffer1 if k <= matched]:
                del self.buffer1[k]
            for k in [k for k in self.buffer2 if k <= matched]:
                del self.buffer2[k]
            if not self.frame_queue.full():
                self.frame_queue.put(combined)

    def _callback(self, trx_mask, data, seq_num, timestamp, sensor_idx):
        try:
            tx_idx = int(trx_mask[1])
            radar_data = np.asarray(data, dtype=np.float32)
            self._ensure_alloc(radar_data.shape[2])
            if radar_data.shape[2] > self.range_bins:
                radar_data = radar_data[:, :, :self.range_bins]
            complex_data = (radar_data[:, 0, :] + 1j * radar_data[:, 1, :]).astype(np.complex64)
            if complex_data.shape[1] < self.range_bins:
                return True                      # short frame: drop, never pad
            vf = self.virtual_frame1 if sensor_idx == 1 else self.virtual_frame2
            flags = self.last_tx_data1 if sensor_idx == 1 else self.last_tx_data2
            vf[tx_idx, :, :] = complex_data
            flags[tx_idx] = True
            if all(flags):
                self._process_assembled_frame(vf.copy(), seq_num, sensor_idx, timestamp)
                flags[0] = flags[1] = False
        except Exception:
            pass
        return True

    def _frame_callback1(self, trx_mask, data, seq_num, timestamp, range_offset, bin_length):
        return self._callback(trx_mask, data, seq_num, timestamp, 1)

    def _frame_callback2(self, trx_mask, data, seq_num, timestamp, range_offset, bin_length):
        return self._callback(trx_mask, data, seq_num, timestamp, 2)

    def get_frame(self):
        try:
            return self.frame_queue.get(timeout=2.0)
        except queue.Empty:
            return None

    def flush_queue(self):
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Exception:
                break

    def close(self):
        for p in (self.preset1_path, self.preset2_path):
            try:
                os.remove(p)
            except Exception:
                pass
        print("[INFO] X7 dual radar closed.")


# ── RECORDING ───────────────────────────────────────────────────────────
def record_trial(radar, duration_s, label):
    radar.flush_queue()
    n_frames = int(duration_s * radar.fps)
    frames = None
    collected = 0
    while collected < n_frames:
        frame = radar.get_frame()
        if frame is None:
            print("\n[WARN] Timeout waiting for a synchronised frame pair.")
            break
        if frames is None:
            frames = np.zeros((n_frames,) + frame.shape, dtype=np.complex64)
        frames[collected] = frame
        collected += 1
        bar = "#" * int(20 * collected / n_frames)
        sys.stdout.write(f"\r    recording {label}  [{bar:<20}] {collected}/{n_frames}")
        sys.stdout.flush()
    print()
    return frames[:collected] if frames is not None else np.empty(0)


def seat_scenarios(n_seats, n_people):
    """All distinct seat combinations for a given occupancy."""
    if n_people == 0:
        return [()]
    return list(itertools.combinations(range(1, n_seats + 1), n_people))


def build_plan(n_seats):
    """Ask what to record and return a list of (n_people, seats) scenarios."""
    print("\n" + "=" * 62)
    print(" SESSION PLAN")
    print("=" * 62)
    print(f" Seats available: {n_seats}  (numbered 1..{n_seats})")

    raw = input(f"\n Which occupancy levels? (e.g. 0,1,2,3  or 'all') [all]: ").strip().lower()
    if raw in ("", "all"):
        levels = list(range(0, n_seats + 1))
    else:
        try:
            levels = sorted({int(x) for x in raw.replace(" ", "").split(",")})
        except ValueError:
            print(" Could not parse, using all.")
            levels = list(range(0, n_seats + 1))
    levels = [l for l in levels if 0 <= l <= n_seats]

    plan = []
    for k in levels:
        combos = seat_scenarios(n_seats, k)
        if k == 0:
            plan.append((0, ()))
            continue
        print(f"\n {k} person(s) -- {len(combos)} possible seat arrangement(s):")
        for i, c in enumerate(combos, 1):
            print(f"   [{i}] seats {','.join(map(str, c))}")
        sel = input(f" Which arrangements? (e.g. 1,3  or 'all') [all]: ").strip().lower()
        if sel in ("", "all"):
            chosen = combos
        else:
            try:
                idx = {int(x) for x in sel.replace(" ", "").split(",")}
                chosen = [combos[i - 1] for i in sorted(idx) if 1 <= i <= len(combos)]
            except (ValueError, IndexError):
                print(" Could not parse, using all.")
                chosen = combos
        plan += [(k, c) for c in chosen]
    return plan


def scenario_tag(n_people, seats):
    return "empty" if n_people == 0 else "seats" + "".join(map(str, seats))


def run_session(args, radar):
    plan = build_plan(args.seats)
    if not plan:
        print("Nothing selected.")
        return

    subject = input("\n Subject / session ID (e.g. S01): ").strip() or "S01"
    try:
        trials = int(input(f" Trials per scenario [{args.trials}]: ").strip() or args.trials)
    except ValueError:
        trials = args.trials

    block = max(1, min(args.block, trials))
    n_rounds = (trials + block - 1) // block
    EMPTY = (0, ())
    # The empty cabin can get more trials than the other scenarios. Its extra
    # visits are spread over every round, so it never ends up as one block.
    target = {sc: trials for sc in plan}
    if EMPTY in target and args.empty_trials:
        target[EMPTY] = max(1, args.empty_trials)
    empty_per_round = (-(-target.get(EMPTY, 0) // (block * n_rounds))
                       if EMPTY in target else 0)
    total = sum(target.values())

    # Tag every file with the sitting it came from, so the confound check is
    # exact instead of inferred from timestamp gaps.
    sitting = time.strftime("%m%d%H%M")
    subj_field = f"{subject}-{sitting}"

    print("\n" + "=" * 62)
    print(f" {len(plan)} scenario(s) x {trials} trial(s) = {total} recordings")
    print(f" interleaved: {n_rounds} round(s), {block} back-to-back trial(s) per visit")
    if EMPTY in target:
        print(f" empty cabin: {target[EMPTY]} trial(s), {empty_per_round} visit(s) per round")
    if block == 1:
        print(" block 1: every trial is its own sit-down -- RE-SEAT before each one")
    print(f" {args.seconds:.0f}s each -> about {total * (args.seconds + 4) / 60:.0f} min")
    print(f" sitting id {sitting}")
    if len({k for k, _ in plan}) < 2:
        print("\n [WARNING] only one occupancy level in this session. Recording a")
        print("           single level per sitting is what confounded the v2 data:")
        print("           the model learns the day, not the people. Include every")
        print("           level in every sitting unless you are topping up.")
    print("=" * 62)

    out_root = Path(args.out)
    done = 0
    remaining = dict(target)

    for rnd in range(n_rounds):
        order = [sc for sc in plan if remaining[sc] > 0 and sc != EMPTY]
        if EMPTY in remaining and remaining[EMPTY] > 0:
            left_rounds = n_rounds - rnd
            visits = -(-remaining[EMPTY] // (block * left_rounds))
            order += [EMPTY] * visits
        random.shuffle(order)
        # keep repeated empty visits apart where possible
        for _ in range(50):
            if all(not (a == b == EMPTY) for a, b in zip(order, order[1:])):
                break
            random.shuffle(order)
        print("\n" + "=" * 62)
        print(f" ROUND {rnd + 1}/{n_rounds}   ({len(order)} scenario(s) this round)")
        print("=" * 62)

        for n_people, seats in order:
            sc = (n_people, seats)
            tag = scenario_tag(n_people, seats)
            k = min(block, remaining[sc])
            if k <= 0:
                continue
            where = "EMPTY -- everyone out of the cabin" if n_people == 0 else \
                    f"{n_people} person(s) in seat(s) {','.join(map(str, seats))}"
            print("\n" + "-" * 62)
            print(f" SCENARIO: {where}")
            print(f" round {rnd + 1}/{n_rounds}, {k} trial(s) now, "
                  f"{remaining[sc]} of {target[sc]} left overall")
            if n_people == 0:
                print(" -> occupants leave the room (or the same far spot every time);")
                print("    the operator stays exactly where they always are")
            else:
                print(" -> EVERYONE stand up and sit back down, even if already in this seat")
            print("-" * 62)
            ans = input(" Position everyone, then press Enter "
                        "('s' skip this visit, 'q' end session): ").strip().lower()
            if ans == "q":
                print(" ending session early.")
                remaining = {x: 0 for x in remaining}
                break
            if ans == "s":
                print(" skipped.")
                continue

            out_dir = out_root / f"{n_people}_people"
            out_dir.mkdir(parents=True, exist_ok=True)
            # An empty-cabin trial needs everyone clear of the radar, the
            # operator included, so it gets a longer countdown.
            cd = args.countdown_empty if n_people == 0 else args.countdown
            for t in range(k):
                idx = target[sc] - remaining[sc] + 1
                print(f"  trial {idx}/{target[sc]}")
                for c in range(cd, 0, -1):
                    sys.stdout.write(f"\r    starting in {c}...  ")
                    sys.stdout.flush()
                    time.sleep(1.0)
                sys.stdout.write("\r" + " " * 30 + "\r")
                data = record_trial(radar, args.seconds, tag)
                if len(data) == 0:
                    print("    no data, skipping.")
                    continue
                fn = (f"{n_people}people_{tag}_{subj_field}_"
                      f"{int(time.time())}_{idx:03d}.npy")
                np.save(out_dir / fn, data)
                remaining[sc] -= 1
                done += 1
                print(f"    saved {fn}   ({done}/{total})")

        if not any(remaining.values()):
            break

    print("\n" + "=" * 62)
    print(f" Session complete: {done} recordings under {out_root}/  (sitting {sitting})")
    print("=" * 62)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual-radar occupancy recorder")
    parser.add_argument("--host1", required=True, help="HostInterfaceId for Sensor 1")
    parser.add_argument("--host2", required=True, help="HostInterfaceId for Sensor 2")
    parser.add_argument("--out", default="human_dataset_v2", help="Output directory")
    parser.add_argument("--seats", type=int, default=4, help="Number of seats (default 4)")
    parser.add_argument("--trials", type=int, default=10, help="Default trials per scenario")
    parser.add_argument("--empty-trials", dest="empty_trials", type=int, default=None,
                        help="Trials for the empty cabin (default: same as the other "
                             "scenarios). Extra empty visits are spread over every round.")
    parser.add_argument("--countdown", type=int, default=3,
                        help="Countdown seconds before each trial (default 3)")
    parser.add_argument("--countdown-empty", dest="countdown_empty", type=int,
                        default=15,
                        help="Countdown before an EMPTY-cabin trial (default "
                             "15). Everyone must be clear of the radar, the "
                             "operator included, so 3 s is not enough.")
    parser.add_argument("--block", type=int, default=5,
                        help="Back-to-back trials per scenario visit (default 5). The "
                             "session repeats in rounds until every scenario has its "
                             "full trial count, reshuffling order each round, so no "
                             "occupancy level is confined to one part of the sitting.")
    parser.add_argument("--preset", default=None, help="Override the radar preset")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS,
                        help=f"Radar frame rate (default {DEFAULT_FPS}). Lower FPS buys "
                             "integration and a longer breathing window.")
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS,
                        help=f"Seconds per trial (default {DEFAULT_SECONDS})")
    args = parser.parse_args()

    iters = int(BASE_ITERATIONS * BASE_FPS / args.fps)
    print(f"[INFO] {args.fps:g} FPS, {iters} iterations/frame "
          f"({10 * np.log10(iters / BASE_ITERATIONS):+.1f} dB vs {BASE_FPS} FPS)")
    print(f"[INFO] {args.seconds:g}s trials -> {int(args.seconds * args.fps)} frames, "
          f"{args.seconds * 0.20:.1f}-{args.seconds * 0.33:.1f} breath cycles per window")

    # An orphaned live-inference worker holds the radar exclusively and makes
    # this fail with a misleading "check the DCNF0/1 jumpers" message.
    try:
        from x7_acq_worker import clear_stale_workers
        clear_stale_workers()
    except Exception:
        pass

    radar = X7DualRadar(args.host1, args.host2, preset_json_path=args.preset, fps=args.fps)
    try:
        time.sleep(2.0)
        run_session(args, radar)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        radar.close()
