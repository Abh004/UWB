"""
x7_acq_worker.py
================
Radar acquisition in its own PROCESS, so the SDK cannot stall inference.

Why this exists
---------------
The Novelda callback SDK delivers frames by calling into Python from its own
native threads, and it holds the GIL across the blocking wait for the next
frame. Anything else running in that interpreter is therefore throttled to the
gaps between frames. Measured on this machine, with both sensors streaming:

    inference, no radar               1.0 ms
    inference, radar streaming   13 000 ms      (13 000x)
    CPU burned during that call      0.01 cores  (system 0.5 of 24)

Nothing is computing during those 13 s -- the main thread is blocked. Raising
sys.setswitchinterval made it worse (15.6 s -> 23.6 s), and it persists after
radar.close() because the SDK's threads outlive it.

Recording was never affected: x7_2s_record_presence does almost nothing in the
main thread between frames, so losing the GIL costs it nothing. Live inference
is the opposite, which is why the live path looked broken while the recorder
looked fine.

The fix is process isolation, not tuning. The child holds the SDK and its GIL;
the parent gets frames over a pipe and has an interpreter to itself. Frames are
7 KB each at 5 FPS, so the transfer is free.

Not a thread: a thread shares the interpreter, and the GIL is exactly what is
contended.

Orphan safety
-------------
The FT4222 is exclusive-access: one stale process holding it makes every later
run fail with

    Could not find a FT4222 device configured in mode 0. Check the hardware
    connection and make sure that the DCNF0/1 jumpers are in position LL.

which points at the jumpers and is therefore actively misleading -- the
hardware is fine and enumerates OK. daemon=True is not enough on its own,
because daemon children are reaped by the parent's atexit handler and that
never runs when the parent is hard-killed (a timeout, a taskkill, a crash).
So the child also watches the parent and exits on its own when it disappears,
and find_stale_workers() lets a caller detect leftovers before touching the
radar. If one is already wedged in the driver, only a USB replug clears it.
"""

import multiprocessing as mp
import os
import queue as _queue
import threading
import time

import numpy as np


def _parent_watchdog(parent_pid, stop_evt, radar_box):
    """Exit the moment the parent dies, so the radar handle is never stranded.

    Hard-killing the parent skips daemon cleanup, and a leftover child keeps
    the FT4222 locked until the USB is physically replugged.
    """
    while not stop_evt.wait(2.0):
        alive = True
        try:
            import psutil
            alive = psutil.pid_exists(parent_pid)
        except ImportError:
            try:
                alive = os.getppid() == parent_pid
            except Exception:
                alive = True
        if not alive:
            try:
                if radar_box.get("radar") is not None:
                    radar_box["radar"].close()
            except Exception:
                pass
            time.sleep(0.2)
            os._exit(0)          # hard exit: the SDK will not unwind cleanly


def _worker(host1, host2, fps, preset, out_q, stop_evt, ready_evt, err_q,
            parent_pid=None):
    """Child process: own the radar, forward frames, keep Python work minimal."""
    radar_box = {"radar": None}
    if parent_pid is not None:
        threading.Thread(target=_parent_watchdog,
                         args=(parent_pid, stop_evt, radar_box),
                         daemon=True).start()
    try:
        from x7_2s_record_presence import X7DualRadar
        radar = X7DualRadar(host1, host2, preset_json_path=preset, fps=fps)
        radar_box["radar"] = radar
    except Exception as exc:                      # surface init failures upstream
        try:
            err_q.put(f"{type(exc).__name__}: {exc}")
        except Exception:
            pass
        return

    ready_evt.set()
    try:
        while not stop_evt.is_set():
            frame = radar.get_frame()
            if frame is None:
                continue
            try:
                # Drop rather than block: a stalled consumer must not push
                # back into the SDK callback thread.
                out_q.put_nowait(np.ascontiguousarray(frame))
            except _queue.Full:
                pass
    except Exception as exc:
        try:
            err_q.put(f"{type(exc).__name__}: {exc}")
        except Exception:
            pass
    finally:
        try:
            radar.close()
        except Exception:
            pass


def find_stale_workers():
    """Orphaned acquisition children still holding the radar.

    Returns [(pid, dead_parent_pid)]. A spawned child's command line carries
    parent_pid=N; if N is gone, nothing will ever reap it.
    """
    try:
        import psutil
    except ImportError:
        return []
    out = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if not (proc.info["name"] or "").lower().startswith("python"):
                continue
            cmd = " ".join(proc.info["cmdline"] or ())
            if "spawn_main" not in cmd or "parent_pid=" not in cmd:
                continue
            ppid = int(cmd.split("parent_pid=")[1].split(",")[0].split(")")[0])
            if ppid != os.getpid() and not psutil.pid_exists(ppid):
                out.append((proc.info["pid"], ppid))
        except Exception:
            continue
    return out


def clear_stale_workers(verbose=True):
    """Kill orphaned workers. Returns (killed, still_wedged).

    A worker blocked inside the FTDI driver can terminate yet linger as a
    zombie that still holds the device; that one needs a USB replug.
    """
    import subprocess
    stale = find_stale_workers()
    killed, wedged = [], []
    for pid, ppid in stale:
        if verbose:
            print(f"[WARN] orphaned acquisition process {pid} "
                  f"(parent {ppid} is gone) still holds the radar; killing it")
        try:
            subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                           capture_output=True, timeout=10)
        except Exception:
            pass
        time.sleep(1.0)
        try:
            import psutil
            (wedged if psutil.pid_exists(pid) else killed).append(pid)
        except Exception:
            killed.append(pid)
    if wedged and verbose:
        print(f"[ERROR] process {wedged} terminated but is still wedged in the "
              f"FTDI driver.\n"
              f"        The FT4222 stays locked until the driver lets go: "
              f"UNPLUG AND REPLUG\n"
              f"        the radar USB cable. The 'DCNF0/1 jumpers' message you "
              f"will otherwise\n"
              f"        get is misleading -- the hardware is fine.")
    return killed, wedged


class RadarProcess:
    """Same get_frame()/close() shape as X7DualRadar, but out of process."""

    def __init__(self, host1, host2, fps=None, preset=None, maxsize=200):
        if fps is None:
            from x7_2s_record_presence import DEFAULT_FPS
            fps = DEFAULT_FPS
        ctx = mp.get_context("spawn")
        self._q = ctx.Queue(maxsize=maxsize)
        self._err = ctx.Queue()
        self._stop = ctx.Event()
        self._ready = ctx.Event()
        self._proc = ctx.Process(
            target=_worker,
            args=(host1, host2, fps, preset, self._q, self._stop,
                  self._ready, self._err, os.getpid()),
            daemon=True,
        )
        self._proc.start()

    def wait_ready(self, timeout=60.0):
        """True once the child has both sensors streaming."""
        if self._ready.wait(timeout):
            return True
        raise RuntimeError(self.error() or
                           "radar acquisition process did not start in time")

    def error(self):
        try:
            return self._err.get_nowait()
        except Exception:
            return None

    def get_frame(self, timeout=2.0):
        try:
            return self._q.get(timeout=timeout)
        except _queue.Empty:
            if not self._proc.is_alive():
                raise RuntimeError(self.error() or
                                   "radar acquisition process died")
            return None

    def flush_queue(self):
        while True:
            try:
                self._q.get_nowait()
            except Exception:
                return

    def qsize(self):
        try:
            return self._q.qsize()
        except Exception:
            return -1

    def close(self):
        self._stop.set()
        if self._proc.is_alive():
            self._proc.join(timeout=5.0)
            if self._proc.is_alive():
                self._proc.terminate()
