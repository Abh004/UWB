"""
x7_selfcheck.py
===============
`X7App.exe selfcheck` (or `python x7_app.py selfcheck`): checks, without
streaming from the radar, the things that break when the app is moved to a
new PC -- the radar SDK loading from the bundle, presets and firmware, the
models, worker processes, and whether Windows sees the radar boards.
"""

import json
import os
import sys
import time


def _probe_worker(q):
    """Runs in a spawned process, like the radar and fine-tune workers do."""
    try:
        import torch  # noqa: F401
        import x7_acq_worker  # noqa: F401
        import x7_gesture_train  # noqa: F401
        q.put("ok")
    except Exception as exc:
        q.put(f"{type(exc).__name__}: {exc}")


def main():
    import x7_paths
    results = []

    def check(name, fn):
        try:
            detail = fn()
            results.append(True)
            print(f"  OK    {name}" + (f"  -- {detail}" if detail else ""))
        except Exception as exc:
            results.append(False)
            print(f"  FAIL  {name}  -- {type(exc).__name__}: {exc}")

    print(f"X7 self-check   ({'packaged app' if x7_paths.FROZEN else 'from source'}, "
          f"Python {sys.version.split()[0]})")
    print(f"  folder: {x7_paths.APP_DIR}")

    def sdk():
        x7_paths.setup_radar_imports()
        import radar_direct_callback  # noqa: F401
        import PySignalFlow as psf
        psf.Flow()
        where = os.path.dirname(psf.__file__)
        if x7_paths.FROZEN and os.path.normcase(where) != os.path.normcase(str(x7_paths.SDK_BIN)):
            raise RuntimeError(f"loaded from {where}, not from the bundle")
        return f"PySignalFlow from {where}"
    check("radar SDK", sdk)

    def flow_nodes():
        # The .sfl flows name Python node classes ("pkg.module.Class") that the
        # SDK imports when a flow loads -- i.e. only once the radar streams.
        import glob
        import importlib
        import re
        names = set()
        for f in glob.glob(str(x7_paths.RADAR_DIR / "Flows" / "*.sfl")):
            raw = open(f, "rb").read()
            names |= {m.decode() for m in re.findall(rb"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*){2,}", raw)}
        if not names:
            raise RuntimeError("no flow files found")
        for name in sorted(names):
            mod, cls = name.rsplit(".", 1)
            getattr(importlib.import_module(mod), cls)
        return ", ".join(sorted(names))
    check("radar flow nodes", flow_nodes)

    def presets():
        out = []
        for name in ("default_preset.json", "human_preset.json"):
            p = x7_paths.preset(name)
            stp = x7_paths.fix_firmware_path(json.load(open(p)))
            if not os.path.isfile(stp["BA22FirmwarePath"]):
                raise FileNotFoundError(f"{name}: firmware {stp['BA22FirmwarePath']}")
            out.append(name)
        return ", ".join(out) + " + firmware"
    check("radar presets and firmware", presets)

    def gesture_model():
        import x7_gesture_core as C
        m = C.GestureModel(os.path.join("models", "gesture_v4_pause.json"))
        return f"{len(m.models)} models, classes {', '.join(m.classes)}, device {m.device}"
    check("gesture model", gesture_model)

    def presence_model():
        json.load(open(os.path.join("models", "presence_zone.json")))
        calib = os.path.join("models", "presence_calib.json")
        return ("calibration present" if os.path.exists(calib)
                else "no presence_calib.json -- run presence-setup calibrate")
    check("presence model", presence_model)

    def datasets():
        n = {d: len([f for f in os.listdir(d) if f.endswith(".npy")]) if os.path.isdir(d) else 0
             for d in ("gesture_dataset_v4_pause",)}
        if not n["gesture_dataset_v4_pause"]:
            raise FileNotFoundError("gesture_dataset_v4_pause (needed by gesture calibration)")
        return f"{n['gesture_dataset_v4_pause']} gesture windows for calibration"
    check("calibration data", datasets)

    def workers():
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(target=_probe_worker, args=(q,))
        t0 = time.time()
        p.start()
        r = q.get(timeout=120)
        p.join(10)
        if r != "ok":
            raise RuntimeError(r)
        return f"started and imported torch in {time.time() - t0:.1f} s"
    check("worker processes", workers)

    def writable():
        p = os.path.join("models", ".write_test")
        with open(p, "w") as f:
            f.write("x")
        os.remove(p)
    check("folder is writable (calibration can be saved)", writable)

    def usb():
        from x7_acq_worker import radar_usb_interfaces, find_stale_workers
        n = radar_usb_interfaces()
        stale = find_stale_workers()
        msg = f"{n} of 4 FT4222H interfaces present" if n is not None else "could not query USB"
        if stale:
            msg += f"; {len(stale)} orphaned radar worker(s) running"
        if n is not None and n < 4:
            raise RuntimeError(msg + " -- radar boards not (fully) connected")
        return msg
    check("radar boards on USB", usb)

    ok = all(results)
    print("\nAll checks passed." if ok else
          f"\n{results.count(False)} check(s) failed (the last one only matters for live use).")
    return 0 if ok else 1


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(main())
