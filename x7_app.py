"""
x7_app.py
=========
One entry point for the live X7 tools -- the script X7App.exe is built from
(see x7_app.spec and build_exe.ps1). Runs from source too:

    python x7_app.py                               menu (what double-clicking X7App.exe shows)
    python x7_app.py combined --host1 529 --host2 545    presence + gesture, with window
    python x7_app.py gesture  --host1 529 --host2 545    gesture only, with window
    python x7_app.py demo [--speed 4]               presence + gesture on recordings, no radar
    python x7_app.py gui-demo                       the window alone, cycling fake gestures
    python x7_app.py presence-setup calibrate --host1 529 --host2 545
                                                    presence check / calibrate / selftest
    python x7_app.py selfcheck                      check SDK, models, workers, USB (no streaming)

Everything after the mode is passed to the underlying script unchanged, so
every option of x7_live_combined.py, x7_gesture_live.py and
x7_presence_setup.py works here (add --help after a mode to list them).

Paths: the working directory is set to the folder holding the exe (or this
file), so the scripts' defaults -- models\\, gesture_dataset_v4_pause\\,
human_dataset_v3\\ -- are found there, and calibration results are written there.
"""

import os
import sys


def _gui_child():
    """X7App.exe --gui-child: the separate window process GestureDisplay starts.
    Imports only the (stdlib-only) GUI module, so the window opens fast."""
    title = "X7 Gesture"
    if "--title" in sys.argv:
        title = sys.argv[sys.argv.index("--title") + 1]
    from x7_gesture_gui import App
    App(title).run()


def _run(mode, rest):
    if mode in ("combined", "demo"):
        import x7_live_combined as m
        if mode == "demo" and "--simulate" not in rest:
            rest = ["--simulate", *rest]
    elif mode == "gesture":
        import x7_gesture_live as m
        if "--no-gui" in rest:
            rest = [a for a in rest if a != "--no-gui"]
        elif "--gui" not in rest:
            rest = ["--gui", *rest]
    elif mode == "presence-setup":
        import x7_presence_setup as m
    elif mode == "gui-demo":
        import x7_gesture_gui
        return x7_gesture_gui._demo()
    elif mode == "selfcheck":
        import x7_selfcheck
        return x7_selfcheck.main()
    else:
        raise SystemExit(f"unknown mode {mode!r} -- run with --help for the list")
    sys.argv = [f"{os.path.basename(sys.argv[0])} {mode}", *rest]
    return m.main()


def _ask(prompt, default):
    try:
        v = input(f"{prompt} [{default}]: ").strip()
        # piped input can start with a byte-order mark, decoded as one
        # character (UTF-8) or three (cp1252, the packaged app's default)
        for bom in (chr(0xFEFF), "".join(map(chr, (0xEF, 0xBB, 0xBF)))):
            v = v[len(bom):] if v.startswith(bom) else v
    except EOFError:
        v = ""
    return v or default


def _menu():
    print("=" * 60)
    print(" X7 radar -- presence + gesture")
    print("=" * 60)
    print("  1  Presence + gesture (live radar)")
    print("  2  Gesture only (live radar)")
    print("  3  Presence calibration (live radar, needs people to sit in turn)")
    print("  4  Demo on recordings (no radar needed)")
    print("  5  Window demo (no radar needed)")
    print("  6  Self-check (is this PC set up correctly?)")
    print("  q  Quit")
    choice = _ask("Choose", "1").lower()
    if choice == "q":
        return None
    if choice in ("4", "5", "6"):
        return ({"4": "demo", "5": "gui-demo", "6": "selfcheck"}[choice], [])
    if choice not in ("1", "2", "3"):
        print(f"unknown choice {choice!r}")
        return None
    host1 = _ask("Host ID of sensor 1", "529")
    host2 = _ask("Host ID of sensor 2", "545")
    hosts = ["--host1", host1, "--host2", host2]
    mode = {"1": "combined", "2": "gesture", "3": "presence-setup"}[choice]
    return (mode, ["calibrate", *hosts] if mode == "presence-setup" else hosts)


def main():
    import multiprocessing
    multiprocessing.freeze_support()        # radar / fine-tune worker processes

    import x7_paths
    os.chdir(x7_paths.APP_DIR)

    args = sys.argv[1:]
    if args and args[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    from_menu = not args
    if from_menu:
        picked = _menu()
        if picked is None:
            return 0
        mode, rest = picked
    else:
        mode, rest = args[0], args[1:]

    try:
        return _run(mode, rest)
    except KeyboardInterrupt:
        print("\n[INFO] stopped")
        return 0
    except SystemExit as e:
        if from_menu and e.code not in (None, 0):
            print(e.code if isinstance(e.code, str) else f"exit code {e.code}")
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if from_menu:                       # double-clicked: keep the console readable
            try:
                input("\nPress Enter to close ...")
            except (EOFError, KeyboardInterrupt):
                pass


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--gui-child":
        _gui_child()
    else:
        sys.exit(main())
