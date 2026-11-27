X7App -- X7 radar presence + gesture
=====================================

Double-click X7App.exe for a menu, or run it from a terminal in this folder:

  X7App.exe combined --host1 529 --host2 545     presence + gesture, with window
  X7App.exe gesture  --host1 529 --host2 545     gesture only, with window
  X7App.exe presence-setup calibrate --host1 529 --host2 545
                                                 presence calibration (also: check, selftest)
  X7App.exe demo                                 presence + gesture on recordings, no radar
  X7App.exe gui-demo                             the window alone, no radar
  X7App.exe selfcheck                            is this PC set up? (SDK, models, USB)

On a new PC, run selfcheck first.

Add --help after a mode to see all of its options (they are the same as the
Python scripts x7_live_combined.py, x7_gesture_live.py, x7_presence_setup.py).
Stop with Ctrl+C in the console window.

Folder contents
  X7App.exe, _internal\        the program (Python, torch, radar SDK, firmware)
  models\                      gesture and presence models; presence calibration
                               (presence_calib.json) is written here
  gesture_dataset_v4_pause\    used by gesture calibration; new calibration
                               captures are saved here
  human_dataset_v3\            recordings for the demo mode

Keep the folder somewhere writable (not C:\Program Files), or calibration
results cannot be saved.

Requirements on the PC
  - Windows 10/11, 64-bit
  - FTDI FT4222 USB driver for the radar boards (Windows usually installs it
    automatically when the boards are plugged in)
  - both radar boards connected: 4 FT4222H USB interfaces must be present
