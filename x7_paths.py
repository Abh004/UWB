"""
x7_paths.py
===========
Where the radar SDK, its flow/preset files and the firmware live -- in the
repository when run from source, inside the bundle when run as X7App.exe
(PyInstaller, see x7_app.spec).

Source layout (this machine)            Bundle layout (X7App\\_internal)
  X7_SDK_0.6_x64-win\\bin\\  (SDK DLLs)     sdk\\       PySignalFlow.pyd, DLLs, Nodes\\
  novelda-uwb-demos\\...\\X7RadarDirectCallback   radar\\   radar_direct_callback.py,
                                                   Flows\\, Presets\\, Utils\\
  novelda-uwb-demos\\Nodes\\Python\\Novelda (Utils)
  X7_SDK_0.6_x64-win\\firmware\\            firmware\\ *.app

The source paths used to be hard-coded as C:\\RR_02\\... in each recorder and
relied on PYTHONPATH / PATH set on this machine; a bundle must not depend on
either, and must not pick up the repo copy of the SDK when both exist.
"""

import os
import sys
from pathlib import Path

FROZEN = bool(getattr(sys, "frozen", False))

if FROZEN:
    BUNDLE = Path(sys._MEIPASS)                       # read-only resources
    APP_DIR = Path(sys.executable).resolve().parent   # models\, datasets next to the exe
    SDK_BIN = BUNDLE / "sdk"
    RADAR_DIR = BUNDLE / "radar"
    UTILS_PARENT = RADAR_DIR
    FIRMWARE_DIR = BUNDLE / "firmware"
    EXTRA_PATHS = ()
else:
    BUNDLE = APP_DIR = Path(__file__).resolve().parent
    _SDK = APP_DIR / "X7_SDK_0.6_x64-win"
    _DEMOS = APP_DIR / "novelda-uwb-demos"
    SDK_BIN = _SDK / "bin"
    RADAR_DIR = _DEMOS / "Demos" / "RadarDirect" / "X7RadarDirectCallback"
    UTILS_PARENT = _DEMOS / "Nodes" / "Python" / "Novelda"
    FIRMWARE_DIR = _SDK / "firmware"
    EXTRA_PATHS = (_DEMOS / "Demos",)

PRESETS_DIR = RADAR_DIR / "Presets"

_done = False


def setup_radar_imports():
    """Make `from radar_direct_callback import RadarDirectCallback` work.
    Idempotent; call it before that import."""
    global _done
    if _done:
        return
    _done = True
    if hasattr(os, "add_dll_directory") and SDK_BIN.is_dir():
        os.add_dll_directory(str(SDK_BIN))
    for p in (SDK_BIN, UTILS_PARENT, *EXTRA_PATHS, RADAR_DIR):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    if FROZEN:
        # radar_direct_callback.py puts the repo's SDK folder first on sys.path
        # when it exists (it does on the development PC). Importing the bundled
        # PySignalFlow first means that copy is the one every module gets.
        import PySignalFlow  # noqa: F401


def preset(name):
    """Path of a stock preset shipped with the radar wrapper."""
    return str(PRESETS_DIR / name)


def fix_firmware_path(preset_dict):
    """Presets store an absolute BA22 firmware path (C:\\RR_02\\X7_SDK...).
    Point it at our firmware folder when the file is there; otherwise leave it
    so the SDK's own "firmware not found" error names the path it wanted."""
    fw = preset_dict.get("BA22FirmwarePath")
    if fw:
        local = FIRMWARE_DIR / Path(fw.replace("\\", "/")).name
        if local.is_file():
            preset_dict["BA22FirmwarePath"] = str(local)
    return preset_dict
