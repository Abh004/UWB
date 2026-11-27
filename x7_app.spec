# PyInstaller spec for X7App.exe -- build with build_exe.ps1, not by hand.
# One-folder build: torch is far too large for a one-file exe, which would
# unpack it to a temp folder on every launch.
#
# Inside the bundle (X7App\_internal) the radar SDK keeps its own layout,
# because x7_paths.py and the SDK look for files relative to each other:
#   sdk\        PySignalFlow.pyd + SDK DLLs (+ Nodes\PythonEmbedNode.dll)
#   radar\      radar_direct_callback.py, Flows\, Presets\, Utils\  (as source)
#   firmware\   BA22 RadarDirect firmware the presets point to

from pathlib import Path

ROOT = Path(SPECPATH)
SDK = ROOT / "X7_SDK_0.6_x64-win"
SDK_BIN = SDK / "bin"
RADAR = ROOT / "novelda-uwb-demos" / "Demos" / "RadarDirect" / "X7RadarDirectCallback"
NODES_PY = ROOT / "novelda-uwb-demos" / "Nodes" / "Python" / "Novelda"
UTILS = NODES_PY / "Utils"

binaries = [(str(p), "sdk") for p in SDK_BIN.glob("*.dll")]
binaries += [(str(p), "sdk") for p in SDK_BIN.glob("PySignalFlow*.pyd")]
binaries += [(str(p), "sdk/Nodes") for p in (SDK_BIN / "Nodes").glob("*.dll")]

datas = [(str(SDK_BIN / "SignalFlow-windows.manifest"), "sdk"),
         (str(RADAR / "radar_direct_callback.py"), "radar")]
datas += [(str(p), "radar/Flows") for p in (RADAR / "Flows").glob("*.sfl")]
datas += [(str(p), "radar/Presets") for p in (RADAR / "Presets").glob("*.json")]
datas += [(str(p), "radar/Utils") for p in UTILS.glob("*.py")]
# Python node the .sfl flows load BY NAME when the flow starts
# ("BasebandPlotter.radar_direct_param_node.RadarDirectParamNode"); without it
# flow.load() fails with "could not assign python module and/or class".
# Only this file: the rest of BasebandPlotter is a Qt plotter we do not use.
datas += [(str(NODES_PY / "BasebandPlotter" / "radar_direct_param_node.py"),
           "radar/BasebandPlotter")]
datas += [(str(p), "firmware") for p in (SDK / "firmware").glob("*RadarDirect*.app")]

hiddenimports = [
    # loaded at runtime from radar\ (source files), so PyInstaller cannot see
    # what they import from the standard library
    "multiprocessing.shared_memory", "multiprocessing.resource_tracker",
    "pickle", "copy", "datetime", "json", "pathlib",
    # imported inside functions / worker processes
    "x7_acq_worker", "x7_2s_record", "x7_2s_record_presence", "x7_gesture_record",
    "x7_gesture_live", "x7_gesture_train", "x7_live_combined", "x7_presence_zone",
    "x7_presence_setup", "x7_physics", "x7_gesture_gui", "x7_paths", "x7_selfcheck",
    "psutil", "msvcrt",
]

excludes = ["torchvision", "torchaudio", "matplotlib", "sklearn", "scipy", "pandas",
            "IPython", "jupyter", "notebook", "tensorboard", "PySignalFlow",
            "radar_direct_callback", "Utils", "BasebandPlotter"]

a = Analysis(
    ["x7_app.py"],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="X7App",
    console=True,           # calibration prompts and key presses use the console
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="X7App", upx=False)
