# Builds dist\X7App\X7App.exe (one-folder app) from x7_app.py.
#
#   powershell -ExecutionPolicy Bypass -File build_exe.ps1
#
# Uses its own venv (.venv-build) with CPU-only torch: the app only runs three
# small models, and the CUDA build in .venv would add ~2.5 GB.
# Python 3.10 is required -- the radar SDK ships PySignalFlow for cp310 only --
# but NOT 3.10.0: its dis module has a bug that crashes PyInstaller
# ("IndexError: tuple index out of range" in dis._get_const_info). The system
# Python here is 3.10.0, so a standalone 3.10 from uv is used:
#   pip install uv;  uv python install 3.10

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py = ".\.venv-build\Scripts\python.exe"
if (-not (Test-Path $py)) {
    $base = (& uv python find 3.10 --no-project --managed-python)
    if (-not $base) { throw "no uv-managed Python 3.10 -- run: uv python install 3.10" }
    & $base -m venv .venv-build
    & $py -m pip install --upgrade pip
    & $py -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
    & $py -m pip install -r requirements-build.txt
}

# This PC sets PYTHONPATH to the Novelda Utils folder; the build must not see
# it, or PyInstaller would freeze those modules instead of shipping them as files.
$env:PYTHONPATH = ""

& $py -m PyInstaller x7_app.spec --noconfirm --clean
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed ($LASTEXITCODE)" }

$out = "dist\X7App"

# Models and data sit NEXT TO the exe (not inside _internal) so they can be
# swapped without rebuilding, and so calibration can write to them.
New-Item -ItemType Directory -Force "$out\models" | Out-Null
$models = @("gesture_v4_pause.json", "gesture_v4_pause_seed0.pt", "gesture_v4_pause_seed1.pt",
            "gesture_v4_pause_seed2.pt", "presence_zone.json", "presence_calib.json")
foreach ($m in $models) {
    if (Test-Path "models\$m") { Copy-Item "models\$m" "$out\models\" -Force }
    else { Write-Warning "models\$m not found -- not copied" }
}
# gesture_dataset_v4_pause: mixed into live gesture calibration (and where new
# calibration captures are saved); human_dataset_v3: the no-radar demo.
foreach ($d in @("gesture_dataset_v4_pause", "human_dataset_v3")) {
    robocopy $d "$out\$d" /E /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "copying $d failed ($LASTEXITCODE)" }
}
Copy-Item "X7App_README.txt" "$out\README.txt" -Force

$size = (Get-ChildItem $out -Recurse | Measure-Object Length -Sum).Sum / 1MB
Write-Host ("Built {0}\X7App.exe  ({1:N0} MB)" -f $out, $size)
exit 0
