<#
    00_setup_env.ps1
    ---------------------------------------------------------------
    Creates an isolated virtual environment for the Antibody
    Developability project and installs all pinned dependencies.

    Usage (from the project root):
        powershell -ExecutionPolicy Bypass -File .\scripts\00_setup_env.ps1

    Notes
      * CPU-only PyTorch is installed on purpose: the primary dataset is
        a few hundred antibodies, so a GPU buys nothing and a CPU wheel
        is ~200 MB instead of ~3 GB. To use a GPU instead, replace the
        torch install line with the CUDA index URL for your driver.
      * The script is idempotent: re-running it reuses the existing venv.
#>

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath    = Join-Path $ProjectRoot ".venv"
$LogDir      = Join-Path $ProjectRoot "logs"
$LogFile     = Join-Path $LogDir "setup_env.log"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Log($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Write-Output $line
    Add-Content -Path $LogFile -Value $line -Encoding utf8
}

# --- 1. Locate a real CPython interpreter -----------------------------------
# `python` on PATH may resolve to a bundled interpreter shipped with some other
# application (e.g. Inkscape), which has no pip. Prefer the per-user install.
$Candidates = @(
    "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
)
$BasePython = $Candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $BasePython) {
    $BasePython = (Get-Command py -ErrorAction SilentlyContinue).Source
    if (-not $BasePython) { throw "No suitable Python interpreter found." }
}
Log "Base interpreter: $BasePython"
& $BasePython --version

# --- 2. Create the virtual environment --------------------------------------
if (-not (Test-Path (Join-Path $VenvPath "Scripts\python.exe"))) {
    Log "Creating virtual environment at $VenvPath"
    & $BasePython -m venv $VenvPath
} else {
    Log "Reusing existing virtual environment at $VenvPath"
}

$Py = Join-Path $VenvPath "Scripts\python.exe"
& $Py -m pip install --upgrade pip setuptools wheel 2>&1 | Tee-Object -Append -FilePath $LogFile

# --- 3. Install dependencies in stages --------------------------------------
# Staged so that one optional failure does not abort the whole install.

Log "Stage 1/5: scientific core"
& $Py -m pip install "numpy>=2.0,<3" "pandas>=2.2" "scipy>=1.13" "scikit-learn>=1.5" `
    "matplotlib>=3.8" "seaborn>=0.13" "statsmodels>=0.14" "joblib>=1.4" `
    "tqdm>=4.66" "pyyaml>=6.0" 2>&1 | Tee-Object -Append -FilePath $LogFile

Log "Stage 2/5: gradient boosting + bio utilities"
& $Py -m pip install "xgboost>=2.1" "lightgbm>=4.5" "biopython>=1.84" "logomaker>=0.8" 2>&1 |
    Tee-Object -Append -FilePath $LogFile

# AntPack supplies IMGT numbering, CDR extraction, germline V/J calls and the
# curated chemical-liability scan. The version is pinned exactly and on
# purpose: 0.3.9+ requires a free academic licence key entered at an
# INTERACTIVE prompt, which would break any unattended build, and 0.5 is
# yanked on PyPI. 0.3.8.6.2 is GPL-3, needs no key, and ships a cp313
# Windows wheel whose only dependency is numpy.
Log "Stage 2b/5: antibody numbering (AntPack, pinned)"
& $Py -m pip install "antpack==0.3.8.6.2" 2>&1 | Tee-Object -Append -FilePath $LogFile

Log "Stage 3/5: PyTorch (CPU build)"
& $Py -m pip install torch --index-url https://download.pytorch.org/whl/cpu 2>&1 |
    Tee-Object -Append -FilePath $LogFile

Log "Stage 4/5: protein language models (AbLang2 primary, ESM-2 baseline)"
& $Py -m pip install "ablang2" 2>&1 | Tee-Object -Append -FilePath $LogFile
& $Py -m pip install "transformers>=4.44" "fair-esm>=2.0" 2>&1 | Tee-Object -Append -FilePath $LogFile

Log "Stage 5/5: explainability (optional)"
try {
    & $Py -m pip install "shap>=0.46" 2>&1 | Tee-Object -Append -FilePath $LogFile
} catch {
    Log "WARNING: shap install failed; the pipeline falls back to permutation importance."
}

# --- 4. Freeze the resolved environment -------------------------------------
$Frozen = Join-Path $ProjectRoot "requirements-lock.txt"
& $Py -m pip freeze | Out-File -FilePath $Frozen -Encoding utf8
Log "Wrote pinned environment to $Frozen"

# --- 5. Smoke test ----------------------------------------------------------
# Written to a temp .py file rather than passed via `-c`: PowerShell strips the
# inner double quotes out of a `-c` argument, which silently corrupts the code.
Log "Running import smoke test"
$SmokeFile = Join-Path $env:TEMP "abdev_smoke_test.py"
@'
import importlib
mods = ['numpy','pandas','scipy','sklearn','matplotlib','seaborn','statsmodels','antpack',
        'xgboost','lightgbm','Bio','logomaker','torch','ablang2','transformers','esm','shap']
failed = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        print('  OK   {:<14} {}'.format(m, getattr(mod, '__version__', 'n/a')))
    except Exception as e:
        print('  FAIL {:<14} {}: {}'.format(m, type(e).__name__, e))
        failed.append(m)
print('')
print('{}/{} modules imported successfully'.format(len(mods) - len(failed), len(mods)))
if failed:
    print('Failed: ' + ', '.join(failed))
'@ | Out-File -FilePath $SmokeFile -Encoding utf8
& $Py $SmokeFile 2>&1 | Tee-Object -Append -FilePath $LogFile
Remove-Item $SmokeFile -ErrorAction SilentlyContinue

Log "Environment setup complete."
