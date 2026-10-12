<#
    run_all.ps1 -- reproduce the entire study from a clean checkout.

    Prerequisite (run once):
        powershell -ExecutionPolicy Bypass -File .\scripts\00_setup_env.ps1

    Then:
        powershell -ExecutionPolicy Bypass -File .\scripts\run_all.ps1

    Options:
        -Quick          small model zoo and fewer repeats, for a smoke test
        -Repeats <n>    override the number of cross-validation repeats
        -SkipFeatures   reuse cached PLM embeddings instead of recomputing
        -Endpoint <k>   run a single endpoint (hic | acsins | tm | psr)

    Total runtime on a 12-core laptop, CPU only: roughly 60-90 minutes for the
    full run, of which the first execution additionally spends ~10 minutes
    downloading the AbLang2 and ESM-2 weights.
#>

param(
    [switch]$Quick,
    [int]$Repeats = 0,
    [switch]$SkipFeatures,
    [string]$Endpoint = ""
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    throw "Virtual environment not found. Run scripts\00_setup_env.ps1 first."
}

$env:PYTHONPATH = $ProjectRoot
$env:PYTHONIOENCODING = "utf-8"   # several assay column names contain non-ASCII

$LogDir = Join-Path $ProjectRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$RunLog = Join-Path $LogDir ("run_all_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

$script:StepNumber = 0
function Step($name, $module, $extraArgs = @()) {
    $script:StepNumber++
    $banner = "`n==== [$script:StepNumber] $name ====".PadRight(72, '=')
    Write-Output $banner
    Add-Content -Path $RunLog -Value $banner -Encoding utf8
    $started = Get-Date
    & $Py -m $module @extraArgs 2>&1 | Tee-Object -Append -FilePath $RunLog
    if ($LASTEXITCODE -ne 0) {
        throw "Step '$name' failed with exit code $LASTEXITCODE. See $RunLog"
    }
    $elapsed = (Get-Date) - $started
    $msg = "---- {0} finished in {1:mm\:ss}" -f $name, $elapsed
    Write-Output $msg
    Add-Content -Path $RunLog -Value $msg -Encoding utf8
}

$expArgs = @()
if ($Quick)        { $expArgs += "--quick" }
if ($Repeats -gt 0){ $expArgs += "--repeats=$Repeats" }
if ($Endpoint)     { $expArgs += "--endpoint=$Endpoint" }

Write-Output "Project root : $ProjectRoot"
Write-Output "Interpreter  : $Py"
Write-Output "Log file     : $RunLog"

# Fail fast on a module that does not even parse. The modelling step runs for
# well over an hour before it reaches the analysis modules, so a syntax error
# in one of those would otherwise surface only after all that compute.
Write-Output "`n==== [0] Pre-flight: compile and import every module ===="
& $Py -m compileall -q (Join-Path $ProjectRoot "src")
if ($LASTEXITCODE -ne 0) { throw "Pre-flight failed: a module does not compile." }
& $Py -m tests.smoke
if ($LASTEXITCODE -ne 0) { throw "Pre-flight failed: smoke tests did not pass." }

Step "Acquire FLAb at the pinned commit"   "src.data.acquire"
Step "Curate datasets and risk labels"     "src.data.curate"
Step "Number chains, extract CDRs, call germlines" "src.features.numbering"
Step "Compute sequence descriptors"        "src.features.descriptors"
Step "Generate leakage-safe splits"        "src.data.splits"

if (-not $SkipFeatures) {
    Step "Build feature matrices (AbLang2 + ESM-2)" "src.features.build"
} else {
    Write-Output "`n==== skipping feature build (-SkipFeatures) ===="
}

Step "Exploratory data analysis"           "src.analysis.eda"
Step "Model sweep, validation, uncertainty" "src.models.experiment" $expArgs
Step "Sequence motif and attribution analysis" "src.analysis.motifs"
Step "Error analysis, applicability domain, ablation" "src.analysis.error_analysis"
Step "Assemble the final report"           "src.report.build_report"

# Exercise the shipped screening tool end to end, on a cohort where the true
# answers are known. If the tool has drifted from the trained model -- a
# renamed descriptor, a changed column order -- this is where it shows up,
# rather than the first time someone tries to screen real candidates.
$script:StepNumber++
Write-Output ("`n==== [$script:StepNumber] Screening-tool demonstration ====".PadRight(72, '='))
& $Py -u (Join-Path $ProjectRoot "scripts\demo_screen.py") 2>&1 |
    Tee-Object -Append -FilePath $RunLog
if ($LASTEXITCODE -ne 0) { throw "Screening demo failed. See $RunLog" }

Write-Output "`nAll steps completed."
Write-Output "  Results : $(Join-Path $ProjectRoot 'results')"
Write-Output "  Report  : $(Join-Path $ProjectRoot 'report\FINAL_REPORT.md')"
Write-Output "  Models  : $(Join-Path $ProjectRoot 'models_store')"
