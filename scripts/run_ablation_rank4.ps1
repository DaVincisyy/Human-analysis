$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$outputDir = Join-Path $projectRoot "outputs\ablations"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

Set-Location $projectRoot
$env:PYTHONUNBUFFERED = "1"
$ErrorActionPreference = "Continue"
& $python "run_ablation_suite.py" "--names" "rank4_a8_all" `
    1> (Join-Path $outputDir "rank4.stdout.log") `
    2> (Join-Path $outputDir "rank4.stderr.log")
$LASTEXITCODE | Set-Content -LiteralPath (Join-Path $outputDir "rank4.exitcode")
exit $LASTEXITCODE
