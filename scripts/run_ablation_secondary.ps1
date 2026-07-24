$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$outputDir = Join-Path $projectRoot "outputs\ablations"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

Set-Location $projectRoot
$env:PYTHONUNBUFFERED = "1"
$ErrorActionPreference = "Continue"
& $python "run_ablation_suite.py" "--names" `
    "alpha16_r16_all" `
    "modules_qv_r16_a32" `
    "data25_r16_a32_all" `
    "data50_r16_a32_all" `
    1> (Join-Path $outputDir "secondary.stdout.log") `
    2> (Join-Path $outputDir "secondary.stderr.log")
$LASTEXITCODE | Set-Content -LiteralPath (Join-Path $outputDir "secondary.exitcode")
exit $LASTEXITCODE
