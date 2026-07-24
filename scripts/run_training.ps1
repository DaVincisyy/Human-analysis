param(
    [string]$Config = "configs/main.yaml"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$configPath = Join-Path $projectRoot $Config

if (-not (Test-Path -LiteralPath $python)) {
    throw "Virtual environment Python not found: $python"
}
if (-not (Test-Path -LiteralPath $configPath)) {
    throw "Config not found: $configPath"
}

$configData = Get-Content -Raw -LiteralPath $configPath
$outputMatch = [regex]::Match($configData, "(?m)^output_dir:\s*(.+?)\s*$")
if (-not $outputMatch.Success) {
    throw "output_dir is missing from $configPath"
}

$outputDir = Join-Path $projectRoot $outputMatch.Groups[1].Value.Trim()
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
$stdoutLog = Join-Path $outputDir "train.stdout.log"
$stderrLog = Join-Path $outputDir "train.stderr.log"
$exitFile = Join-Path $outputDir "train.exitcode"

Set-Location $projectRoot
$env:PYTHONUNBUFFERED = "1"
$ErrorActionPreference = "Continue"
& $python "train_retrieval.py" "--config" $Config 1> $stdoutLog 2> $stderrLog
$LASTEXITCODE | Set-Content -LiteralPath $exitFile
exit $LASTEXITCODE
