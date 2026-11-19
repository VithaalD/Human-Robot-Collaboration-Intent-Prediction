param([Parameter(ValueFromRemainingArguments=$true)][string[]]$PythonArgs)
$ErrorActionPreference = 'Stop'
$python = Join-Path $env:USERPROFILE '.venvs\intent-gpu\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Existing intent-gpu environment was not found.' }
$env:YOLO_CONFIG_DIR = Join-Path $PSScriptRoot '.runtime\ultralytics'
New-Item -ItemType Directory -Path $env:YOLO_CONFIG_DIR -Force | Out-Null
$env:PYTHONDONTWRITEBYTECODE = '1'
Push-Location $PSScriptRoot
try { & $python @PythonArgs; $result = $LASTEXITCODE } finally { Pop-Location }
exit $result
