param(
    [string]$BasePython = (Join-Path $env:USERPROFILE 'anaconda3\envs\DEEPLABCUT\python.exe'),
    [string]$EnvironmentPath = (Join-Path $env:USERPROFILE '.venvs\intent-gpu')
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $BasePython -PathType Leaf)) {
    throw 'Supply -BasePython with the path to a Python 3.10 or 3.11 installation.'
}
$intentPython = Join-Path $EnvironmentPath 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $intentPython -PathType Leaf)) {
    & $BasePython -m venv $EnvironmentPath
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the isolated environment.' }
}
& $intentPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip setup failed.' }
& $intentPython -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
$intentRequirements = Join-Path $PSScriptRoot 'requirements-gpu-lock.txt'
if (-not (Test-Path -LiteralPath $intentRequirements -PathType Leaf)) {
    $intentRequirements = Join-Path $PSScriptRoot 'requirements-gpu.txt'
}
& $intentPython -m pip install -r $intentRequirements
if ($LASTEXITCODE -ne 0) { throw 'Project package installation failed.' }
& $intentPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Installed package dependencies are inconsistent.' }
& $intentPython -B (Join-Path $PSScriptRoot 'tools\verify_gpu.py')
if ($LASTEXITCODE -ne 0) { throw 'GPU verification failed. See the output above.' }
Write-Output "GPU environment is ready: $intentPython"
