param(
    [string]$VideoPath = "",
    [int]$CameraIndex = 0,
    [int]$SampleDuration = 8,
    [double]$RunSeconds = 0,
    [double]$StatusInterval = 0.5,
    [string]$DataDirectory = $env:INTENT_DATA_DIR,
    [string]$WeightsPath = $env:INTENT_WEIGHTS_PATH
)

$ErrorActionPreference = "Stop"

$intentPython = Join-Path $env:USERPROFILE "anaconda3\envs\DEEPLABCUT\python.exe"
$intentScript = Join-Path $PSScriptRoot "intent_engine_v3_1_windows.py"
$intentProject = Split-Path -Parent $PSScriptRoot
$intentDataRelative = "raw videos for mecha lab-20260904T185555Z-1-001\raw videos for mecha lab"
if (-not $DataDirectory) {
    $intentDataCandidates = @(
        (Join-Path $intentProject $intentDataRelative),
        (Join-Path (Split-Path -Parent $intentProject) $intentDataRelative)
    )
    $DataDirectory = $intentDataCandidates | Where-Object { Test-Path -LiteralPath $_ -PathType Container } | Select-Object -First 1
    if (-not $DataDirectory) { $DataDirectory = $intentDataCandidates[1] }
}
$intentWeights = $WeightsPath
if (-not $intentWeights) {
    $intentWeights = Join-Path $DataDirectory "cnnlstm-Epoch-196-Loss-0.01737015192823795.pth"
}
if (-not (Test-Path -LiteralPath $intentWeights -PathType Leaf)) {
    throw "Checkpoint not found: $intentWeights. Supply -WeightsPath or set INTENT_WEIGHTS_PATH; alternatively set -DataDirectory or INTENT_DATA_DIR to your raw videos for mecha lab folder."
}

foreach ($requiredFile in @($intentPython, $intentScript, $intentWeights)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Required file not found: $requiredFile"
    }
}

$engineArguments = @(
    $intentScript,
    "--resume_path", $intentWeights,
    "--class_labels", "INTERACTION", "PASSTHRU", "WAIT",
    "--n_classes", "3",
    "--sample_size", "150",
    "--sample_duration", $SampleDuration.ToString(),
    "--smoothing_window", "3",
    "--confidence_threshold", "0.75",
    "--camera_warmup_frames", "0",
    "--camera_index", $CameraIndex.ToString(),
    "--max_infer_hz", "10",
    "--run_seconds", $RunSeconds.ToString([Globalization.CultureInfo]::InvariantCulture),
    "--status_interval", $StatusInterval.ToString([Globalization.CultureInfo]::InvariantCulture)
)

if ($VideoPath) {
    $resolvedVideo = (Resolve-Path -LiteralPath $VideoPath -ErrorAction Stop).Path
    $engineArguments += @("--video_path", $resolvedVideo)
}

& $intentPython @engineArguments
exit $LASTEXITCODE
