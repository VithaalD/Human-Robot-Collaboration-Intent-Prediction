# Run Intent Engine v3.1 on this laptop

This Windows-ready runner uses the supplied epoch-196 ResNet-101 + LSTM checkpoint and the original three-class mapping:

1. `INTERACTION` -> `STOP` when confidence is at least 0.75
2. `PASSTHRU` -> `CAUTION`
3. `WAIT` -> `CLEAR`

It preserves v3.1's 150 x 150 preprocessing, pixel normalization, eight-frame default window, three-prediction smoothing window, and 0.75 stop threshold.

## Run one of the supplied videos

Open PowerShell in this project folder and run:

```powershell
.\run_intent_v3_1.ps1 -VideoPath "C:\Users\vitha\Downloads\raw videos for mecha lab-20260904T185555Z-1-001\raw videos for mecha lab\INTERACTION\int1.mov"
```

The video loops until you press `Ctrl+C`. To stop automatically after 20 seconds:

```powershell
.\run_intent_v3_1.ps1 -VideoPath "C:\Users\vitha\Downloads\raw videos for mecha lab-20260904T185555Z-1-001\raw videos for mecha lab\PASS THRU\pass1.mov" -RunSeconds 20
```

## Run from a connected camera

```powershell
.\run_intent_v3_1.ps1 -CameraIndex 0
```

The runner tries an Intel RealSense first when its Python package is installed, followed by Windows DirectShow, Media Foundation, and OpenCV's automatic backend.

If Windows blocks the script because of its PowerShell execution policy, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_intent_v3_1.ps1 -VideoPath "C:\path\to\video.mov"
```

## Controls and output

- Press `Ctrl+C` to stop cleanly.
- `STOP` means an interaction prediction met the 0.75 confidence threshold.
- `CAUTION` means pass-through was predicted.
- `CLEAR` means wait/no action was predicted, or interaction confidence was below the stop threshold.
- This program reports an intent flag; it does not send a physical stop command to a robot.

The currently available PyTorch environment is CPU-only. It ran at roughly two inference windows per second during validation. Installing a CUDA-enabled PyTorch environment can use the laptop's RTX 5070 for substantially better live responsiveness without changing the learned weights.

## External data and collaborator setup

Raw videos may stay outside the Git repository. The monitor and terminal runner search for `raw videos for mecha lab-20260904T185555Z-1-001/raw videos for mecha lab` inside the project first, then next to the project. The epoch-196 checkpoint currently lives with those external videos and must also be obtained separately.

For another location, set `INTENT_DATA_DIR` to the inner `raw videos for mecha lab` folder. Set `INTENT_WEIGHTS_PATH` to the full epoch-196 checkpoint path to keep weights separately. The GUI also supports its Browse buttons, and opens even without the external data folder. Webcam inference needs the checkpoint but does not need recorded videos. The terminal runner accepts `-DataDirectory` and `-WeightsPath` overrides.

Example for the terminal runner:

```powershell
.\run_intent_v3_1.ps1 -WeightsPath "D:\models\cnnlstm-Epoch-196-Loss-0.01737015192823795.pth" -VideoPath "D:\videos\example.mov"
```

New webcam recordings are saved under the project in `Camera Recordings`, which is excluded from Git.
