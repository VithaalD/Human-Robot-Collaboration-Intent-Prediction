# Run the local Intent Monitor

This desktop GUI uses the previously verified v3.1 Windows engine and your epoch-196 checkpoint. It displays a video/webcam preview, class confidence bars, STOP/CAUTION/CLEAR status, and an event log. Its layout follows the GUI included in your research files. The linked YouTube video could not be inspected, so an exact visual match has not been verified.

## First launch on this laptop

1. Extract the entire ZIP into a folder you can find, for example Desktop\Intent Monitor. Keep its files together.
2. Double-click **Start Intent Monitor.cmd**. A terminal and the desktop window will open. Leave the terminal open while using the monitor.
3. The weights and INTERACTION\int1.mov fields are already filled in using the current locations on your laptop.
4. Leave **Video** selected and click **Start**. Allow several seconds for the checkpoint to load.
5. The video loops. The right panel shows the model's three probabilities and current status. CPU inference updates more slowly than the video preview.
6. Click **Stop** before changing sources. Close the window to quit.

No package installation is needed on this laptop: the launcher uses C:\Users\vitha\anaconda3\envs\DEEPLABCUT\python.exe, which currently includes PyTorch 2.10.0+cpu, TorchVision 0.25.0+cpu, OpenCV 4.11.0, Pillow, and Tkinter. The launcher does not activate or modify that environment. The application performs inference locally and does not require internet access.

## Try the previously tested videos

Click **Stop**, then the **Video** row's **Browse** button, and choose a file under:

    C:\Users\vitha\Downloads\raw videos for mecha lab-20260904T185555Z-1-001\raw videos for mecha lab

Examples:

| File | Previously observed prediction | Status |
| --- | --- | --- |
| INTERACTION\int1.mov | INTERACTION, up to approximately 97% | STOP |
| PASS THRU\pass1.mov | PASSTHRU, up to approximately 98% | CAUTION |
| WAIT\wait1.mov | WAIT, approximately 92–95% | CLEAR |

These were spot checks, not an accuracy benchmark. Live sampling and smoothing mean displayed values vary through playback. In this setup session, int1.mov was checked again and produced INTERACTION at 96.7% and STOP, with about 1.9 predictions per second.

## Use your webcam

1. Confirm the webcam works in the Windows Camera app, then close that app.
2. In the monitor, click **Stop**, select **Webcam**, leave camera index at **0**, and click **Start**.
3. If opening the camera fails, enable camera access and desktop-app camera access in Windows Settings. Close other applications using the camera. For an external camera, try index **1**.

The earlier session found no accessible webcam at indices 0–5. This session verified recorded-video input; live webcam operation remains unverified. You can use the video mode without any camera or robot hardware.

## Settings preserved from the tested runner

- Epoch-196 three-class checkpoint: cnnlstm-Epoch-196-Loss-0.01737015192823795.pth
- Class order: INTERACTION, PASSTHRU, WAIT
- Input image size: 150 × 150; temporal window: 8 frames
- Original pixel normalization; smoothing over 3 predictions
- INTERACTION becomes STOP when confidence is at least 75%; PASSTHRU becomes CAUTION; otherwise CLEAR

CLEAR is the model's flag, not a verified safety judgment. This program only displays status and does not send robot commands. Retraining is unnecessary to use this GUI. The supplied architecture is ResNet-101 followed by a standard LSTM, not a bidirectional LSTM.

## If something does not open

- Missing weights/video: use the corresponding **Browse** button to select the file in its current location. Keep using the epoch-196 three-class checkpoint.
- run_intent_v3_1.ps1 also discovers the external data folder and is terminal-only. Use Start Intent Monitor.cmd for the GUI.
- Launcher reports Python missing: it expects the existing DEEPLABCUT environment at the path above. Do not substitute the MSYS2 Python that previously appeared on PATH.
- The terminal shows an error: retain its full text for troubleshooting. No environment changes are needed for the tested CPU setup.
- Slow predictions: the current PyTorch installation is CPU-only. The GPU is not enabled by this launcher. GPU acceleration can be configured separately; it is not required to run the interface.

## Verification performed

The new GUI started, loaded the checkpoint, opened the video, and shut down automatically without an error. A separate check through the same preview-enabled engine verified decoded preview frames, nonzero model probabilities, and the INTERACTION/STOP result. The YouTube reference and final visual layout were not visually inspected.

## External data and collaborator setup

Raw videos may stay outside the Git repository. The monitor and terminal runner search for `raw videos for mecha lab-20260904T185555Z-1-001/raw videos for mecha lab` inside the project first, then next to the project. The epoch-196 checkpoint currently lives with those external videos and must also be obtained separately.

For another location, set `INTENT_DATA_DIR` to the inner `raw videos for mecha lab` folder. Set `INTENT_WEIGHTS_PATH` to the full epoch-196 checkpoint path to keep weights separately. The GUI also supports its Browse buttons, and opens even without the external data folder. Webcam inference needs the checkpoint but does not need recorded videos. The terminal runner accepts `-DataDirectory` and `-WeightsPath` overrides.

Example for the terminal runner:

```powershell
.\run_intent_v3_1.ps1 -WeightsPath "D:\models\cnnlstm-Epoch-196-Loss-0.01737015192823795.pth" -VideoPath "D:\videos\example.mov"
```

New webcam recordings are saved under the project in `Camera Recordings`, which is excluded from Git.
