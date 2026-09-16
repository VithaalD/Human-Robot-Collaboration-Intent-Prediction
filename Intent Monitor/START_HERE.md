# Intent Monitor

Double-click **Start Recording.cmd** in the project folder for camera recording without inference, or **Start Intent Monitor.cmd** for the monitor. The interface includes Record only, Video, and Webcam modes.

See [Recording and GPU setup](../RECORDING_AND_GPU.md) for collection steps, session identifiers, camera resolution, frame timing, GPU installation, and testing.

Recording-only needs no model checkpoint. For video/webcam inference, select the epoch-196 three-class checkpoint or a compatible newly trained checkpoint and the appropriate frame window. The current default is 16 frames; new checkpoints are checked for input-setting compatibility. Historical epoch-196 training settings were not saved, so its frame window is an explicit assumption.

The epoch-196 weights and raw videos currently remain external to the repository. The monitor checks `raw videos for mecha lab-20260904T185555Z-1-001/raw videos for mecha lab` inside or next to the project. Set `INTENT_DATA_DIR` / `INTENT_WEIGHTS_PATH` or use Browse for other locations.

The monitor displays predictions only and does not send robot commands. Camera/angle changes need separate evaluation. Previously observed example-video predictions are historical spot checks, not validation of new footage or the new frame setting.
