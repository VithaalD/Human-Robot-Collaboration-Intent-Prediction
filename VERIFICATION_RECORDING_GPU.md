# Verification of recording and GPU preparation

GPU checks performed on 2026-09-15 on NVIDIA GeForce RTX 5070 Laptop GPU (7.96 GiB):

- PyTorch 2.10.0+cu128 and TorchVision 0.25.0+cu128.
- Ultralytics 8.4.153.
- No broken package requirements (`pip check`).
- Synthetic CNN optimizer step, 16-frame LSTM forward/backward, GPU NMS, YOLO11 nano forward/backward passed.
- Existing epoch-196 checkpoint loaded onto CUDA; actual ResNet101 + LSTM completed 16-frame, 150x150, batch-size-1 inference, backward pass and SGD optimizer step. Peak tensor allocation was approximately 1.25 GiB for that synthetic test. This is not a guarantee of available memory for other settings or Adam optimizer training.
- Verification did not save changed checkpoint weights, train a screwdriver detector, or measure model accuracy.
- Desktop UI smoke check passed in the user desktop session with no camera discovery and no model loading. The sandbox's restricted account could not initialize Tcl; the actual desktop account could.

The user's new physical camera has not been connected/tested by these checks. Record a pilot take and inspect delivered dimensions, actual FPS and timing diagnostics before collecting the full dataset.

Final integration: all 51 recording/temporal regression tests passed. The updated PowerShell runner loaded the historical checkpoint on CUDA, decoded the external int1.mov video with the default 16-frame setting, produced predictions, and completed the requested five-second run cleanly. Legacy preprocessing emitted deprecation warnings only. These checks do not establish accuracy on the new camera.
