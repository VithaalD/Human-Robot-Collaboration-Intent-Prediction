# Try the small prototype

Double-click Start Small Prototype.cmd in the project root. It selects the trained epoch-28 checkpoint and eight-frame setting automatically. The monitor loads pixel scaling, mean/std, sampling rate and class order from the checkpoint. A mismatched frame count is rejected.

## Recorded video
1. Leave Video selected.
2. Use Browse beside Video to choose a recording (one is preselected).
3. Press Start to see the camera preview, predicted category and class probabilities.
4. Press Stop before changing the source. Video playback loops and follows original capture timestamps when matching metadata is present. Keep recordings and their matching metadata folders together.

## Live camera
1. Connect the lab RealSense RGB camera and recreate the training camera angle/workspace.
2. Open Start Small Prototype.cmd, select Webcam, and click Refresh if needed.
3. Choose the RealSense RGB camera from the Camera list, retain eight inference frames, and press Start.
4. Try hands outside the tabletop or resting at its edge (WAIT), hands farther over the tabletop without task handling (PASSTHRU), and task-tool/workpiece handling (INTERACTION).
5. Allow roughly one second for the initial frame window; watch category probabilities as well as the large status indicator. Press Stop when finished.

Record only does not run predictions. The display's STOP/CAUTION/CLEAR flags are existing threshold/smoothing indicators; the category and probabilities show the model output. The 0.75 confidence threshold and three-prediction smoothing are inherited settings, not calibrated robot-control guarantees. This interface displays predictions and does not issue robot movement commands.

## Verified October 10
- Checkpoint metadata, eight-frame/eight-fps sampling and ImageNet preprocessing integrated.
- All nine extracted-frame validation predictions reproduced in the monitor, including the previously known error (8/9 ground-truth accuracy).
- First isolated inference about 442 ms; median subsequent inference about 61 ms. This includes preprocessing and GPU inference for one window, not total action-response latency.
- Timed ten-second replay delivered preview frames and predictions; first prediction about 1.2 seconds after starting the loaded engine. Capture and inference threads stopped cleanly. Actual desktop Video-mode startup/shutdown also passed.
- UI setup check and 70 sampling/temporal/recording regression tests passed.

## Limits discovered
The short continuous replay of take 2 displayed only WAIT. A diagnostic using the same nine validation windows decoded directly from original videos scored 6/9, compared with 8/9 on the extracted JPEGs. This indicates sensitivity to the source-frame preparation path; original-resolution decoding, resizing and JPEG preparation differ. It is not a new held-out test score, and the exact cause is not yet isolated. No model weights or test-set decisions were changed to improve this diagnostic.

The earlier 10/10 reserved-test result applies to selected extracted-frame windows, not full raw-video playback or live camera accuracy. The application is ready for exploratory testing; robust continuous-video behavior is not established.

Available cameras here: ASUS FHD webcam and OBS Virtual Camera. The RealSense lab camera was not connected, so live lab accuracy, transitions and end-to-end latency could not be validated. No laptop-camera footage was captured during these checks.

Next: test at the lab, log failures near workspace boundaries and action transitions, and align/verify source-frame preparation before treating the offline scores as live performance. Keep model evaluation separate from any subsequent preprocessing/model changes.
