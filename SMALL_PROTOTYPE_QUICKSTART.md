# Try the full-dataset prototype
Updated October 15, 2026.

Double-click Start Small Prototype.cmd in the project root. It selects the October 15 full-dataset checkpoint (best epoch 12), with eight-frame/eight-fps input.

For recordings: select Video, Browse to an AVI, then press Start. Keep matching metadata alongside the videos so playback uses capture timestamps. Press Stop before changing sources.

For the lab camera: connect the RealSense RGB camera, select Webcam, click Refresh, choose the RGB camera, and press Start. Match the training view. Record only does not run predictions. Predictions are display-only and do not issue robot commands.

Try WAIT with hands outside or at the immediate table edge, PASSTHRU with hands farther into the tabletop without task handling, and INTERACTION with task tool/workpiece handling. Watch category changes and probabilities near transitions. Allow roughly one second to collect the initial window, plus inference time.

Validation: 15/16 correct (INTERACTION 6/6, PASSTHRU 5/6, WAIT 4/4). One PASSTHRU was predicted INTERACTION. These are model-selection validation results; the 16 reserved test clips have not been scored. Live lab accuracy remains unverified.

All 8,420 prepared frames match live preprocessing exactly. All 16 validation windows produce exactly equal prepared/original-frame probabilities in the monitor. A ten-second recorded replay passed. See the active run's FULL_DATASET_TRAINING_SUMMARY.md for split, labels and limitations. Previous model files and active selection are preserved.
