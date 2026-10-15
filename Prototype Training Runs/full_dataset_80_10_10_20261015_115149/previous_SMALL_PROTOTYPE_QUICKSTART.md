# Try the rebuilt small prototype
Updated October 12, 2026.

Double-click Start Small Prototype.cmd in the project root. It selects the verified original-frame-trained checkpoint (epoch 5) and eight-frame/eight-fps input. Shared preprocessing matches training and original-video input exactly in the completed checks.

For recordings: leave Video selected, Browse to an AVI, then press Start. Keep matching metadata alongside the videos so playback uses original capture timestamps. Press Stop before changing sources.

For the lab camera: connect the RealSense RGB camera, select Webcam, click Refresh, choose the RGB camera, and press Start. Match the training view. Record only does not run model predictions. Predictions are display-only and do not issue robot commands.

Try WAIT with hands outside or at the table edge, PASSTHRU with hands farther into the tabletop without task handling, and INTERACTION with task tool/workpiece handling. Observe the category and probability bars, especially near transitions. Allow roughly one second to collect the first input window.

Current validation: 8/9 correct on original-video windows (INTERACTION 2/2, PASSTHRU 0/1, WAIT 6/6). The missed PASSTHRU was labeled WAIT. This is an exploratory prototype, not proven live accuracy. The previous model's 10/10 test result does not apply to this rebuilt model. RealSense lab testing is still pending.

The rebuilt dataset and direct original-video path produced exactly equal tensors on 2,413 checked frames and identical probabilities on nine validation windows. A short recorded replay displayed INTERACTION and WAIT and stopped cleanly. See the active run's REBUILD_TRAINING_SUMMARY.md for full details.
