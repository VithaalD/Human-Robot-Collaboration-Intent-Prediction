# Fusion Viewer

## Open the application
Extract the ZIP into the Human-Robot-Collaboration-Intent-Prediction repository folder. Keep the resulting FusionViewer directory together. Double-click Open_Fusion_Viewer.cmd, then select **Play lab demo**. The viewer uses the included demo, timing metadata, and three matching model files relative to this folder. During playback, Q or Escape closes the viewer and R starts a fresh sequence.

Choose **Choose a video** for another recording. For lab evaluation, connect the RealSense camera, select its Windows RGB camera index and click **Start live camera**. The app starts with camera index 0, which may be the laptop webcam. The UI displays predictions only.

## Environment
The launcher uses the existing Windows environment at %USERPROFILE%\.venvs\intent-gpu. On a different Windows NVIDIA workstation, supply a Python 3.10 or 3.11 executable and run the following from the extracted FusionViewer directory in PowerShell. The setup script installs the pinned packages and verifies the GPU.

    .\setup_gpu.ps1 -BasePython 'C:\path\to\python.exe'

If startup fails, check the launcher error dialog and the newest .runtime/launcher log. The optional native SDK CLI mode requires pyrealsense2. The default Windows RGB camera mode uses OpenCV.

## Recorded validation results
Both the original and fusion models correctly classified 15 of 16 validation recordings. Baseline cross-entropy was 0.1575 and macro F1 was 0.9441. Selected fusion cross-entropy was 0.1975 and macro F1 was 0.9267. The fusion checkpoint comes from epoch 1 of the 0.00001 learning-rate run. YOLO features changed the model scores. Validation has yet to show an improvement over baseline. Full run reports in results/ contain historical paths for provenance.

## Reproduce inference and training
From this directory, run:

    .\run_yolo_python.ps1 tools/run_fusion_viewer.py --demo
    .\run_yolo_python.ps1 tools/run_fusion_viewer.py --video "C:\path\to\recording.avi"

The Windows release includes the labeled feature cache in .runtime/paired-cache-v1. It contains 127 training and 16 validation recordings. The head consumes 8 frames at 8 FPS with 300 CNN features and 18 object features per frame. The intent labels are INTERACTION, PASSTHRU and WAIT. To repeat the selected training configuration, run the next command in a separate extracted copy because it exports to models/intent_yolo_fusion.pth:

    .\run_yolo_python.ps1 tools/train_fusion_model.py --learning-rate 0.00001

Rebuilding the cache from raw footage also needs the original repository's active training selection, annotations and sampling manifest, with valid paths to the source recordings and prepared frames. These research inputs are separate from the viewer package.

## Evaluation status
Live RealSense performance and the 16 reserved test recordings await evaluation. The existing splits share sessions and participants. Overlap with YOLO training sources remains unverified. The replay verifies application operation. Use fresh lab recordings for the planned comparison. The local demo footage, metadata and generated caches are excluded from ordinary Git staging. The two trained models plus the baseline checkpoint use the included Git LFS rules for .pt and .pth.
