# Recording new footage and using the GPU

## Record without running a model

1. Connect the camera and double-click **Start Recording.cmd** in the project folder.
2. Select the camera and requested resolution. Begin with 1920 x 1080 at 30 FPS; try 1280 x 720 if the camera cannot maintain that rate.
3. Use the **Record only** mode. It previews/saves camera frames without loading PyTorch or a model checkpoint. Start the camera preview and check the actual delivered resolution.
4. Enter a session ID (for example `20260915_A`), participant ID (`P01`), camera-setup ID (`CAM02_ANGLE01`), and tool (`screwdriver`). Use a new camera-setup ID when moving/replacing the camera, and a new session ID for a new collection session. Participant IDs should be consistent across sessions.
5. Select the scenario `INTERACTION`, `PASSTHRU`, or `WAIT`, then record the take. Take numbers are allocated automatically. The settings and numbering are persisted locally.
6. Inspect several pilot takes before collecting the full dataset. Keep the screwdriver in the scene for all classes, including waiting and pass-through. Save clean camera footage, not desktop screenshots.

Outputs are an original-resolution MJPEG AVI and matching JSON metadata under **Camera Recordings** in this project. This folder is ignored by Git. Preserve each AVI/JSON pair and back them up separately. Metadata includes collection identifiers, scenario, camera information, actual dimensions, requested/reported FPS, frame timestamps, and capture-timing summary. A scenario is a planned take label, not a verified per-frame annotation.

AVI playback is encoded at nominal FPS; actual capture timing may vary. Use JSON timestamps for event annotations and later resample consistently when preparing training data. Timing-gap estimates are diagnostics, not proof of a hardware frame-drop count. The same recorder also works during webcam inference, but record-only avoids inference load and the checkpoint requirement.

Keep all clips/frames from the same take in the same dataset split. Group by session and, when evaluating unfamiliar people, participant. Never scatter adjacent frames between training and testing. Review action-onset/end times after capture.

## What changed about the frame window

The original matching trainer cropped 16 frames even if another `--sample_duration` was requested; the monitor/runner defaulted to 8. At 30 frames per second that is approximately half a second versus a quarter second of visual context. The LSTM can execute either length, which is why no dimension error appeared, but different temporal context can change predictions.

Training and validation now honor `--sample_duration`; the matching engine, monitor and runner default to **16**. Use the same setting when training and running a checkpoint. New checkpoints record input configuration; the engine checks it before inference and rejects mismatches instead of silently proceeding. Explicit 8-frame runs remain available for legacy comparisons.

The existing epoch-196 checkpoint does not identify its historical training duration. The 16-frame default follows the supplied training source; it does not prove the old checkpoint was trained with 16 or that 16 improves its accuracy. Compare settings on held-out data, and retrain for the new camera. Matching frame count alone does not match elapsed time if capture/extraction FPS differs. See [FRAME_SAMPLING.md](FRAME_SAMPLING.md) for details.

## GPU environment

Run **setup_gpu.ps1** once to create `%USERPROFILE%\.venvs\intent-gpu`. It installs PyTorch 2.10.0 and TorchVision 0.25.0 with CUDA 12.8, plus Ultralytics and the project training dependencies. This matches the original PyTorch/TorchVision version pair while enabling this laptop's RTX 5070. The existing DEEPLABCUT environment stays separate. The virtual environment uses the existing Python installation as its base, so retain that base installation.

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_gpu.ps1
```

The monitor launcher and terminal runner prefer the GPU environment when present. Set `INTENT_PYTHON` to override the Python executable. **Open GPU Terminal.cmd** opens a terminal with this environment active in the project folder.

Run the repeatable hardware checks there:

```powershell
python tools\verify_gpu.py
python -m pip check
```

The GPU check performs an optimizer step, an LSTM forward/backward pass, detection NMS, and a YOLO11 nano forward/backward pass on synthetic inputs. No model weights are downloaded for this check. Passing it establishes execution support, not screwdriver detection accuracy. No screwdriver model has been trained yet.

The custom intent trainer still requires the prepared frame folders and its annotation JSON. From `intentpredictionattempt4-main`, an initial command for the three-class dataset is:

```powershell
python main.py --dataset ucf101 --video_path "D:\dataset\frames" --annotation_path "D:\dataset\annotations.json" --n_classes 3 --sample_duration 16 --sample_size 150 --batch_size 1 --num_workers 0 --use_cuda --gpu 0
```

Replace the two dataset paths. Start with batch size 1 on the 8 GB GPU, then measure available memory before increasing it. The annotation class order must be `INTERACTION`, `PASSTHRU`, `WAIT` when using the existing monitor's labels. Detection training is a separate pipeline and needs reviewed person/screwdriver bounding boxes, for example exported from a Roboflow object-detection project. Future `.pt` detector checkpoints and `.pth` intent checkpoints are configured for Git LFS.

## Verify before the full collection

- Confirm focus and screwdriver visibility during motion and partial occlusion.
- Check actual resolution, achieved capture rate and timestamp gaps using the saved metadata.
- Keep lighting/camera conditions represented across all three classes.
- Reserve independent sessions for validation and final testing.
- Record at least a few pilot takes per class before collecting all 160.
