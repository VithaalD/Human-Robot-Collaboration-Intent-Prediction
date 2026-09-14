# Human-Robot-Collaboration-Intent-Prediction
This is a project as a part of UGA's MERIT Lab

## Get the project

Install Git and [Git LFS](https://git-lfs.com/), then run:

```sh
git lfs install
git clone https://github.com/VithaalD/Human-Robot-Collaboration-Intent-Prediction.git
cd Human-Robot-Collaboration-Intent-Prediction
git lfs pull
```

The repository includes source code, desktop monitors, documentation, and three `.pth` checkpoint files stored through Git LFS. Raw videos, webcam recordings, caches, and training logs are excluded. Install Git LFS before cloning to retrieve the checkpoint contents.

## Project layout

- `Intent Monitor/`: current desktop monitor, camera selection, recording support, and inference engine. Start with [its setup guide](Intent%20Monitor/START_HERE.md).
- `intentpredictionattempt4-main/`: training/research code and the [terminal inference runner](intentpredictionattempt4-main/RUN_INTENT_V3_1.md).
- `test code and model weights for mecha lab-20260904T185537Z-1-001/`: supplied training/test code and model checkpoints.
- `Start Intent Monitor.cmd`: Windows desktop launcher.

## External videos and the active monitor checkpoint

Obtain the raw videos separately from the project owner. They are intentionally not published here. The current desktop monitor and terminal runner use the epoch-196 three-class checkpoint `cnnlstm-Epoch-196-Loss-0.01737015192823795.pth`, which currently lives with the external videos and is also not in this repository. Obtain that checkpoint separately; the included older checkpoints are not automatic replacements for it.

The runners look for `raw videos for mecha lab-20260904T185555Z-1-001/raw videos for mecha lab` inside the project or next to it. For a different layout, set `INTENT_DATA_DIR` to the inner data folder and `INTENT_WEIGHTS_PATH` to the full checkpoint path. The GUI also has Browse buttons; the PowerShell runner accepts `-DataDirectory`, `-WeightsPath`, and `-VideoPath`. Webcam inference needs the checkpoint but does not need prerecorded videos.

The Windows launchers currently expect Python at `%USERPROFILE%\anaconda3\envs\DEEPLABCUT\python.exe`. Collaborators can create that environment or run `python "Intent Monitor/intent_monitor.py"` using their own environment. The monitor needs PyTorch, TorchVision, OpenCV, Pillow, NumPy, and Tkinter. Consult the setup guide for the previously tested environment; the legacy requirements file is not a complete monitor environment specification.

## Contributing

Create a branch for your changes and open a pull request. Keep raw videos out of commits. New `.pth` files use Git LFS automatically through `.gitattributes`; commit that file when changing LFS tracking rules.

