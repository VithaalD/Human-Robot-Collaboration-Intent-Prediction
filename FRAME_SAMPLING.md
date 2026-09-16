# Frame sampling: training and inference

The original training loaders hardcoded 16 frames per clip even when `--sample_duration` requested another value. The desktop and terminal inference engines defaulted to 8. The CNN-LSTM accepts either sequence length, so this disagreement could change predictions without causing a tensor-shape error.

The default is now **16 frames** in training, validation, the desktop monitor, and the terminal runner. Both training datasets receive the selected duration. Validation always returns exactly that many frames; short clips repeat their frame indices, and long clips are cropped. Transforms no longer change the stored dataset indices. The monitor offers 8 or 16 frames, and training/terminal options remain configurable.

## Historical checkpoint limitations

The supplied epoch-196 checkpoint does not record its training frame duration. **16 matches the available training source's default; it does not prove how that historical checkpoint was trained.** An explicit 8-frame override remains available for controlled comparisons. Use labeled recordings that were not used for training to decide which setting performs better; do not assume changing the window improves accuracy.

A 16-frame window contains twice as many frames as an 8-frame window and requires more computation per prediction. Its elapsed time depends on the input frame rate. For example, uninterrupted input at 30 frames/second represents approximately 0.53 seconds of samples for 16 frames and 0.27 seconds for 8 (the first-to-last timestamp spans are 0.50 and 0.23 seconds). Recorded/extracted training frames may use a different frame rate, so matching counts alone does not guarantee matching motion timescales.

## Running a selected duration

From `intentpredictionattempt4-main`, run the PowerShell launcher with `-SampleDuration 16` (the default) or `-SampleDuration 8`. The Python engine uses `--sample_duration 16` or `--sample_duration 8`. In the desktop monitor, choose the corresponding frame count before starting inference.

The terminal launcher uses `INTENT_PYTHON` when set, otherwise the installed `~/.venvs/intent-gpu` environment when available, then the older `DEEPLABCUT` environment as a fallback.

For future training, pass `--sample_duration 16` or your deliberately chosen duration to `main.py`. Training and validation now both honor it. This change does not start training or modify existing weight files.

## New checkpoint metadata

Newly saved checkpoints contain an `input_config` dictionary with frame duration, input resolution, pixel normalization divisor, mean, standard deviation, RGB color order, class count, and the exact class-index order from the annotations. Inference checks recorded settings before constructing the model and stops with a corrective message if they disagree. For example, an 8-frame checkpoint loaded with 16 frames produces a message to use `--sample_duration 8`. The engine exposes `--sample_size`, `--norm_value`, `--mean`, `--std`, and `--class_labels` for the other recorded settings.

Older checkpoints without metadata remain usable and produce a warning that historical training settings are unknown. Resuming training also checks available metadata and loads tensors onto the selected device. Checkpoint loading uses PyTorch's restricted `weights_only=True` format on supported versions; old versions without that option retain their existing loader behavior.

## Verification

`tests/test_temporal.py` checks both source trees with generated image fixtures: training and validation output exactly 8 or 16 frames, duration reaches both dataset constructors, crops do not mutate indices, the final random-crop window is reachable, metadata rejects conflicting inference settings before model allocation, old checkpoints support either duration, and training resume rejects a recorded duration mismatch. No training run is needed for these checks.
