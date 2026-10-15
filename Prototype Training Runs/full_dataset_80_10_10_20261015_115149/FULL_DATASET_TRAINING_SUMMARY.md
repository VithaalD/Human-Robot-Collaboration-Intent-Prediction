# Full dataset retraining - October 15, 2026

Completed retraining from the supplied October 15 archive roster. The active prototype uses the second run, best epoch 12, selected by validation loss. Training completed 37 epochs, including 25 epochs without improvement after the best checkpoint.

## Dataset and split

The archive contains 163 folders but 159 unique recordings: 59 INTERACTION, 60 PASSTHRU, 40 WAIT. Four exact duplicate sequences appeared as both PASSTHRU and WAIT. The previously corrected WAIT copies were retained; the PASSTHRU duplicates were excluded from this training export. No original archive or recording was deleted. There is one fewer unique INTERACTION recording than the requested 60.

| Split | INTERACTION | PASSTHRU | WAIT | Total |
|---|---:|---:|---:|---:|
| Training | 47 | 48 | 32 | 127 |
| Validation | 6 | 6 | 4 | 16 |
| Reserved test | 6 | 6 | 4 | 16 |
| Total | 59 | 60 | 40 | 159 |

This is approximately 80/10/10 (79.87/10.06/10.06 percent), stratified by category and participant, seed 20261015. Each source stays in exactly one split. Participants and recording sessions overlap across splits; it is not a new-person or new-session evaluation. Some older sources were used in previous experiments, so the test set is not a wholly unseen research benchmark.

## Process

1. Imported the archive without executing the partner's scripts. Checked annotations, sequential frame filenames, frame counts, duplicate contents and matching local recording metadata.
2. Read the partner preprocessing code: FFmpeg resizes to height 240 and saves JPEG frames; it does not trim action boundaries or perform label review. Replaced its split with the requested approximate 80/10/10 split.
3. Matched all 159 unique archive clips to original local AVI files. Checked first, middle and last images against the archive (477 alignment spot checks). Retained the longest previously reviewed same-class range for each of 59 older sources. Used supplied clip labels for the other 100 recordings; those labels have not received a new frame-by-frame review.
4. Sampled original videos using actual capture timestamps at 8 fps. Rebuilt 8,420 lossless PNG frames with shared RGB/bilinear 150x150 preprocessing. Every prepared frame produced an exactly equal tensor to the live preprocessing path. This avoids adding the partner export's JPEG compression to model inputs.
5. Trained the same ResNet101 plus three-layer LSTM architecture. The ImageNet backbone and its batch-normalization statistics are frozen; the projection, LSTM and classifier are trained. Used random consecutive eight-frame training windows, class-balanced replacement sampling, AdamW and deterministic center eight-frame validation windows. One selected window is scored per validation clip.
6. Ran two learning rates on the identical split. The first (0.001) selected epoch 12, validation loss 0.55673 and 10/16 correct, missing every INTERACTION. A bounded lower-rate run (0.0001) selected epoch 12, loss 0.15766 and 15/16 correct. Both runs are preserved. Validation was used for this selection; test predictions were never computed.
7. Reloaded the saved checkpoint; reproduced its validation metrics and checked the cached-feature training path against the full model. Verified all 16 validation windows through the actual inference engine on original AVI frames: exactly equal prepared/original tensors and probabilities, and identical labels. A ten-second threaded video replay produced predictions and stopped cleanly; weights were unchanged.
8. Selected the verified new checkpoint for Start Small Prototype.cmd, preserving the previous active selection and documentation in this run folder. Live lab-camera testing remains pending.

## Validation results

| Category | Correct / evaluated | Recall |
|---|---:|---:|
| INTERACTION | 6 / 6 | 100% |
| PASSTHRU | 5 / 6 | 83.33% |
| WAIT | 4 / 4 | 100% |
| Overall | 15 / 16 | 93.75% accuracy |

Macro F1: 0.9441. The single error was session_20261013_take_0053_vithaal_camera_1_PASSTHRU_c43355ab, predicted INTERACTION.

These are small validation-set results used for model selection, not final test or continuous live accuracy. They are not directly comparable to prior runs on different splits. New-clip transition frames can differ from their supplied clip labels. Short recorded playback is an engineering check, not a labeled accuracy test.

## Next steps

Review action boundaries and the missed PASSTHRU clip; evaluate the reserved 16 clips once model choices are fixed; then test continuous live lab footage and a genuinely fresh recording session. Locate or record the missing INTERACTION clip if the exact 60/60/40 target is required.

## Reproducibility and files

- Dataset: Full Dataset 20261015/prepared_original_frames; rebuild_verification.json contains source hashes, exclusions and selected frame indices.
- best_checkpoint.pth, training_report.json, run_config.json, split_manifest.json and history.json describe this run.
- training_comparison.json records both runs; live_input_validation.json records original-video checks.
- Tools: prepare_full_dataset_20261015.py; train_full_dataset_20261015.py --learning-rate 0.0001 --patience 25; validate_full_dataset_20261015.py.
- The preparation tool intentionally refuses to overwrite an existing prepared dataset.
