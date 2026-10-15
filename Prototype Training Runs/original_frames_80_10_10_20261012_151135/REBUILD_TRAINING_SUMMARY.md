# Original-frame rebuild and retraining
Completed October 12, 2026.

## Outcome
The measured training/inference preprocessing mismatch is removed. All 2,413 prepared frames produced tensors exactly equal to those from their original decoded video frames through the monitor preprocessing (maximum absolute error 0). For all nine fixed validation windows, prepared-frame and original-video inference probabilities also matched exactly (maximum absolute error 0).

The new checkpoint correctly classified 8/9 validation windows through both paths. The previous model classified 6/9 of these original-video windows. This is an initial validation comparison, not a new independent test result. One PASSTHRU example is still predicted as WAIT.

## Process
1. Preserved the old JPEG dataset, trained model, reports and reviewed ranges. Reused the existing source-recording split: 47 training, 6 validation, 6 testing. No labels or frame selections changed.
2. Resolved current original AVI paths by source-recording UUID, verified capture timestamps and frame counts, and decoded recordings in order. Recorded original-video SHA-256 hashes.
3. Selected the same 2,413 source frames in 93 reviewed segments. Converted BGR to RGB, used shared bilinear 150x150 resizing and wrote lossless PNGs. These are already model-sized for efficient storage; resizing a 150x150 stored image to 150x150 leaves it unchanged. This avoids an additional lossy JPEG stage, but does not remove compression already present in the original AVI.
4. Added one shared preprocessing implementation used by training and the active desktop inference engine: RGB, bilinear 150x150, divide by 255, ImageNet channel mean/std. Read back every PNG and compared its prepared tensor with the actual engine transform applied directly to the original image. All 2,413 were exactly equal. Test frames were rebuilt and checked for preprocessing consistency only, without test predictions or scoring.
5. Retrained from the pretrained starting point, not from the old task-trained checkpoint. Kept the same seed, architecture, source split, 8-frame/8-fps contract, class-balanced sampling, random training windows, optimizer and validation-loss selection rule. Froze the ResNet101 backbone and trained the projection/LSTM/classifier, caching frozen image features for speed.
6. Training stopped after 20 epochs; epoch 5 had the best validation loss. Reloaded its checkpoint, checked full-model versus cached-feature predictions, and ran the actual monitor on the original-video validation windows. Saved per-window results.
7. Ran a ten-second original-video replay and checked startup, preview and clean thread shutdown. The replay displayed INTERACTION and WAIT; the old model's earlier replay displayed only WAIT. This is not a labeled continuous-video accuracy test. RealSense lab testing is still pending.
8. Connected Start Small Prototype.cmd to the new verified model using active_prototype.json. The old checkpoint remains available. UI setup and selected-checkpoint checks passed; 72 preprocessing/timing/recording regression tests passed (dependency deprecation warnings only).

## Dataset
| Split | Original recordings | INTERACTION segments | PASSTHRU segments | WAIT segments | Total segments |
|---|---:|---:|---:|---:|---:|
| Training | 47 | 16 | 12 | 46 | 74 |
| Validation | 6 | 2 | 1 | 6 | 9 |
| Historical reserved test | 6 | 2 | 2 | 6 | 10 |

All 59 sources are September 29 recordings. October 8 footage was not added because reviewed segment annotations are still needed. Folder labels alone were not treated as per-frame ground truth.

## Initial validation
| True class | Predicted INTERACTION | Predicted PASSTHRU | Predicted WAIT | Correct |
|---|---:|---:|---:|---:|
| INTERACTION | 2 | 0 | 0 | 2/2 |
| PASSTHRU | 0 | 0 | 1 | 0/1 |
| WAIT | 0 | 0 | 6 | 6/6 |

Overall accuracy: 8/9 (88.9%). Validation loss: 0.3963. Macro F1: 0.641. Per-class F1: INTERACTION 1.000, PASSTHRU 0.000, WAIT 0.923. The same aggregate 8/9 as the old JPEG validation score hides a different error distribution; PASSTHRU is not demonstrated reliable. The model makes no PASSTHRU prediction among these nine windows. Only one PASSTHRU validation segment is available, so no broad class-performance conclusion is justified.

No test-set predictions were computed for this new checkpoint. The old model's 10/10 test result does not apply to this model. That test set has already been inspected historically; subsequent scoring should be described as a follow-up evaluation. Fresh, independently reviewed recording-day data are needed for a stronger final test.

## Next steps
- Review and add more representative PASSTHRU segments, especially interior hand motion versus edge/walk-by WAIT, from additional recording sessions.
- Evaluate continuous action transitions and per-class validation performance, not overall accuracy alone. Any revised checkpoint-selection rule or hyperparameters must be chosen on validation data, not the test set.
- Test with the lab RealSense camera and confirm framing, lighting and 8-fps timing. No live-camera accuracy guarantee follows from preprocessing parity.
- Obtain the partner's original extraction script and exact settings if available, for provenance and confirming the old JPEG pipeline. It was not necessary for this rebuild and does not imply their frame extraction was inherently wrong; matching training and deployment preprocessing was the missing requirement.

## Artifacts
- Prototype Original Frame Pilot: separate lossless dataset, unchanged labels/split, sampling provenance and rebuild_verification.json.
- This run: best_checkpoint.pth, run_config.json, split_manifest.json, annotations.json, history.json, training_report.json, live_input_validation.json, checkpoint hash and script snapshots.
- Prototype Training Runs/active_prototype.json selects this verified checkpoint for the launcher.
- Original JPEG training reports and checkpoint are preserved; latest_step6.json remains historical.

Rebuild reproduction: tools/rebuild_original_frame_pilot.py refuses to overwrite an existing export. Training: python tools/train_original_frame_pilot.py. Initial validation: python tools/validate_original_frame_pilot.py. Use the intent-gpu Python environment from the project root. New runs are separate and do not automatically replace the active launcher model until validated and explicitly selected.
