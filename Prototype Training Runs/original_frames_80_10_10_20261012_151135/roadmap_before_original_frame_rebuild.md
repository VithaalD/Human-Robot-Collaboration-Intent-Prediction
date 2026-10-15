# Small-scale prototype roadmap
Updated October 10, 2026. Active input: **8 frames at 8 fps**. Steps 1-7 are complete for the reviewed September 29 pilot. Step 8 software integration and recorded-video checks are complete; live lab validation remains pending because the RealSense camera was not connected.

| Step | Work | Status |
|---|---|---|
| 1 | Import and verify data | Complete: 59 original trials, 3,607 extracted frames; source correspondence verified. |
| 2 | Review labels and ranges | Complete for the original pilot: 131 selected ranges, 2,613 labeled frames; 994 uncertain frames excluded. October 8 footage still needs segment annotation. |
| 3 | Connect ranges and timing | Complete: 93 usable segments, 2,413 frames; 38 ranges too short for eight real frames excluded. |
| 4 | Prepare splits | Updated: user-requested 80/10/10 split by original recording, rounded to 47/6/6. No recording crosses splits. Original export and split preserved separately. |
| 5 | Smoke test | Complete October 7: training, gradients, checkpoint save/reload and input-contract checks passed. |
| 6 | Train small prototype | Complete October 10: 43 epochs, early stopping, best checkpoint epoch 28. Frozen pretrained ResNet101 backbone; trained projection, LSTM and classifier with balanced training sampling. |
| 7 | Evaluate frozen prototype | Complete October 10: frozen epoch-28 checkpoint correctly classified all 10 test segments from six recordings. Per-class precision/recall/F1 all 1.000; small same-session test only. Model unchanged. |
| 8 | Live inference | Software ready: prototype launcher, saved preprocessing, video replay and UI checks passed. About 61 ms median warmed inference. Live lab validation pending: RealSense unavailable; raw-video diagnostic scored 6/9 versus extracted-frame validation 8/9. |
| 9 | Expand and repeat | Pending: review October 8 ranges, record more INTERACTION and PASSTHRU, and evaluate across held-out recording days. Compare 8 versus 16 frames using validation data. |

## Active split for step 6
These are reviewed segments, not additional independent recordings.

| Split | Source recordings | INTERACTION | PASSTHRU | WAIT | Total segments |
|---|---:|---:|---:|---:|---:|
| Training | 47 | 16 | 12 | 46 | 74 |
| Validation | 6 | 2 | 1 | 6 | 9 |
| Reserved test | 6 | 2 | 2 | 6 | 10 |
| Total | 59 | 20 | 15 | 58 | 93 |

Actual source proportions: 79.66% / 10.17% / 10.17%. The new run's annotations.json is authoritative for this split; Prototype 8 Frame Pilot/annotations.json and per_frame_labels.csv retain the historical split and must not be used as the current split assignments.

## Training results
Best validation checkpoint: epoch 28, loss 0.2343, 8/9 correct (88.9%), macro F1 0.859. One WAIT segment was predicted as PASSTHRU. Validation includes only one PASSTHRU segment. These figures were used for checkpoint selection and are not a final test score or evidence of readiness for deployment.

All recordings in this run are from September 29, with both participants across splits. Labels were AI-reviewed, not independently confirmed human ground truth. The 38 October 8 recordings are not part of this trained prototype because they lack reviewed segment ranges.

Training uses frozen pretrained image features and learns the projection, temporal and classification layers. Inverse-frequency sampling addresses training class imbalance. The checkpoint includes the full model, preprocessing contract, optimizer state and training settings. Cached features were verified against full-model inference; checkpoint reload and source/timing checks passed.

## Current recording inventory and rules
After October 10 cleanup, Camera Recordings contains 97 videos: 40 INTERACTION, 17 PASSTHRU, 40 WAIT. Session dates: 59 September 29, 38 October 8. Folder counts are not training sample counts. Takes 89-104 were moved to WAIT; takes 105-106 still require mixed-segment review.

- WAIT: hands outside the tabletop or immediately at its edge, without task handling; includes ordinary walk-bys outside the workspace.
- PASSTHRU: hands/arms farther into the tabletop workspace without task handling.
- INTERACTION: task-tool or workpiece handling, including at an edge.

Reviewed ranges assign labels to their contained frames; uncertain transitions are excluded. Keep all derivatives of an original recording together when splitting data.

## Active artifacts

Training run: `Prototype Training Runs/prototype_80_10_10_20261010_141855`.

Read its `TRAINING_SUMMARY.md` for full results, `annotations.json` for active split assignments, `split_manifest.json` for source groups, and `best_checkpoint.pth` for the trained model. `Prototype Training Runs/latest_step6.json` points to the completed run.

Reproduce: `python tools/train_prototype_80_10_10.py` using the intent-gpu environment. Use RGB 150x150, pixel scaling by 255, ImageNet mean/std from checkpoint input_config, and eight frames at eight fps. The old smoke-test normalization is not compatible with this checkpoint.

The pre-step-6 roadmap is preserved as `roadmap_before_step6.md` in this run. Earlier source review and smoke-test artifacts remain unchanged.

## Step 7 results and hands-on testing
Reserved test: 10/10 correctly classified (INTERACTION 2/2, PASSTHRU 2/2, WAIT 6/6), macro F1 1.000, loss 0.079666. Used one fixed center eight-frame window per segment and the frozen best-validation checkpoint. Verified unchanged checkpoint hash and source-disjoint split. No test-based tuning.

These results cover six recordings from the original September 29 session, not continuous live video, new people or new recording days. The tiny class supports do not establish 100% real-world accuracy. The test set is now evaluated and should not be used to tune this model.

Read TEST_SUMMARY.md and test_report.json in the active training run. Prototype Training Runs/latest_step7.json contains the current evaluation report. Step 6 reports remain historical records of the training stage, when the test set was untouched.

Hands-on testing is next, in step 8: connect this saved model to video/live-camera inference using its exact preprocessing and sampling contract and verify displayed labels and latency. No further training or full 60/60/40 collection is required to try the prototype, the software connection is now completed, with live lab validation still pending (see step 8 results below).

## Step 8 implementation and results
Use Start Small Prototype.cmd; see SMALL_PROTOTYPE_QUICKSTART.md. Saved checkpoint metadata now configures monitor preprocessing. The dedicated launcher selects the trained model, eight frames and Video mode. Recording-only operation remains model-free.

Verified extracted-frame prediction parity (all nine prior outputs), wrong-duration rejection, timestamp-based video replay, preview, clean thread shutdown and actual desktop video startup/shutdown. Seventy regression tests passed. Median warmed single-window processing was about 61 ms; first replay prediction about 1.2 seconds after starting the loaded engine. These are engineering measurements, not action-response latency guarantees.

Material limitation: continuous take-2 replay displayed only WAIT. Direct original-video decoding of the nine validation windows gave 6/9 correct, versus 8/9 on extracted JPEGs. The model is sensitive to the input preparation path; exact cause requires further investigation. The 10/10 reserved extracted-window result must not be presented as live/raw-video accuracy. No weights were changed.

RealSense was not connected; only laptop/virtual cameras were detected. Software can now be tried, but live lab accuracy and transition behavior remain unverified. Connect the lab camera and perform the quickstart checks; investigate the raw/extracted preparation difference before relying on continuous predictions.

## Frame-path investigation — October 10
Controlled validation-only experiments confirmed compression sensitivity: original frames and resize-only variants scored 6/9; matching the extracted JPEG quantization restored 8/9. Expected source positions matched all 72 tested frames; sequential and seek decoding were identical. Ten JPEG quantization configurations were observed, so no single quality-setting fix has been established. Model and production preprocessing are unchanged. Next: build a separate original-frame dataset with shared train/inference preprocessing, verify tensor parity, retrain and validate on raw/continuous footage. See the active run's frame_path_investigation/REPORT.md for detailed evidence and the steps 1-8 history.
