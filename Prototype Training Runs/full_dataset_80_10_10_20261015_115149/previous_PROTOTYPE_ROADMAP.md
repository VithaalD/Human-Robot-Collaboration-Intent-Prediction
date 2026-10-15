# Small prototype roadmap
Updated October 12, 2026. Active model is trained on original-video frames with shared live-input preprocessing. Input remains eight frames at eight fps.

| Step | Current status |
|---|---|
| 1. Import and verify | Original 59-recording pilot verified; source recordings preserved. |
| 2. Review ranges | 131 selected ranges; 93 long enough for eight real frames. October 8 footage still requires segment annotation. |
| 3. Build training data | Rebuilt 2,413 frames into lossless PNGs directly from original AVI frames; separate Prototype Original Frame Pilot export. |
| 4. Split and preprocessing checks | Same 47/6/6 source split (approximately 80/10/10). All 2,413 frame tensors exactly match live preprocessing. Seventy-two regression tests passed. |
| 5. Training engineering checks | Checkpoint reload and original full-model/cached-feature parity passed. Original smoke-test history preserved. |
| 6. Train and initial validation | Complete for new dataset: 20 epochs, best epoch 5, 8/9 original-video validation windows correct. PASSTHRU validation example missed. |
| 7. Reserved evaluation | Old JPEG model was evaluated historically; NEW checkpoint has not been scored on test data. Old 10/10 result does not apply. Fresh held-out data needed for stronger independent evaluation. |
| 8. Application integration | New checkpoint selected by Start Small Prototype.cmd. Prepared/original window probabilities exactly match; recorded replay works. Live lab validation remains pending. |
| 9. Expand and repeat | Review October 8 recordings, add representative PASSTHRU and multi-session data, evaluate continuous transitions and fresh held-out recording days. |

## Active split
| Split | Recordings | INTERACTION | PASSTHRU | WAIT | Segments |
|---|---:|---:|---:|---:|---:|
| Training | 47 | 16 | 12 | 46 | 74 |
| Validation | 6 | 2 | 1 | 6 | 9 |
| Historical test | 6 | 2 | 2 | 6 | 10 |

All segments from one source stay together. Labels and source indices are unchanged. Both participants occur across subsets in one recording day; this is not new-person or new-session validation.

## Findings
The previous JPEG-trained model scored 6/9 on original-video validation windows. The new original-frame-trained model scores 8/9 through both file and live-engine paths, with zero input-tensor and probability differences in the parity checks. INTERACTION: 2/2; PASSTHRU: 0/1 (predicted WAIT); WAIT: 6/6. Macro F1 0.641. Fixing preprocessing consistency did not solve class reliability.

Shared processing: decoded BGR to RGB, bilinear resize to 150x150, divide by 255, ImageNet mean/std. Prepared PNGs store the resized RGB pixels losslessly. Eight-frame random training windows and center validation windows retained. This removes additional JPEG artifacts from training preparation, not compression already in original recordings.

Ten-second replay now displayed INTERACTION and WAIT; it was an engineering check rather than continuous labeled scoring. RealSense lab validation is still pending. No new-model test score exists.

## Active files

Run: `Prototype Training Runs/original_frames_80_10_10_20261012_151135`. See its `REBUILD_TRAINING_SUMMARY.md`, `training_report.json` and `live_input_validation.json`.

Dataset: `Prototype Original Frame Pilot`. Launcher: `Start Small Prototype.cmd`. Active selection: `Prototype Training Runs/active_prototype.json`.

Old datasets, models, investigation reports and roadmap snapshots are preserved. Current recording-folder counts last audited October 10: 40 INTERACTION, 17 PASSTHRU, 40 WAIT; these are not trained segment counts.
