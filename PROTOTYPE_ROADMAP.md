# Prototype roadmap
Updated October 15, 2026. Active model uses the full supplied archive roster and eight frames at eight fps.

| Step | Status |
|---|---|
| 1. Import and verify | Complete: 163 archive folders, 159 unique recordings; four conflicting duplicate PASSTHRU copies excluded in favor of corrected WAIT copies. |
| 2. Review labels/ranges | Previous reviewed same-class cores retained for 59 recordings. Other 100 use supplied clip labels; new frame-level review remains pending. |
| 3. Build training data | Complete: 8,420 lossless PNG frames rebuilt from matching original videos with live preprocessing. |
| 4. Split and preprocessing checks | Complete: 127/16/16 sources, approximately 80/10/10. No source overlaps. All 8,420 tensors exactly match live processing. |
| 5. Engineering checks | Complete: saved checkpoint reload and cached/full-model checks passed. |
| 6. Train and validate | Complete: selected lower-learning-rate run; 37 epochs, best epoch 12. Validation 15/16, macro F1 0.9441. |
| 7. Reserved evaluation | Pending for this checkpoint: 16 clips reserved and unscored. Historical model scores do not apply. |
| 8. Application integration | Complete for recorded input: exact parity on 16 validation windows and successful replay. Start Small Prototype.cmd selects this model. Live lab test pending. |
| 9. Improve and expand | Review boundaries and missed PASSTHRU; test fresh recording sessions; locate/record one missing INTERACTION clip. |

## Active split

| Split | INTERACTION | PASSTHRU | WAIT | Total recordings |
|---|---:|---:|---:|---:|
| Training | 47 | 48 | 32 | 127 |
| Validation | 6 | 6 | 4 | 16 |
| Reserved test | 6 | 6 | 4 | 16 |

One prepared clip per unique source, with eight-frame windows drawn from each clip. Participants and dates overlap across subsets. Some older sources appeared in previous experiments. This is not a fresh-session benchmark.

Validation: INTERACTION 6/6, PASSTHRU 5/6, WAIT 4/4. The missed PASSTHRU was predicted INTERACTION. Validation was used to choose between two training runs; it is not final test accuracy.

Active run: Prototype Training Runs\full_dataset_80_10_10_20261015_115149. Read its FULL_DATASET_TRAINING_SUMMARY.md, training_report.json and live_input_validation.json. Previous datasets, checkpoints, and documentation snapshots remain available.
