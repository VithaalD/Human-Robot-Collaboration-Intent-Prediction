# Step 6: trained small prototype
Completed October 10, 2026.

Trained the eight-frame/eight-fps prototype using the 59 previously reviewed September 29 recordings, not unreviewed October 8 footage. New source-group split: 47 training, 6 validation, 6 reserved test recordings (79.66%, 10.17%, 10.17%). All segments from a recording stay together. This supersedes the older pilot split for this run; the original pilot export remains preserved.

| Split | Recordings | INTERACTION segments | PASSTHRU segments | WAIT segments | Total segments |
|---|---:|---:|---:|---:|---:|
| Training | 47 | 16 | 12 | 46 | 74 |
| Validation | 6 | 2 | 1 | 6 | 9 |
| Test | 6 | 2 | 2 | 6 | 10 |

Used an ImageNet-pretrained ResNet101 with its image backbone and batch-normalization statistics frozen. Trained the 300-dimensional projection, three-layer LSTM and classifier. Cached frozen image features for efficiency; the final checkpoint contains the full original CNN-LSTM model. Training sampled classes with inverse-frequency replacement weights calculated only from training segments, and selected random consecutive eight-frame windows. Validation uses one deterministic center window per segment. No test images were loaded or scored.

Reproducibility: seed 20261010, AdamW, initial head learning rate 0.001, weight decay 0.001, batch size 8, gradient clipping 5, validation-loss learning-rate reduction, at most 100 epochs, early stopping after 15 epochs without validation-loss improvement of at least 0.0001. Label-only grouped stratification was fixed before training. Class balance does not add independent footage.

Training stopped after 43 epochs. Best checkpoint: epoch 28, selected by lowest unweighted validation cross entropy (0.2343). Validation accuracy: 8/9 (88.9%); macro F1: 0.859. The one error was a WAIT segment from take 42 predicted as PASSTHRU.

| Class | Correct / validation support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| INTERACTION | 2/2 | 1.000 | 1.000 | 1.000 |
| PASSTHRU | 1/1 | 0.500 | 1.000 | 0.667 |
| WAIT | 5/6 | 1.000 | 0.833 | 0.909 |

These are checkpoint-selection validation results, not test performance. Six recordings and one PASSTHRU validation segment are too few for reliable generalization claims. Both participants occur across splits in one session; reviewed labels are AI-reviewed, not independently confirmed human ground truth. October 8 footage still needs reviewed ranges before training inclusion.

Verification passed: all 93 segment provenance/timing records; disjoint source recordings across splits; at least eight distinct real frames per segment; finite training losses and clipped gradient norms; checkpoint reload reproduced validation loss; original full-model predictions matched cached-feature predictions within 0.00000650 on the checked validation input.

## Files and next step
- best_checkpoint.pth: full model, optimizer state, epoch, input contract and training settings.
- annotations.json: active labels and 80/10/10 assignments for this run; frames remain in Prototype 8 Frame Pilot/frames.
- split_manifest.json and split_verification.json: source groups, segment counts and verification.
- history.json and training_report.json: epoch history, validation metrics and predictions.
- run_config.json: reproducible training choices.
- training_script.py: script snapshot (reference copy; run the tools/ version from the project).

Use the saved checkpoint's input_config: RGB, 150x150 bilinear resize, divide by 255, ImageNet mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225], 8 frames at 8 fps; class order INTERACTION, PASSTHRU, WAIT. This differs from the old smoke test's normalization. Live inference must adopt this exact contract in step 8.

Reproduce from the project root with the intent-gpu Python environment:

    python tools/train_prototype_80_10_10.py

Step 7 is next: evaluate this frozen best checkpoint on the 10 reserved test segments from six recordings, without further checkpoint selection on test scores. Step 8 then connects and checks live inference. No live robot integration was changed here.
