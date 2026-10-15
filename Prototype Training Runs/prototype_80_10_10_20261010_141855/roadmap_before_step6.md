# Small-scale prototype roadmap
Updated October 7, 2026. Active pilot: **8 frames per input at 8 fps**.
Steps 1–5 are complete for a conservative, AI-reviewed within-session pilot. A short training smoke test passed; full prototype training and accuracy evaluation remain pending.

| Step | Work | Status |
|---|---|---|
| 1 | Import and verify data | Complete: 59 trials, 3,607 frames; originals preserved and imported hashes verified. |
| 2 | Review labels and ranges | Complete for selected ranges: 131 segments, 2,613 labeled frames; 994 uncertain frames UNASSIGNED. This is not exhaustive labeling or independent human ground truth. |
| 3 | Connect ranges and frame timing | Complete: 93 segments containing 2,413 frames materialized into training-format folders. 38 segments containing 200 frames held out for length. All extracted frames checked against corresponding original video frames and capture timestamps. |
| 4 | Prepare splits and verify loaders | Complete for pilot: original 36/11/12 trial split preserved; 57/17/19 exported segments. Explicit within-session option added; source-trial leakage remains prohibited. All actual loaders verified at eight frames; 66 automated tests passed. CUDA available. |
| 5 | Run model training smoke test | Complete: pretrained ResNet101 + LSTM initialized; two epochs on six balanced training segments and three validation segments; 12 weight updates; finite losses/gradients; checkpoint saved/reloaded with identical predictions; incompatible 16-frame setting rejected. |
| 6 | Train small prototype | NEXT: choose class balancing from training data, train reproducibly, monitor validation loss and per-class performance, retain best validation checkpoint. Start a deliberate full run; the smoke checkpoint is not a validated model. |
| 7 | Evaluate frozen prototype | Pending: reserved-test precision/recall/F1, confusion matrix and source-trial counts; disclose single-session limits. |
| 8 | Try live inference | Pending: matching eight-frame/eight-fps checkpoint settings; measure latency and errors near edges, deeper workspace and task interactions. |
| 9 | Expand and repeat | Pending: planned 60 INTERACTION / 60 PASSTHRU / 40 WAIT across more sessions/times of day; genuine session-held-out evaluation. Compare eight versus sixteen frames on validation data before final test evaluation. |

## Prepared split
These are segments, not additional independent recordings.

| Split | INTERACTION | PASSTHRU | WAIT | Segments | Source trials |
|---|---:|---:|---:|---:|---:|
| Training | 12 | 9 | 36 | 57 | 36 |
| Validation | 4 | 3 | 10 | 17 | 11 |
| Testing | 4 | 3 | 12 | 19 | 12 |
| Total | 20 | 15 | 58 | 93 | 59 |

WAIT dominates the training selection; class balance must be addressed during training design.
Take 18 remains corrected to INTERACTION for its seven-frame workpiece-manipulation core. That core is excluded from this unpadded eight-frame export, although other usable segments from that trial remain.

## Frame labels and active rule
- WAIT: hands outside the tabletop or resting at its edge, without task handling.
- PASSTHRU: hands farther into the workspace without task handling, including stationary hands.
- INTERACTION: task-tool/workpiece handling, including at an edge.

Each reviewed range assigns its class to every frame inside it. A single original trial can contain several classes. Separate exported folders prevent windows crossing a reviewed boundary. Unassigned frames are excluded rather than forced into a class; short labeled segments remain in the audit.
The edge/interior boundary is judged visually and uncertain transitions stay unassigned.

Prototype 8 Frame Pilot/per_frame_labels.csv lists all 3,607 source-frame positions, capture times, labels or UNASSIGNED, export inclusion and subset.

## Verification
All extracted JPEG hashes match the imported inventory. Source videos decode to matching frame counts; capture timestamps agree with the review inventory. Corresponding original/extracted images passed approximate resize/JPEG comparison: maximum frame mean absolute difference 5.826 on a 0–255 scale, threshold 12. This is approximate visual correspondence, not pixel-identical proof.
Actual training, validation and test loaders were exercised. Each sample has shape 8 x 3 x 150 x 150; training/validation tensors were checked for finite values. Validation uses one deterministic window per segment in this initial setup.
All 66 timing, temporal-window and split-protection tests passed. An initial test attempt hit a Windows temporary-directory permission error; rerunning in a fresh project-local directory resolved it. Dependency deprecation warnings remain.

## Artifacts in Prototype 8 Frame Pilot
- annotations.json and frames/: training-format labels and physically separated selected frames.
- frames/sampling.json: source indices, timestamps, participants and real session IDs.
- reviewed_ranges_snapshot.json: frozen review used for this export.
- pilot_config.json: explicit eight-frame/eight-fps configuration.
- preparation_summary.json and loader_verification.json: verified results.
- held_out_short_segments.json: 38 segments excluded for length.
- frame_correspondence.json: source-video comparison results.
- per_frame_labels.csv: complete frame-level label and inclusion ledger.

## Continue or reproduce
Run from the project root using the intent-gpu Python environment.

Check only, without training:

    python tools/run_reviewed_pilot.py

Start training later, when proceeding to the model-training stages:

    python tools/run_reviewed_pilot.py --train --epochs 10

The launcher reads the saved pilot configuration and enables the explicit within-session option. Training runs get separate output directories under Prototype Training Runs.

Prepare a separate sixteen-frame comparison later:

    python tools/prepare_reviewed_pilot.py --window 16 --output "Prototype 16 Frame Pilot"
    python tools/run_reviewed_pilot.py --dataset "Prototype 16 Frame Pilot"

The sixteen-frame export requires sixteen real sampled frames and may contain fewer examples. Retrain with matching settings; do not silently change an eight-frame checkpoint to sixteen frames. Global repository defaults remain unchanged; the pilot configuration explicitly selects eight.

## Limits
All trials are from session_20260929; both participants occur across splits. This is a trial-held-out within-session pilot, not a new-session or new-person evaluation. Multiple segments are not independent recordings.
AI-reviewed labels are not guaranteed perfect. No model training, accuracy evaluation or live inference has been performed in steps 1–4.


## Recording-folder reclassification — October 7, 2026
Takes 18, 44, 47 and 53 were moved from PASSTHRU to WAIT with renamed AVI/JSON pairs and updated metadata/references. Recording folders now contain 20 INTERACTION, 15 PASSTHRU and 24 WAIT videos. These four remain mixed source videos; only reviewed WAIT segments are used by the current pilot. Prepared segment counts and splits are unchanged. Previous original-category counts above describe the pre-move inventory. The move log and metadata backups are in Prototype Dataset Review/review/recording_reclassification_20261007.

## Step 5 results — October 7, 2026
A smoke test is a short engineering check of the training workflow, not a model-accuracy benchmark.

- Model: ImageNet-pretrained ResNet101 feature extractor plus newly initialized LSTM/classification layers.
- Input: eight real frames sampled at eight fps, 150 x 150 RGB.
- Device: NVIDIA GeForce RTX 5070 Laptop GPU.
- Two epochs on six training segments (two per class), validation on three segments (one per class). These are segments, not nine independent recordings.
- Twelve optimizer steps completed; finite model outputs, losses and gradients. Classifier weights changed.
- Training loss: 1.1188 then 1.1647. Validation loss: 1.0990 then 1.1004. No consistent loss improvement was demonstrated.
- Saved model and optimizer state, reloaded successfully with zero prediction difference on the checked input; resume advances to epoch 3.
- Checkpoint loading correctly rejects switching to a sixteen-frame input configuration.
- Reserved test set was not evaluated. No reliable accuracy claim or readiness for live use follows from this test.
- Roughly 42 seconds including pretrained-weight download; peak allocated GPU tensor memory approximately 1,224 MB. This is not a whole-GPU memory or inference-latency benchmark.
- Existing dependency deprecation warnings did not prevent the test.

Results: Prototype 8 Frame Pilot/latest_smoke_test.json.
Reproduce: python tools/smoke_reviewed_pilot.py.
The smoke checkpoint is for verification only. Full prototype training is step 6.
