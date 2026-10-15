# Weekly update — Human–Robot Intent Prediction
Progress through October 7, 2026.

## Slide 1 — Dataset preparation and labeling
- Imported and checked 59 recorded trials containing 3,607 extracted frames.
- Completed an AI-assisted visual label review; corrected a PASSTHRU example showing workpiece manipulation.
- Defined the labeling rule: edge-resting hands = WAIT; hands farther into the workspace without task handling = PASSTHRU; tool/workpiece handling = INTERACTION.
- Connected reviewed ranges to the training pipeline: 93 usable segments containing 2,413 frames.
- Prepared 57 training, 17 validation and 19 test segments, keeping every source trial in one split.
- Current recording folders: 20 INTERACTION, 15 PASSTHRU, 24 WAIT. Four mixed recordings were reorganized under WAIT because their retained pilot segments are WAIT; this does not label their entire videos WAIT.

## Slide 2 — Training pipeline verified
- Configured the pilot for eight-frame inputs at eight fps.
- Verified actual data loading and passed 66 timing, frame-window and split-protection tests.
- Completed a GPU model smoke test: two epochs on six balanced training segments, with three validation segments.
- Confirmed forward prediction, loss calculation, finite gradients, 12 weight updates, and successful checkpoint saving/reloading.
- Reloaded predictions matched exactly; incompatible input-window settings were rejected.

## Slide 3 — Current limitations and next steps
- Steps 1–5 complete; full prototype training is next.
- The smoke test confirms the workflow runs, not useful classification accuracy. Loss did not improve consistently over this tiny run.
- The reserved test set has not been evaluated.
- All recordings are from one session; current splits support a within-session pilot only.
- WAIT is overrepresented among prepared segments; address class balance during full training.
- Collect more varied INTERACTION and PASSTHRU examples, plus additional recording sessions and lighting conditions toward the 60–60–40 target.
- Next: train a baseline, select a checkpoint using validation results, evaluate the reserved test set, then test live behavior.
