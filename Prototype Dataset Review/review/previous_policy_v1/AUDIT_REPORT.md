# Dataset import and label audit — steps 1 and 2
Completed 2026-10-05. No training was run.

## Import results
Imported all 3,667 files from the supplied ZIP, preserving its directory structure and original annotations under imported_original. Every imported file was byte-verified against the ZIP using SHA-256. Original archive and Camera Recordings were not modified.
Archive SHA-256: ed520aa1590de2f5fbe392d5eb60c7b764e8ac2380af3a01300e7de5caf437d4

59 trials contain 3,607 readable JPEG frames, each 427 x 240. Trial lengths are 45–70 frames. All 59 have matching local recording metadata and matching frame/timestamp counts. Metadata-derived rates are approximately 7.98–8.01 fps. The original split is retained: training 36, validation 11, testing 12. There are 0 exact cross-trial duplicate JPEG hash groups.

## Visual review results
AI visual inspection covered all frames of all 59 trials using chronological contact sheets. Original 1920 x 1080 video frames were additionally inspected for take 18. This is a visual audit, not independent human ground truth or a guarantee of perfect frame-level boundaries.

| Class | Original trial labels | Reviewed action cores | Cores with at least 16 consecutive frames |
|---|---:|---:|---:|
| INTERACTION | 20 | 21 | 16 |
| PASSTHRU | 19 | 18 | 10 |
| WAIT | 20 | 20 | 20 |

All 20 WAIT trials pass the visible no-hand-in-workspace rule. Folded arms, hands at sides, and gestures outside the tabletop are acceptable. No idle tabletop-hand frames from other classes were relabeled WAIT.

Take 18 is a clear class mismatch: the hand grips and rotates the workpiece, despite the PASSTHRU label. The reviewed core (frames 31–37) is labeled INTERACTION. See take18_detail.jpg. The original annotation remains preserved.

21 trials are accepted over their entire length (20 WAIT and take 35, which holds/stabilizes the workpiece throughout). The other 38 have conservative action cores recorded instead of accepting the entire clip. The cores deliberately omit idle lead-in/tail periods and unclear transitions. These are not exhaustive action boundaries; other valid segments may exist.

13 reviewed cores are shorter than 16 frames: takes 5, 9, 12, 18, 19, 21, 23, 39, 44, 46, 47, 50, 53. They remain in the audit, flagged separately; none were padded, stretched, discarded from the source, or mixed with idle frames to meet a length target. The 16-frame column is only a length check for consecutive-frame input, not confirmation that the training loader can use them yet.

## Label policy and unresolved limits
INTERACTION includes visible workpiece stabilization/manipulation or handling the task screwdriver. Takes 36 and 41 mainly show tool handling rather than sustained workpiece use. This interpretation is documented explicitly.
PASSTHRU cores contain non-task arm/hand movement, sometimes with brief pauses. Takes 44, 46 and 47 especially contain long stationary tabletop-hand periods, excluded from the selected cores. Such stationary intrusion is never WAIT under your rule, but its final class requires an explicit policy before adding it to training.

All trials come from session_20260929; both participants appear across the existing splits. Trial separation alone does not establish generalization to another recording session or new people. No session identifiers or splits were invented or changed.

The audit cannot establish hidden intent, exact physical contact during occlusion, model accuracy, or performance with a moving robot. Original-resolution video was not reviewed in full for every trial; all extracted frames were reviewed, with targeted original-resolution inspection for the identified mismatch. Metadata/frame count agreement is not a complete source-to-extracted-frame correspondence or timing validation.

## Files and next-step handoff
- trial_label_audit.csv: one row per trial, original/reviewed class, selected source frame range, timestamps, length flag and rationale.
- reviewed_segments.json: the same decisions and policy in machine-readable form.
- index.html: browsable trial list with links to all-frame contact sheets.
- inventory.json and frame_sha256.json: provenance and import inventory.
- take18_detail.jpg: original-resolution evidence of the class mismatch.

The reviewed annotations are saved separately. No source frames were physically trimmed and no repository training annotations or code were overwritten. The manifest is NOT a drop-in training annotation file: the next stage must enforce its ranges or create trimmed folders, preserve 8 fps timing, and explicitly resolve the single-session pilot split limitation. A model has not been trained or tested.
