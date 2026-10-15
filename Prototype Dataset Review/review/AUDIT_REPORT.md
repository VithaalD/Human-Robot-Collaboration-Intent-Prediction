Current status (October 7, 2026): reviewed ranges are connected through the separate Prototype 8 Frame Pilot export. See PROTOTYPE_ROADMAP.md at the project root. Original review details below describe annotation decisions.

# Updated label audit — edge hands are WAIT
This update supersedes the earlier no-hands-on-table policy. Originals and the complete v1 review are preserved.

## Active labeling rule
- WAIT: hands outside the tabletop or resting immediately at its perimeter, without tool/workpiece handling.
- PASSTHRU: hands/arms farther into the workspace without task handling, including stationary hands.
- INTERACTION: handling the task tool or manipulating/stabilizing the workpiece, including at an edge.

The perimeter is judged visually. No physical width or calibrated pixel boundary was supplied. Borderline positions and transitions remain unassigned rather than being forced into a class. Moving through the interior takes PASSTHRU even if the hand starts or ends at the edge.

## What changed
The machine-readable review now supports multiple labeled segments per source trial. Clear opening/ending WAIT intervals and selected longer edge holds have been added; action cores are retained. These segments are a conservative selection, not exhaustive labeling of every frame. Original images and partner annotations remain unchanged. Take 18's workpiece manipulation remains INTERACTION.

All 20 original WAIT trials remain accepted. Previously excluded edge-hand footage is no longer automatically disallowed. Deeper non-task stationary hands are PASSTHRU under the active rule; excluded transition footage is not automatically restored.

## 8 fps versus 8 or 16 frames
The code's sample_fps default is 8; sample_duration default is 16. Both 8-frame and 16-frame inputs are supported. At 8 fps, these collect roughly 1 second and 2 seconds of samples respectively (first-to-last spans 0.875 and 1.875 seconds at exact 8 fps). The earlier 16-frame flags were only checks against the current default, not a requirement for valid footage. Short clips can be repeated by existing loader logic, but that does not add motion information.

The CSV and JSON now include both 8-frame and 16-frame length flags. No training/inference defaults were changed. Use the same sample rate and window length for training and inference. Neither length has been demonstrated superior on these data.

## Remaining limits
AI visual review is not independent human ground truth. Old all-frame contact sheets were supplemented by targeted edge/context reinspection. Some very short WAIT segments are retained as annotations even though they are too short for an unpadded 8-frame input.

Selected ranges have been materialized in a separate eight-frame pilot export and verified with the real loaders; no model has been trained. All segments from a trial must remain in the same split. All source trials are still from one recording session, so cross-session performance remains untested.

The CSV now has one row per SEGMENT, not one row per trial. The v2 JSON uses a segments list. The previous report and single-core annotations are in previous_policy_v1 for provenance only.

## Updated selection counts
- source_trials: 59
- annotated_segments: 131
- annotated_frames: 2613
- segments_at_least_8_frames: 93
- segments_at_least_16_frames: 53
- new_wait_frames: 639
