# New-angle false-interaction investigation

## Finding and limits
The reported failure is confirmed. The provided 27.37-second video is a desktop screen recording of Intent Monitor, not the original camera stream. It shows high INTERACTION scores before hands enter and after they leave. The GUI scores change; this is not evidence of a frozen display.

Diagnostic replay used only the camera preview crop (x=58..626, y=214..641), excluding the GUI, starting after the initial OBS view. These inputs have already been resized and compressed, so replay probabilities cannot be compared exactly to the original displayed probabilities. Labels below are provisional visual annotations, not independently reviewed ground truth. All samples from this single recording are correlated.

## Controlled checks
- Current checkpoint, eight frames: all five sampled empty-scene/WAIT windows were classified INTERACTION. The five sampled handling windows were also classified INTERACTION.
- Sixteen frames: all five sampled WAIT windows were still classified INTERACTION.
- Grayscale: none of the five sampled WAIT windows were correctly classified WAIT; four became PASSTHRU and one remained INTERACTION. This does not isolate gearbox color from angle/background effects.
- Repeating the empty-scene frame near seven seconds eight times produced about 99.1% INTERACTION. Motion is not necessary to trigger this particular false positive.
- In nine reference windows from int1.mov, pass1.mov, and wait1.mov, the top labels matched the corresponding folder labels. This is a regression spot check, not held-out accuracy.
- Source inspection found that the engine's mean subtraction, RGB conversion, and 0..255 input scale agree with the supplied training-code defaults. The actual historical training command was not supplied, so this is not proof of every historical training setting.

The baseline INTERACTION probabilities for empty windows ranged approximately 60.3–99.2%; for handling windows they ranged 63.1–87.3%. These overlapping scores cannot be separated correctly by a single INTERACTION threshold on these examples.

## Interpretation
The most plausible explanation is poor generalization to the changed camera view and scene. Original examples use a high downward-facing view; this recording uses a low frontal view with a different background, object scale/position and green gearbox. Original example layouts also differ between classes, suggesting a possible background/object-position shortcut. The shortcut hypothesis is not proven causal; controlled new recordings are required to separate angle, color, background and action.

No checkpoint, threshold, frame-window setting, or raw prediction was changed. The recognition problem remains unresolved for this new view. No model was trained on this screen recording or falsely presented as independently validated.

## Software update provided
Intent Monitor now offers Record example: WAIT / PASSTHRU / INTERACTION and Record 10 sec. It saves camera images only to the project's Camera Recordings folder, using MJPEG AVI plus a JSON file of actual capture timestamps and source/scenario metadata. Encoded playback uses nominal 30 FPS; timestamps record actual capture timing. The operator's scenario tag is not a verified per-frame annotation. Startup and display identify webcam setups as unvalidated. Recording does not train the model.

## Data needed to complete the correction
1. Keep the intended deployment camera position fixed initially. Use the new Record example controls in Webcam mode to collect several separate takes for each scenario.
2. WAIT: gearbox present, no interaction, with both empty scenes and a person standing nearby.
3. PASSTHRU: someone passing the workspace without reaching for the gearbox, including passes close enough to confuse the model.
4. INTERACTION: approach, reach, contact, handling, withdrawal; include stationary contact as well as motion.
5. Record separate sessions for development and final evaluation. Keep people/sessions together in each split. Reserve new takes never used in model fitting or threshold choice.
6. Review timestamps and define onset/end labels. Fine-tune the matching three-class architecture with original and new-angle examples, then measure event recall, false alerts, timing, and regression on the old setup.

For immediate reference behavior, reposition the camera toward the original high downward-facing view and compare recordings. That is a troubleshooting experiment, not a guarantee of correctness. The monitor remains a research display and cannot establish clearance for UR5e motion.
