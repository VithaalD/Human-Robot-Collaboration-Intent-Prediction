# Step 7: reserved-test evaluation
Completed October 10, 2026.

Evaluated the frozen epoch-28 checkpoint selected during step 6 on the predefined test set: 10 reviewed segments from six September 29 recordings. Each prediction used the center eight consecutive frames at eight fps with the checkpoint's RGB, resize and ImageNet normalization settings. No training, checkpoint selection, threshold adjustment or test-based tuning was performed. The checkpoint SHA-256 was verified before and after evaluation and remained unchanged.

| True class | Predicted INTERACTION | Predicted PASSTHRU | Predicted WAIT | Correct |
|---|---:|---:|---:|---:|
| INTERACTION | 2 | 0 | 0 | 2/2 |
| PASSTHRU | 0 | 2 | 0 | 2/2 |
| WAIT | 0 | 0 | 6 | 6/6 |

Accuracy: 10/10 = 100%. Macro F1 and balanced accuracy: 1.000. Precision, recall and F1 were 1.000 for each class. Cross-entropy loss: 0.079666. An always-WAIT baseline would score 6/10 = 60%. All evaluated segments in all six test recordings were correct; this is not a whole-video continuous prediction score.

## Interpretation
This prototype passed this small reserved-data test. It does not establish 100% real-world accuracy. The test contains only two INTERACTION and two PASSTHRU segments, and all footage comes from the same session and known participants as training. Multiple segments from one recording are correlated. The labels were AI-reviewed, not independently human-confirmed. Boundaries, continuous action transitions, different lighting, October 8 footage and live camera behavior have not been evaluated by this test.

The step-6 validation score was 8/9 (88.9%); the test score being higher on this tiny sample is not evidence that test footage is generally easier or that the model improved after training. The model was unchanged.

## When hands-on testing is available
The trained checkpoint exists now and has now been tested offline. The next step (step 8) is connecting it to the recording/live-camera interface and verifying exact preprocessing, eight-frame/eight-fps sampling, class order and camera timing. After that setup, the user can walk through WAIT, PASSTHRU and INTERACTION and see predictions. No additional training or full 60/60/40 collection is required before this exploratory demonstration. Live integration and latency are not yet verified; inference must use the saved checkpoint contract rather than the old smoke-test defaults.

Keep test results fixed for this model. If these results influence later development, use fresh held-out recordings for an unbiased future evaluation. Review the October 8 recordings before using them for formal labeled scoring.

## Artifacts
- test_report.json: per-segment predictions, probabilities, exact input windows, confusion matrix, metrics and checkpoint hash.
- evaluation_script.py: reference snapshot of tools/evaluate_prototype_step7.py.
- best_checkpoint.pth: unchanged trained model.

Run from the project root with the intent-gpu environment: python tools/evaluate_prototype_step7.py. If the report already exists, the script returns the saved result rather than rescoring.
