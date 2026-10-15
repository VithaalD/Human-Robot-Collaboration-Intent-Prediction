# Why original-video predictions differ
Investigation: October 10, 2026. Frozen epoch-28 model; nine validation windows only. Reserved test set was not used. Checkpoint hash remains unchanged. No production preprocessing or model weights changed.

## Finding
A training/inference image-preparation mismatch, especially JPEG compression, explains the observed loss on these validation windows. This is strong experimental evidence for compression sensitivity, not proof that compression explains every continuous/live prediction error.

Training images are 427x240 JPEGs from the partner's extraction. Original AVI frames decode to 1920x1080. Both paths ultimately reach 150x150 RGB with the same pixel scaling and ImageNet normalization, but the training path already includes substantial resizing and lossy JPEG compression. Equal final dimensions do not make the input pixels identical.

## Controlled investigation
1. Fixed the checkpoint, class order, nine windows and eight source indices per window.
2. Decoded original videos sequentially and by random seek: corresponding frames were pixel-identical.
3. Compared each of 72 extracted frames against original positions -2 through +2 using a common resize. The expected zero-offset frame was the closest match for all 72. No evidence of an off-by-one alignment problem in these windows.
4. Ran identical model preprocessing and direct forward passes, bypassing live vote smoothing and display thresholds, to isolate the image path.
5. Compared direct original frames, five intermediate resizing methods, high-quality JPEG variants, and then JPEG encoding using the actual quantization tables measured from each corresponding extracted image.

| Input preparation | Correct / 9 validation windows |
|---|---:|
| Partner's extracted JPEGs | 8/9 |
| Original video frames directly | 6/9 |
| Resize originals to 427x240 first (linear, area, cubic, Pillow bilinear or Lanczos) | 6/9 for every method |
| Intermediate resize plus high-quality JPEG at quality 95 | 6/9 for every tested method |
| Intermediate resize plus measured training-JPEG quantization and 4:2:0 subsampling | 8/9 for every tested method |

With linear or Pillow-bilinear resizing plus matched compression, all nine predicted labels matched the extracted-frame path, including its existing WAIT-to-PASSTHRU error. The direct raw path added two errors: take 5 INTERACTION became PASSTHRU; take 45 PASSTHRU became WAIT. Neither eight-fps capture nor the eight-frame window explains that controlled difference because both were held fixed.

Ten distinct JPEG quantization-table configurations occurred in the 72 tested images. The diagnostic used each image's measured settings, which are not automatically available for future live frames. It therefore establishes a mechanism, not a ready-to-deploy universal JPEG-quality fix. Encoded results were still not pixel-identical to the partner images. The exact extraction command/encoder remains unknown.

The earlier source-correspondence audit checked whether extracted pictures corresponded to the right source frames within a visual tolerance; it did not establish identical model-input tensors or prediction equivalence. Step 7's 10/10 score remains a selected extracted-frame result and cannot be extrapolated to raw/live input. The continuous take-2 replay showing only WAIT may also involve transition windows and smoothing; this experiment did not fully diagnose that separate behavior.

## Recommended fix, not yet applied
1. Keep original recordings, reviewed label ranges and source identifiers. Request the partner's exact extraction command/settings for provenance; no need to discard the recordings or redo all labels.
2. Create a separate dataset version directly from the original AVI files at the existing reviewed source indices/timestamps. Avoid the partner JPEG compression path. Apply one shared decode/color/resize/normalization implementation in training, recorded-video inference and live inference. Prefer lossless intermediate storage or decoded-frame caching.
3. Add a parity check: the same original frame/window routed through training and inference should yield equal preprocessing tensors within a defined tolerance and matching predictions.
4. Retrain the temporal/classification layers (and evaluate any additional changes on validation only) on that consistent dataset, preserving source-group isolation. Mere deployment preprocessing changes cannot be assumed to repair the current trained model.
5. Validate original-video windows and continuous clips, including action boundaries, edge hands and walking; separately assess smoothing and confidence behavior. Then test with the real lab camera at its training angle.
6. Reserve fresh reviewed recordings for a later unbiased final evaluation. The earlier test has already been examined; distinguish follow-up diagnostics from a new untouched test result. Review October 8 segments before formal scoring.

A compatibility experiment recreating a documented fixed extraction pipeline could help the current checkpoint, but per-frame matched compression in this investigation is not a general live solution. Do not introduce arbitrary JPEG-quality tuning based on these nine outcomes.

## Original frames versus extracted validation windows
An original frame is a decoded image from the AVI. An extracted frame represents the corresponding moment saved as a smaller JPEG. A validation window is a selected sequence of eight such frames inside one reviewed label range, here the center eight frames. It is neither a new recording nor the entire source clip. A continuous live stream supplies sliding windows, including mixed-action transitions that clean single-label evaluation windows may omit.

## What steps 1-8 did with the partner's work
The partner had already provided frame extraction and annotations/split organization. We reused that material, rather than starting again from unprocessed video.

1. Import/verify: checked the delivered archive, 59 source trials and 3,607 JPEG frames, hashes, timestamps and correspondence to original footage.
2. Label review: reviewed semantic ranges under the user's edge/interior/task rules rather than assuming the folder label applied to every frame. Selected 131 ranges totaling 2,613 labeled frames; left 994 uncertain frames unassigned. Labels are AI-reviewed, not independent human ground truth.
3. Training export: connected ranges to source timestamps and sampled at eight fps. Retained 93 ranges with at least eight real frames, totaling 2,413 frames; excluded 38 too-short ranges. Exported separate folders so a training window does not cross a selected label boundary.
4. Split/loaders: initially checked the inherited split, then superseded it with the requested approximately 80/10/10 source-group split: 47/6/6 recordings and 74/9/10 segments. Every segment from an original recording stays in one subset. Verified timing, shapes and source isolation.
5. Smoke test: two short epochs on six training and three validation segments checked finite computation, gradients and checkpoint save/reload. This was an engineering test, not the final prototype or accuracy claim.
6. Actual training: initialized ImageNet ResNet101, froze its image backbone and trained the projection, three-layer LSTM and classifier. Used class-balanced sampling and random eight-frame training windows. Cached frozen image features for efficiency. Ran 43 epochs with early stopping and selected epoch 28 by validation loss. Saved the complete model and preprocessing contract.
7. Reserved evaluation: evaluated that fixed model on ten selected extracted-frame windows from six reserved recordings: 10/10 correct. No test-based tuning. Small same-session/same-participant coverage limits the result.
8. Application integration: connected the checkpoint to the monitor, loaded its normalization and frame settings, added Start Small Prototype.cmd, checked prediction parity and timing, exercised real recorded-video playback, and ran 70 regression tests. This exposed the raw/extracted mismatch. The RealSense camera was absent, so lab live validation remains pending.

The additional reviewed segments were labeling/training selections within supplied source trials. They did not create more independent recordings, and they do not imply the partner had failed to prepare or splice the data. The missing consistency check was whether original/live inputs resembled the supplied training JPEGs closely enough for this particular model.
