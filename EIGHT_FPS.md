# Using the existing approximately 8 FPS recordings

Keep recording as before. The camera request, original AVI files, original metadata,
and 10-second button are unchanged. Five-to-six-second manually stopped takes can
be prepared: `incomplete` from stopping early is not treated as corrupted footage.

## Playback and inference

Open an original AVI in Intent Monitor. It automatically finds the matching JSON
in the scenario metadata folder (or beside an older flat recording) and plays
using capture timestamps, not the misleading 30 FPS AVI header. Keep AVI/JSON pairs
together. Ordinary media players still follow the original AVI header.

Model input is sampled at 8 FPS by default. New checkpoints carry `sample_fps`;
inference automatically uses their saved rate. An explicit conflicting CLI
`--sample_fps` is rejected. Legacy checkpoints remain loadable with a warning:
their historical sampling rate is unknown; they are not proven accurate at 8 FPS.
The sequence remains 16 real frames (about 1.875 seconds first-to-last at 8 FPS).
Eight-frame sequences are also supported, but require matching training settings.
Long gaps and video replay boundaries clear the sequence. Duplicate inference on
an unchanged frame window is skipped. No interpolated motion is generated.

## Prepare recordings

From the project folder, using the GPU Python environment:

```powershell
python tools/prepare_8fps.py --output "Prepared 8 FPS"
```

This produces full-resolution JPG frames, `frames/sampling.json` with original
frame indices/timestamps, and `review.csv`. Existing outputs are never overwritten.
Original recordings are unchanged. Near-8 FPS timing jitter is tolerated without
accidentally discarding every other frame. Frame selection uses the same sampler
as live inference. A long gap within a requested segment is rejected, not hidden.

The initial export is a review dataset, NOT verified training labels. In review.csv,
use original capture seconds for start/end (end exclusive), check the label and
participant ID, assign a subset (training/validation/testing), and mark reviewed
rows `true`. Duplicate a row to label separate action segments within one take.
The INTERACTION folder does not imply that its opening/closing WAIT frames are
interaction. Keep all segments of each session in one split. For evaluation on
new people, also keep participants separate. The current single session cannot
provide an independent session-held-out validation set by itself.

Export reviewed segments to a new output directory:

```powershell
python tools/prepare_8fps.py --review "Prepared 8 FPS/review.csv" --output "Reviewed 8 FPS"
```

Then from intentpredictionattempt4-main, after separate training and validation
sessions have been reviewed:

```powershell
python main.py --dataset ucf101 --video_path "../Reviewed 8 FPS/frames" --annotation_path "../Reviewed 8 FPS/annotations.json" --n_classes 3 --sample_fps 8 --sample_duration 16 --sample_size 150 --batch_size 1 --num_workers 0 --use_cuda --gpu 0
```

Both trainer copies default to 8 FPS and verify the preparation manifest and
reviewed annotation contract. They save the rate in checkpoints and check it on
resume. Existing unverified frame datasets require explicit `--legacy_timing`;
this records an unknown rate rather than incorrectly asserting they are 8 FPS.
This update does not train or validate a new model. Keep original metadata backed up.
