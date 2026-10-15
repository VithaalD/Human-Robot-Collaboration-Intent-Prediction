"""Prepare existing timestamped recordings at 8 FPS without modifying originals.

Without --review: export all takes and a review CSV, never training labels.
With --review: export only confirmed, time-bounded segments and annotations.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frame_sampling import FrameSampler, recording_timestamps

LABELS = ['INTERACTION', 'PASSTHRU', 'WAIT']
FIELDS = ['video', 'start_seconds', 'end_seconds', 'label', 'subset', 'confirmed', 'participant_id', 'session_id', 'notes']


def prepare(recordings, output, fps=8.0, review=None):
    recordings, output = Path(recordings).resolve(), Path(output).resolve()
    FrameSampler(fps)  # validate rate before creating output
    if output.exists():
        raise ValueError('Output already exists; choose a new folder to preserve prior work.')
    if review:
        with open(review, newline='', encoding='utf-8-sig') as stream:
            requests = [r for r in csv.DictReader(stream) if r.get('confirmed', '').strip().lower() == 'true']
        if not requests:
            raise ValueError('No confirmed segments in review CSV')
    else:
        requests = [{'video': p.relative_to(recordings).as_posix()} for p in sorted(recordings.rglob('*.avi'))]
    if not requests:
        raise ValueError('No AVI recordings found')
    jobs, groups = [], {}
    # Validate every requested segment before creating any output.
    for number, request in enumerate(requests):
        video = (recordings / request['video']).resolve()
        if not video.is_relative_to(recordings):
            raise ValueError('Video must be inside the recordings folder')
        cap = cv2.VideoCapture(str(video))
        try:
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            stamps, meta = recording_timestamps(video, count, cap.get(cv2.CAP_PROP_FPS))
        finally:
            cap.release()
        if not meta or not stamps:
            raise ValueError(f'Original capture metadata required: {video}')
        if str(meta.get('stop_reason', '')).startswith('recording error'):
            raise ValueError(f'Capture error requires separate repair/review: {video}')
        start = float(request.get('start_seconds', 0))
        end = float(request.get('end_seconds', stamps[-1] + 1e-6))
        if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end <= stamps[-1] + .126:
            raise ValueError(f'Invalid segment bounds: {video}')
        label = request.get('label', meta['scenario'])
        if label not in LABELS:
            raise ValueError(f'Invalid label: {label}')
        subset = request.get('subset', '')
        group = meta.get('split_group_id') or meta['session_id']
        if review:
            if subset not in ('training', 'validation', 'testing'):
                raise ValueError('Each confirmed segment needs training, validation or testing subset')
            if group in groups and groups[group] != subset:
                raise ValueError('Keep each recording session entirely in one split')
            groups[group] = subset
        sampler, indices = FrameSampler(fps), []
        for i, stamp in enumerate(stamps):
            if start <= stamp < end:
                accepted = sampler.accept(stamp)
                if sampler.discontinuity:
                    raise ValueError(f'Timing gap in segment: {video}; exclude gap using separate reviewed segments')
                if accepted:
                    indices.append(i)
        if len(indices) < 2:
            raise ValueError(f'Segment has fewer than two frames: {video}')
        name = video.stem + (f'_segment_{number:04d}' if review else '')
        jobs.append((video, stamps, meta, label, subset, group, name, indices, start, end, request))
    output.mkdir(parents=True)
    manifest = {'sample_fps': fps, 'selection': 'real_frames_fixed_grid_10_percent_jitter_tolerance', 'clips': {}}
    annotations = {'labels': LABELS, 'sample_fps': fps, 'annotations_confirmed': bool(review), 'database': {}}
    review_rows = []
    for video, stamps, meta, label, subset, group, name, indices, start, end, request in jobs:
        folder = output / 'frames' / label / name
        folder.mkdir(parents=True)
        wanted, saved = set(indices), 0
        cap = cv2.VideoCapture(str(video))
        try:
            for i in range(indices[-1] + 1):
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f'Video ended before metadata: {video}')
                if i in wanted:
                    saved += 1
                    if not cv2.imwrite(str(folder / f'image_{saved:05d}.jpg'), frame):
                        raise OSError('Could not write extracted frame')
        finally:
            cap.release()
        (folder / 'n_frames').write_text(str(saved))
        clip = {'source_recording_id': meta['source_recording_id'], 'split_group_id': group,
                'session_id': meta['session_id'], 'participant_id': request.get('participant_id', meta['participant_id']),
                'source_video': str(video), 'source_frame_indices': indices,
                'source_timestamps_seconds': [stamps[i] for i in indices],
                'sample_fps': fps, 'frames': saved, 'original_status': meta['status']}
        manifest['clips'][name] = clip
        if review:
            annotations['database'][name] = {'subset': subset, 'annotations': {'label': label},
                'source_recording_id': meta['source_recording_id'], 'split_group_id': group}
        review_rows.append(dict(video=video.relative_to(recordings).as_posix(), start_seconds=start,
            end_seconds=end, label=label, subset=subset, confirmed='true' if review else 'false',
            participant_id=clip['participant_id'], session_id=meta['session_id'],
            notes='Review action boundaries and participant ID; times are ORIGINAL capture seconds.'))
    (output / 'frames' / 'sampling.json').write_text(json.dumps(manifest, indent=2))
    if review:
        (output / 'annotations.json').write_text(json.dumps(annotations, indent=2))
    with (output / 'review.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(review_rows)
    print(f'Prepared {len(jobs)} clips, {sum(c["frames"] for c in manifest["clips"].values())} real frames at {fps:g} FPS: {output}')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recordings', type=Path, default=Path(__file__).resolve().parents[1] / 'Camera Recordings')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample_fps', type=float, default=8.0)
    parser.add_argument('--review', type=Path)
    args = parser.parse_args()
    prepare(args.recordings, args.output, args.sample_fps, args.review)
