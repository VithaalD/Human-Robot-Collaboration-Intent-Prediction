"""Reviewed segment manifest -> synchronized, grouped training windows."""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from .features import InputConfig, LABELS, validate_times, sample_indices, motion_features
from .dlc import read_pose_table
from .models import file_hash, load_video_encoder, image_tensor, RGB_CONFIG


def read_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1 or not manifest.get('takes'):
        raise ValueError('Expected schema_version 1 and nonempty takes')
    groups = {name: {} for name in ('session_id', 'take_id', 'video')}
    if manifest.get('group_participants', True):
        groups['participant_id'] = {}
    for take in manifest['takes']:
        if take.get('split') not in ('train', 'val', 'test') or take.get('annotations_confirmed') is not True:
            raise ValueError('Every take needs a split and explicitly confirmed segment annotations')
        for name in ('take_id', 'session_id', 'participant_id', 'camera_setup_id', 'video', 'metadata', 'pose'):
            if not isinstance(take.get(name), str) or not take[name].strip():
                raise ValueError(f'Missing {name}')
        for name in ('video', 'metadata', 'pose'):
            take[name] = str((path.parent / take[name]).resolve())
        for key, owners in groups.items():
            identity = take[key].casefold()
            if identity in owners and (key in ('take_id', 'video') or owners[identity] != take['split']):
                raise ValueError(f'Duplicate take or split leakage: {key}={take[key]}')
            owners[identity] = take['split']
        previous_end = -1
        if not take.get('segments'):
            raise ValueError('Every take needs reviewed start/end segments')
        for seg in take['segments']:
            start, end = seg['start'], seg['end']
            if not np.isfinite([start, end]).all() or start < 0 or end <= start or start < previous_end or seg['label'] not in LABELS:
                raise ValueError('Segments must be ordered, nonoverlapping, finite, and use known labels')
            previous_end = end
    return manifest


def read_take(take):
    metadata = json.loads(Path(take['metadata']).read_text(encoding='utf-8'))
    for key in ('take_id', 'session_id', 'participant_id', 'camera_setup_id'):
        if metadata.get(key) != take[key]:
            raise ValueError(f'Recording metadata disagrees with manifest: {key}')
    if metadata.get('video_file') and metadata['video_file'] != Path(take['video']).name:
        raise ValueError('Recording metadata names a different video')
    times = validate_times(metadata['frame_timestamps_seconds'])
    points = read_pose_table(take['pose'])
    if len(points) != len(times):
        raise ValueError('Pose row count does not match recording timestamps')
    if times[0] < 0:
        raise ValueError('Recording timestamps must be seconds since capture start')
    return times, points


def iter_windows(times, points, size, cfg, stride):
    if not np.isfinite(stride) or stride <= 0:
        raise ValueError('Window stride must be positive')
    first = times[0] + (cfg.frames - 1) / cfg.sample_hz
    for end in np.arange(first, times[-1] + 1e-8, stride):
        end = times[np.searchsorted(times, end + 1e-9, side='right') - 1]
        try:
            ids = sample_indices(times, end, cfg)
            f, m, c, quality = motion_features(points[ids], times[ids], size, cfg)
        except ValueError:
            yield float(end), None
            continue
        yield float(end), (ids, f, m, c, quality)


def prepare(manifest_path, output, cfg, stride=0.2, rgb_weights=None, device='cpu'):
    manifest = read_manifest(manifest_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Use an empty output directory to avoid mixing datasets')
    encoder = load_video_encoder(rgb_weights, cfg, device) if rgb_weights else None
    records, counts = [], {'timing_rejected': 0, 'quality_rejected': 0, 'unlabeled': 0}
    # Detect copied videos under different names, including across sessions.
    seen_sources = set()
    pose_model = None
    for take in manifest['takes']:
        source_hash = file_hash(take['video'])
        if source_hash in seen_sources:
            raise ValueError('Duplicate source video content in manifest')
        seen_sources.add(source_hash)
        provenance = json.loads(Path(take['pose']).with_suffix('.pose.json').read_text(encoding='utf-8'))
        if provenance.get('pose_sha256') != file_hash(take['pose']):
            raise ValueError('Pose table changed since extraction')
        if provenance.get('video_sha256') != source_hash:
            raise ValueError('Pose file was extracted from a different video')
        if not provenance.get('pose_identity'):
            raise ValueError('Missing pose model provenance; use the extract command')
        if pose_model is not None and pose_model != provenance['pose_identity']:
            raise ValueError('All takes must use the same pose model and extraction settings')
        pose_model = provenance['pose_identity']
        times, points = read_take(take)
        cap = cv2.VideoCapture(take['video'])
        try:
            if not cap.isOpened():
                raise ValueError(f'Cannot open video: {take["video"]}')
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            # Decode sequentially, retaining only RGB tensors needed for windows.
            windows = []
            for end, result in iter_windows(times, points, size, cfg, stride):
                if result is None:
                    counts['timing_rejected'] += 1
                    continue
                ids, f, m, c, quality = result
                segment = next((s for s in take['segments'] if s['start'] <= end < s['end']), None)
                if segment is None:
                    counts['unlabeled'] += 1
                    continue
                if quality < cfg.min_valid:
                    counts['quality_rejected'] += 1
                    continue
                # Labels describe the endpoint; preceding context may cross an onset.
                windows.append((end, ids, f, m, c, quality, segment['label']))
            needed = {int(i) for w in windows for i in w[1]} if encoder else set()
            frames = {}
            frame_count = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_count in needed:
                    frames[frame_count] = image_tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_count += 1
            if frame_count != len(times):
                raise ValueError('Decoded video length differs from pose/timestamps; refusing alignment')
            if any(s['end'] > times[-1] + 1 / float(cfg.sample_hz) for s in take['segments']):
                raise ValueError('Reviewed segment extends past recording')
            for end, ids, f, m, c, quality, label in windows:
                arrays = dict(features=f, mask=m, likelihood=c, frame_ids=ids,
                              timestamps=times[ids], label=np.int64(LABELS.index(label)))
                if encoder:
                    with torch.inference_mode():
                        embedding, logits = encoder(torch.stack([frames[int(i)] for i in ids])[None].to(device))
                    arrays.update(rgb=embedding[0].cpu().numpy(), rgb_logits=logits[0].cpu().numpy())
                filename = f'{len(records):07d}.npz'
                np.savez_compressed(output / filename, **arrays)
                records.append(dict(file=filename, take_id=take['take_id'], session_id=take['session_id'],
                                    participant_id=take['participant_id'], split=take['split'],
                                    camera_setup_id=take['camera_setup_id'], end=end, quality=quality, label=label))
        finally:
            cap.release()
    if not records:
        raise ValueError('No usable labeled windows; inspect timestamps and tracking quality')
    index = dict(format='pose-windows-v1', input_config=cfg.dictionary(), rgb_config=RGB_CONFIG,
                 pose_identity=pose_model, rgb_sha256=file_hash(rgb_weights) if rgb_weights else None,
                 manifest_sha256=file_hash(manifest_path), takes=manifest['takes'], records=records,
                 rejection_counts=counts, stride=stride, group_participants=manifest.get('group_participants', True))
    (output / 'index.json').write_text(json.dumps(index, indent=2), encoding='utf-8')
    return dict(windows=len(records), **counts)


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, folder, split):
        self.folder = Path(folder)
        self.index = json.loads((self.folder / 'index.json').read_text(encoding='utf-8'))
        if self.index.get('format') != 'pose-windows-v1':
            raise ValueError('Unsupported prepared dataset')
        self.cfg = InputConfig(**self.index['input_config'])
        self.records = [r for r in self.index['records'] if r['split'] == split]
        if not self.records:
            raise ValueError(f'No usable {split} windows')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        with np.load(self.folder / self.records[i]['file'], allow_pickle=False) as data:
            return {k: torch.from_numpy(np.array(data[k])) for k in ('features', 'mask', 'likelihood', 'label', 'rgb', 'rgb_logits') if k in data}
