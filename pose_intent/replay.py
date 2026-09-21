"""Recorded-video inference using the same causal features as live operation."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from .features import validate_times
from .data import iter_windows
from .dlc import read_pose_table
from .models import file_hash
from .runtime import FramePacket, PoseIntentRuntime


def replay(checkpoint, video, metadata, pose, output, rgb_weights=None, device='cpu'):
    output = Path(output)
    if output.exists():
        raise ValueError('Choose a new prediction log path')
    runtime = PoseIntentRuntime(checkpoint, rgb_weights, device)
    source = json.loads(Path(metadata).read_text(encoding='utf-8'))
    times = validate_times(source['frame_timestamps_seconds'])
    points = read_pose_table(pose)
    provenance = json.loads(Path(pose).with_suffix('.pose.json').read_text(encoding='utf-8'))
    if provenance.get('pose_sha256') != file_hash(pose):
        raise ValueError('Pose table changed since extraction')
    if provenance['video_sha256'] != file_hash(video) or provenance['pose_identity'] != runtime.saved['pose_identity']:
        raise ValueError('Pose source/model differs from recording or checkpoint')
    if len(points) != len(times):
        raise ValueError('Pose and timestamp counts differ')
    cap = cv2.VideoCapture(video)
    frames = []
    try:
        if not cap.isOpened():
            raise ValueError('Cannot open video')
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = np.array(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).resize((150, 150), Image.Resampling.BILINEAR))
            frames.append(rgb)
    finally:
        cap.release()
    if len(frames) != len(times):
        raise ValueError('Decoded video and timestamp counts differ')
    rows = []
    for end, result in iter_windows(times, points, (width, height), runtime.cfg, 1 / runtime.cfg.sample_hz):
        if result is None:
            rows.append(dict(timestamp=end, status='UNAVAILABLE', reason='Timing gap', label=None, probabilities=None))
            continue
        ids = result[0]
        observations = [(FramePacket(int(i), float(times[i]), frames[i], (width, height)), points[i]) for i in ids]
        row = runtime.predict(observations, end)
        rows.append(dict(row, timestamp=end))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row) + '\n')
    return dict(windows=len(rows), available=sum(r['status'] == 'OK' for r in rows), output=str(output))
