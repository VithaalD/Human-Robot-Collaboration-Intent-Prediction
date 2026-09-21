"""Export full-frame DLC coordinates with video/model provenance."""
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from .dlc import DLCPredictor
from .features import LANDMARKS
from .models import file_hash


def pose_identity(project, model, snapshot):
    return dict(project_sha256=file_hash(project), model_sha256=file_hash(model),
                snapshot_sha256=file_hash(snapshot), adapter='dlc-3.0.1-bu-v1')


def extract(video, project, model, snapshot, output, device='cpu'):
    output = Path(output)
    sidecar = output.with_suffix('.pose.json')
    if output.exists() or sidecar.exists() or output.suffix.lower() != '.csv':
        raise ValueError('Choose a new .csv output path')
    predictor = DLCPredictor(project, model, snapshot, device)
    cap = cv2.VideoCapture(str(video))
    poses = []
    try:
        if not cap.isOpened():
            raise ValueError('Cannot open source video')
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            poses.append(predictor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    finally:
        cap.release()
    if len(poses) < 2:
        raise ValueError('Video has fewer than two decodable frames')
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = pd.MultiIndex.from_product([['pose_intent'], LANDMARKS, ['x', 'y', 'likelihood']],
                                         names=['scorer', 'bodyparts', 'coords'])
    pd.DataFrame(np.asarray(poses).reshape(len(poses), -1), columns=columns).to_csv(output)
    sidecar.write_text(json.dumps(dict(pose_identity=pose_identity(project, model, snapshot),
                                      video_sha256=file_hash(video), pose_sha256=file_hash(output)), indent=2), encoding='utf-8')
    return dict(frames=len(poses), output=str(output))
