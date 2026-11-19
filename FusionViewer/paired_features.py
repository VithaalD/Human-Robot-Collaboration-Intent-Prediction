"""Synchronized CNN/YOLO features. This module does not predict intent."""
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from PIL import Image

from frame_sampling import FrameSampler
from yolo_features import FEATURE_COUNT, extract_yolo_features, feature_schema


@dataclass(frozen=True)
class PairedFrame:
    source_id: str
    frame_id: int
    timestamp: float
    cnn: np.ndarray
    objects: np.ndarray

    @property
    def combined(self):
        return np.concatenate((self.cnn, self.objects)).astype(np.float32)


class PairedFeatureExtractor:
    """Load once, then extract both features from the same original BGR frame.

    Sequential single-consumer API. YOLO receives full-resolution BGR; CNN
    receives the baseline RGB transform. Neither model is trained here.
    """
    def __init__(self, engine, detector, confidence=0.25, image_size=640):
        if not 0 <= confidence <= 1 or image_size <= 0:
            raise ValueError('Invalid detector settings')
        self.engine = engine
        self.detector = detector
        self.confidence = confidence
        self.image_size = image_size
        self.cnn = engine._model.resnet.eval()
        self.device = engine.device

    @classmethod
    def from_checkpoints(cls, intent_weights, yolo_weights, **kwargs):
        monitor = str(Path(__file__).resolve().parent / 'Intent Monitor')
        if monitor not in sys.path:
            sys.path.insert(0, monitor)
        from intent_monitor import config
        from intent_engine_v3_1_windows import IntentEngine
        from ultralytics import YOLO
        for path in (intent_weights, yolo_weights):
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        engine = IntentEngine(config(intent_weights, sample_duration=8))
        if engine.cfg.sample_fps != 8:
            raise ValueError('This integration requires the 8 FPS baseline')
        return cls(engine, YOLO(str(yolo_weights)), **kwargs)

    def __call__(self, frame):
        cnn, objects = self.extract_batch([frame])
        return cnn[0], objects[0]

    def extract_batch(self, frames):
        """Batch the same paired computation for offline cache generation."""
        if not frames:
            raise ValueError('Expected at least one frame')
        originals, tensors = [], []
        for frame in frames:
            if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                    or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0):
                raise ValueError('Expected a nonempty uint8 BGR image')
            original = np.ascontiguousarray(frame).copy()
            originals.append(original)
            rgb = Image.fromarray(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            tensors.append(self.engine._transform(rgb))
        with torch.inference_mode():
            # Ultralytics can change global cuDNN precision settings on setup.
            # Keep the CNN branch consistent before and after detector warmup.
            previous_tf32 = torch.backends.cuda.matmul.allow_tf32
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
                with torch.backends.cudnn.flags(allow_tf32=False):
                    cnn = self.cnn(torch.stack(tensors).to(self.device)).detach().float().cpu().numpy().copy()
            finally:
                torch.backends.cuda.matmul.allow_tf32 = previous_tf32
            results = self.detector.predict(originals, conf=self.confidence, imgsz=self.image_size,
                                            device=str(self.device), verbose=False)
        if len(results) != len(originals):
            raise ValueError('Detector result count does not match input frames')
        objects = np.stack([extract_yolo_features(result) for result in results])
        if cnn.shape != (len(frames), 300) or not np.isfinite(cnn).all():
            raise ValueError('Invalid CNN features')
        return cnn, objects

    def metadata(self):
        cfg = self.engine.cfg
        return {'sample_fps': 8, 'sequence_length': 8, 'cnn_features': 300,
                'combined_features': 318, 'cnn_precision': 'float32_tf32_disabled', 'object_schema': feature_schema(),
                'cnn_preprocessing': {'size': cfg.sample_size, 'norm_value': cfg.norm_value,
                                     'mean': cfg.mean, 'std': cfg.std, 'color_order': 'RGB'},
                'detector': {'confidence': self.confidence, 'imgsz': self.image_size,
                             'color_order': 'BGR', 'input': 'original resolution'}}


class PairedFeatureStream:
    """Sample source time at 8 FPS and buffer eight synchronized feature rows.

    Call push with every captured/decoded frame and its source timestamp.
    For live capture use a monotonic capture timestamp, not inference completion
    time. Reset/source change/time discontinuity clears the previous window.
    """
    def __init__(self, extractor):
        self.extractor = extractor
        self.sampler = FrameSampler(8.0)
        self.frames = deque(maxlen=8)
        self.source_id = None

    def reset(self):
        self.sampler.reset()
        self.frames.clear()
        self.source_id = None

    def push(self, frame, timestamp, frame_id, source_id):
        if not isinstance(source_id, str) or not source_id:
            raise ValueError('A nonempty source_id is required')
        if self.source_id != source_id:
            self.reset()
            self.source_id = source_id
        accepted = self.sampler.accept(float(timestamp))
        if self.sampler.discontinuity:
            self.frames.clear()
        if not accepted:
            return None
        try:
            cnn, objects = self.extractor(frame)
            cnn = np.asarray(cnn, dtype=np.float32).copy()
            objects = np.asarray(objects, dtype=np.float32).copy()
            if cnn.shape != (300,) or objects.shape != (FEATURE_COUNT,):
                raise ValueError('Expected 300 CNN and 18 object features')
            if not np.isfinite(cnn).all() or not np.isfinite(objects).all():
                raise ValueError('Non-finite paired features')
        except Exception:
            self.reset()
            raise
        record = PairedFrame(source_id, frame_id, float(timestamp), cnn, objects)
        self.frames.append(record)
        return record

    def window(self):
        if len(self.frames) != 8:
            return None
        return {'features': np.stack([f.combined for f in self.frames]),
                'timestamps': np.array([f.timestamp for f in self.frames], dtype=np.float64),
                'frame_ids': [f.frame_id for f in self.frames],
                'source_id': self.source_id}
