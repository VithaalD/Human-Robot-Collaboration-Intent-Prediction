"""Bounded pose worker and synchronized live classifier; no robot actuation."""
from collections import deque
from dataclasses import dataclass
import json
import queue
import threading
import time
import cv2
import numpy as np
import torch
from PIL import Image
from .features import LABELS, sample_indices, motion_features
from .models import load_intent, load_video_encoder, image_tensor, file_hash
from .dlc import DLCPredictor
from .extraction import pose_identity


@dataclass(frozen=True)
class FramePacket:
    frame_id: int
    timestamp: float
    rgb: np.ndarray
    original_size: tuple | None = None


class PoseWorker:
    """Latest-frame queue bounds latency; pose and image stay in one packet."""
    def __init__(self, predictor, capacity=128):
        self.predictor = predictor
        self.pending = queue.Queue(maxsize=1)
        self.completed = deque(maxlen=capacity)
        self.lock = threading.Lock()
        self.closed = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True, name='DLC-pose')
        self.thread.start()

    def submit(self, packet):
        if self.closed.is_set():
            return
        try:
            self.pending.put_nowait(packet)
        except queue.Full:
            try:
                self.pending.get_nowait()
            except queue.Empty:
                pass
            try:
                self.pending.put_nowait(packet)
            except queue.Full:
                pass

    def snapshot(self):
        with self.lock:
            return list(self.completed), self.error

    def _run(self):
        while not self.closed.is_set():
            try:
                packet = self.pending.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                pose = self.predictor(packet.rgb)
                size = (packet.rgb.shape[1], packet.rgb.shape[0])
                thumbnail = np.array(Image.fromarray(packet.rgb).resize((150, 150), Image.Resampling.BILINEAR))
                packet = FramePacket(packet.frame_id, packet.timestamp, thumbnail, size)
                with self.lock:
                    self.completed.append((packet, pose))
                    self.error = None
            except Exception as exc:
                with self.lock:
                    self.error = str(exc)
                    self.completed.clear()

    def close(self):
        self.closed.set()
        self.thread.join(timeout=5)


class PoseIntentRuntime:
    def __init__(self, checkpoint, rgb_weights=None, device='cpu'):
        self.model, self.cfg, self.saved = load_intent(checkpoint, device)
        self.device = device
        self.video = None
        if self.saved['modality'] == 'fusion':
            if not rgb_weights or file_hash(rgb_weights) != self.saved['rgb_sha256']:
                raise ValueError('Fusion requires the exact RGB checkpoint used to prepare training features')
            self.video = load_video_encoder(rgb_weights, self.cfg, device)

    def predict(self, observations, now):
        unknown = dict(status='UNAVAILABLE', label=None, probabilities=None)
        if len(observations) < self.cfg.frames:
            return dict(unknown, reason='Collecting movement history')
        packets, poses = zip(*observations)
        times = np.array([p.timestamp for p in packets])
        age = now - times[-1]
        if age < 0 or age > self.cfg.max_age:
            return dict(unknown, reason='Pose data is stale', age_seconds=float(age))
        try:
            ids = sample_indices(times, times[-1], self.cfg)
        except ValueError as exc:
            return dict(unknown, reason=str(exc))
        selected = [packets[int(i)] for i in ids]
        shape = selected[0].rgb.shape
        size = selected[0].original_size or (shape[1], shape[0])
        if any((p.original_size or (p.rgb.shape[1], p.rgb.shape[0])) != size for p in selected):
            return dict(unknown, reason='Camera dimensions changed')
        f, m, c, quality = motion_features(np.array(poses)[ids], times[ids], size, self.cfg)
        if quality < self.cfg.min_valid:
            return dict(unknown, reason='Insufficient reliable landmarks', quality=quality)
        with torch.inference_mode():
            rgb = None
            if self.video is not None:
                clip = torch.stack([image_tensor(p.rgb) for p in selected])[None].to(self.device)
                rgb, _ = self.video(clip)
            tensors = [torch.from_numpy(a)[None].to(self.device) for a in (f, m, c)]
            probabilities = self.model(*tensors, rgb).softmax(-1)[0].cpu().numpy()
        if not np.isfinite(probabilities).all():
            return dict(unknown, reason='Invalid model output')
        return dict(status='OK', label=LABELS[int(probabilities.argmax())], probabilities=probabilities.tolist(),
                    frame_id=selected[-1].frame_id, timestamp=selected[-1].timestamp,
                    window_start=selected[0].timestamp, quality=quality, age_seconds=float(age))


def live(checkpoint, project, model_config, snapshot, rgb_weights=None, camera=0, device='cpu', output=None):
    runtime = PoseIntentRuntime(checkpoint, rgb_weights, device)
    if runtime.saved['pose_identity'] != pose_identity(project, model_config, snapshot):
        raise ValueError('Live DLC configuration/snapshot differs from training extraction')
    predictor = DLCPredictor(project, model_config, snapshot, device)
    # Keep a fixed amount of original-resolution history, independent of capture rate.
    worker = PoseWorker(predictor, capacity=max(32, int(runtime.cfg.frames / runtime.cfg.sample_hz * 120) + 16))
    cap = cv2.VideoCapture(camera)
    stop = threading.Event()
    display_lock = threading.Lock()
    display = {'frame': None, 'error': None}
    capture_thread = None
    log = None
    try:
        if not cap.isOpened():
            raise ValueError('Cannot open camera')
        if output:
            log = open(output, 'x', encoding='utf-8')

        def capture():
            frame_id = 0
            while not stop.is_set():
                ok, bgr = cap.read()
                timestamp = time.perf_counter()
                if not ok:
                    with display_lock:
                        display['error'] = 'Camera capture failed'
                    stop.set()
                    return
                with display_lock:
                    display['frame'] = bgr.copy()
                worker.submit(FramePacket(frame_id, timestamp, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
                frame_id += 1

        capture_thread = threading.Thread(target=capture, daemon=True, name='pose-intent-capture')
        capture_thread.start()
        result = dict(status='UNAVAILABLE', reason='Collecting movement history', label=None, probabilities=None)
        last_id = -1
        while not stop.is_set():
            with display_lock:
                bgr = None if display['frame'] is None else display['frame'].copy()
            observations, error = worker.snapshot()
            if error:
                result = dict(status='UNAVAILABLE', reason=error, label=None, probabilities=None)
            elif observations and observations[-1][0].frame_id != last_id:
                last_id = observations[-1][0].frame_id
                result = runtime.predict(observations, time.perf_counter())
            # Include RGB/intent computation time in freshness, not only DLC latency.
            if observations and time.perf_counter() - observations[-1][0].timestamp > runtime.cfg.max_age:
                result = dict(status='UNAVAILABLE', reason='Prediction is stale', label=None, probabilities=None)
            if bgr is not None:
                text = result.get('label') or result.get('reason', 'Unavailable')
                cv2.putText(bgr, text[:100], (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
                cv2.imshow('Pose Intent - review only - Q to close', bgr)
            if log:
                log.write(json.dumps(dict(result, reported_at=time.perf_counter())) + '\n')
                log.flush()
            if cv2.waitKey(30) & 0xff == ord('q'):
                break
        if display['error']:
            raise RuntimeError(display['error'])
    finally:
        stop.set()
        cap.release()
        if capture_thread:
            capture_thread.join(timeout=5)
        worker.close()
        cv2.destroyAllWindows()
        if log:
            log.close()
