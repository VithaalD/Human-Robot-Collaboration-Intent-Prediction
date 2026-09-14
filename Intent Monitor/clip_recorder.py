"""Save camera frames with capture timestamps for subsequent annotation."""
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import cv2


class ClipRecorder:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.lock = threading.Lock()
        self.writer = None
        self.active = False
        self.message = 'Record a scene to collect examples for model correction.'

    def start(self, scenario, camera, fps=30.0, duration=10.0):
        with self.lock:
            if self.active:
                raise RuntimeError('A recording is already active.')
            if scenario not in ('INTERACTION', 'PASSTHRU', 'WAIT'):
                raise ValueError('Choose a valid recording scenario.')
            self.folder.mkdir(parents=True, exist_ok=True)
            stem = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + scenario + '_' + uuid4().hex[:8]
            self.path = self.folder / (stem + '.avi')
            self.fps = float(fps) if 1 <= fps <= 120 else 30.0
            self.duration = duration
            self.requested = time.monotonic()
            self.first_frame_time = None
            self.last_frame_time = self.requested
            self.timestamps = []
            self.meta = dict(scenario=scenario, camera=camera, created=datetime.now().isoformat(),
                             annotations_confirmed=False, frame_timestamps_seconds=self.timestamps,
                             nominal_video_fps=self.fps,
                             note='Scenario is the operator-selected activity for this take, not verified per-frame labels. Use frame timestamps for timing; AVI playback uses nominal FPS.')
            self.writer = None
            self.active = True
            self.message = 'Waiting for the first camera frame…'

    def feed(self, frame, captured_at):
        with self.lock:
            if not self.active:
                return
            try:
                if self.first_frame_time is None:
                    self.first_frame_time = captured_at
                    h, w = frame.shape[:2]
                    self.meta['frame_size'] = [w, h]
                    self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*'MJPG'), self.fps, (w, h))
                    if not self.writer.isOpened():
                        raise RuntimeError('Cannot create video recording.')
                self.last_frame_time = captured_at
                elapsed = captured_at - self.first_frame_time
                if elapsed >= self.duration:
                    self._finish('duration reached')
                    return
                h, w = frame.shape[:2]
                if [w, h] != self.meta['frame_size']:
                    raise RuntimeError('Camera resolution changed during recording.')
                self.writer.write(frame)
                self.timestamps.append(elapsed)
                self.message = f'Recording {self.meta["scenario"]}: {elapsed:.1f} / {self.duration:.0f} sec'
            except Exception as exc:
                self._finish('recording error: ' + str(exc))

    def _finish(self, reason):
        self.active = False
        if self.writer:
            self.writer.release()
            self.writer = None
        self.meta['frames'] = len(self.timestamps)
        self.meta['stop_reason'] = reason
        try:
            self.path.with_suffix('.json').write_text(json.dumps(self.meta, indent=2), encoding='utf-8')
            self.message = ('Saved ' if self.timestamps and 'error' not in reason else 'Incomplete recording: ') + self.path.name
        except OSError as exc:
            self.message = 'Could not save recording metadata: ' + str(exc)

    def stop(self, reason='stopped by operator'):
        with self.lock:
            if self.active:
                self._finish(reason)

    def status(self):
        with self.lock:
            if self.active and time.monotonic()-self.last_frame_time > 5:
                self._finish('recording error: no camera frames received')
            return self.active, self.message
