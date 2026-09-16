"""Save original camera frames and reviewed-later collection metadata."""
import copy
import math
import threading
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
from recording_session import SessionStore, atomic_json


class ClipRecorder:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.sessions = SessionStore(folder)
        self.lock = threading.Lock()
        self.writer = None
        self.active = False
        self.completed_count = 0
        self.message = 'Enter collection IDs, then record a scene. Labels need review.'

    def start(self, scenario, camera, fps=30.0, duration=10.0, metadata=None, camera_settings=None):
        with self.lock:
            if self.active:
                raise RuntimeError('A recording is already active.')
            if scenario not in ('INTERACTION', 'PASSTHRU', 'WAIT'):
                raise ValueError('Choose a valid recording scenario.')
            if not math.isfinite(float(fps)) or not 1 <= float(fps) <= 120:
                raise ValueError('Recording FPS must be between 1 and 120.')
            if not math.isfinite(float(duration)) or float(duration) <= 0:
                raise ValueError('Recording duration must be positive.')
            identifiers = self.sessions.reserve_take(metadata or {})
            recording_id = uuid4().hex
            stem = identifiers['take_id'] + '_' + scenario + '_' + recording_id[:8]
            self.path = self.folder / (stem + '.avi')
            self.fps = float(fps)
            self.duration = float(duration)
            self.requested = time.perf_counter()
            self.first_frame_time = None
            self.last_frame_time = self.requested
            self.timestamps = []
            self.observed_end_elapsed = None
            self.meta = dict(identifiers, schema_version=2, source_recording_id=recording_id,
                             scenario=scenario, camera=camera, created=datetime.now().astimezone().isoformat(),
                             status='recording', video_file=self.path.name,
                             annotations_confirmed=False, frame_timestamps_seconds=self.timestamps,
                             nominal_video_fps=self.fps, requested_duration_seconds=self.duration,
                             camera_settings=copy.deepcopy(camera_settings or {}),
                             note='Scenario is a take-level intention, not per-frame ground truth. Use capture timestamps for timing; AVI playback uses nominal FPS. Keep all derivatives of this take in one dataset split.')
            atomic_json(self.path.with_suffix('.json'), self.meta)
            self.writer = None
            self.active = True
            self.message = 'Waiting for the first camera frame…'

    def feed(self, frame, captured_at):
        with self.lock:
            if not self.active:
                return
            try:
                if not math.isfinite(captured_at):
                    raise ValueError('Invalid capture timestamp.')
                if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype.name != 'uint8':
                    raise ValueError('Camera did not deliver a BGR uint8 image.')
                if self.first_frame_time is None:
                    self.first_frame_time = captured_at
                    h, w = frame.shape[:2]
                    self.meta['frame_size'] = [w, h]
                    self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*'MJPG'), self.fps, (w, h))
                    if not self.writer.isOpened():
                        raise RuntimeError('Cannot create video recording.')
                elapsed = captured_at - self.first_frame_time
                if self.timestamps and elapsed <= self.timestamps[-1]:
                    raise ValueError('Capture timestamps did not increase.')
                self.last_frame_time = captured_at
                self.observed_end_elapsed = elapsed
                if elapsed >= self.duration:
                    self._finish('duration reached')
                    return
                h, w = frame.shape[:2]
                if [w, h] != self.meta['frame_size']:
                    raise RuntimeError('Camera resolution changed during recording.')
                self.writer.write(frame)
                self.timestamps.append(elapsed)
                self.message = f'Recording {self.meta["take_id"]} / {self.meta["scenario"]}: {elapsed:.1f} / {self.duration:.0f} sec'
            except Exception as exc:
                self._finish('recording error: ' + str(exc))

    def _finish(self, reason):
        self.active = False
        if self.writer is not None:
            writer, self.writer = self.writer, None
            try:
                writer.release()
            except Exception as exc:
                reason = 'recording error: could not finish video: ' + str(exc)
        intervals = [b - a for a, b in zip(self.timestamps, self.timestamps[1:])]
        span = self.timestamps[-1] if self.timestamps else 0.0
        actual_fps = (len(self.timestamps) - 1) / span if span > 0 else None
        gap_threshold = 1.5 / self.fps
        # The first frame beyond the requested duration is not written, but its
        # arrival still reveals a possible stall after the last saved frame.
        final_gap = (max(0.0, self.observed_end_elapsed - span)
                     if self.timestamps and self.observed_end_elapsed is not None else None)
        coverage = min(self.duration, span + 1.0 / self.fps) if self.timestamps else 0.0
        coverage_tolerance = max(2.0 / self.fps, self.duration * 0.05)
        short_coverage = coverage < self.duration - coverage_tolerance
        timing = {'captured_frame_count': len(self.timestamps), 'capture_span_seconds': span,
                  'achieved_fps': actual_fps, 'nominal_playback_seconds': len(self.timestamps) / self.fps,
                  'max_frame_gap_seconds': max(intervals) if intervals else None,
                  'gap_threshold_seconds': gap_threshold,
                  'gaps_over_threshold': sum(gap > gap_threshold for gap in intervals),
                  'observed_end_elapsed_seconds': self.observed_end_elapsed,
                  'final_observed_gap_seconds': final_gap,
                  'saved_coverage_seconds': coverage,
                  'coverage_tolerance_seconds': coverage_tolerance}
        warnings = []
        if len(self.timestamps) < 2:
            warnings.append('Too few frames; repeat this take.')
        if actual_fps is not None and not 0.9 * self.fps <= actual_fps <= 1.1 * self.fps:
            warnings.append('Capture rate differs from nominal playback rate by over 10%; inspect timing before training.')
        if timing['gaps_over_threshold']:
            warnings.append('Uneven frame timing; gaps are observations, not confirmed dropped-frame counts.')
        if final_gap is not None and final_gap > gap_threshold:
            warnings.append('Gap after the final saved frame; the boundary observation was not added to the video.')
        if short_coverage:
            warnings.append('Saved footage does not cover the requested duration; repeat this take.')
        complete = reason == 'duration reached' and len(self.timestamps) >= 2 and not short_coverage
        self.meta.update(frames=len(self.timestamps), stop_reason=reason,
                         status='complete' if complete else 'incomplete', timing=timing, quality_warnings=warnings)
        try:
            atomic_json(self.path.with_suffix('.json'), self.meta)
            prefix = 'Saved ' if complete else 'Incomplete take: '
            self.message = prefix + self.path.name
            if actual_fps is not None:
                self.message += f' • {actual_fps:.1f} captured FPS'
            if warnings:
                self.message += ' • Review timing'
        except OSError as exc:
            self.message = 'Could not save recording metadata: ' + str(exc)
        self.completed_count += 1

    def stop(self, reason='stopped by operator'):
        with self.lock:
            if self.active:
                self._finish(reason)

    def status(self):
        with self.lock:
            if self.active and time.perf_counter() - self.last_frame_time > 5:
                self._finish('recording error: no camera frames received')
            return self.active, self.message
