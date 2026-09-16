"""Recording regression tests use synthetic frames and fake cameras only."""
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

MONITOR = Path(__file__).resolve().parents[1] / 'Intent Monitor'
sys.path.insert(0, str(MONITOR))
from clip_recorder import ClipRecorder
from recording_session import RecordingSession, SessionStore
from camera_devices import NamedCamera

IDS = {'session_id': 'session_01', 'participant_id': 'p01',
       'camera_setup_id': 'view_a', 'tool': 'screwdriver'}


class RecordingTests(unittest.TestCase):
    def test_real_video_timestamps_metadata_and_persistence(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = ClipRecorder(folder)
            identifiers = dict(IDS)
            camera_settings = {'requested_frame_size': [1920, 1080], 'driver_reported_fps': 30}
            recorder.start('INTERACTION', 'synthetic', duration=.31,
                           metadata=identifiers, camera_settings=camera_settings)
            identifiers['participant_id'] = 'changed_after_start'
            camera_settings['requested_frame_size'][0] = 1
            pending = json.loads(recorder.path.with_suffix('.json').read_text())
            self.assertEqual(pending['status'], 'recording')
            base = time.perf_counter()
            for index, stamp in enumerate([0, .03, .06, .1, .2, .3, .32]):
                frame = np.full((48, 64, 3), index * 25, dtype=np.uint8)
                recorder.feed(frame, base + stamp)
            self.assertFalse(recorder.active)
            metadata = json.loads(recorder.path.with_suffix('.json').read_text())
            self.assertEqual(metadata['status'], 'complete')
            self.assertEqual(metadata['frames'], 6)
            self.assertEqual(metadata['participant_id'], 'p01')
            self.assertEqual(metadata['camera_settings']['requested_frame_size'], [1920, 1080])
            self.assertEqual(metadata['split_group_id'], 'session_01')
            self.assertFalse(metadata['annotations_confirmed'])
            self.assertAlmostEqual(metadata['timing']['achieved_fps'], 5 / .3, places=4)
            self.assertGreater(metadata['timing']['gaps_over_threshold'], 0)
            capture = cv2.VideoCapture(str(recorder.path))
            count = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                self.assertEqual(frame.shape, (48, 64, 3))
                count += 1
            capture.release()
            self.assertEqual(count, 6)
            restored = ClipRecorder(folder)
            self.assertEqual(restored.sessions.settings(), IDS)
            restored.start('WAIT', 'synthetic', metadata=IDS)
            self.assertEqual(restored.meta['take_number'], 2)
            restored.stop()
            self.assertEqual(restored.meta['status'], 'incomplete')
            self.assertEqual(restored.meta['frames'], 0)

    def test_stall_before_duration_boundary_is_incomplete(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = ClipRecorder(folder)
            recorder.start('WAIT', 'synthetic', fps=30, duration=10, metadata=IDS)
            base = time.perf_counter()
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            for index in range(181):
                recorder.feed(frame, base + index / 30)
            # Less than the five-second camera timeout, but the final four
            # seconds of the requested footage were never captured.
            recorder.feed(frame, base + 10.1)
            metadata = json.loads(recorder.path.with_suffix('.json').read_text())
            self.assertEqual(metadata['status'], 'incomplete')
            self.assertEqual(metadata['stop_reason'], 'duration reached')
            self.assertEqual(metadata['frames'], 181)
            timing = metadata['timing']
            self.assertAlmostEqual(timing['capture_span_seconds'], 6)
            self.assertAlmostEqual(timing['achieved_fps'], 30)
            self.assertEqual(timing['gaps_over_threshold'], 0)
            self.assertAlmostEqual(timing['observed_end_elapsed_seconds'], 10.1)
            self.assertAlmostEqual(timing['final_observed_gap_seconds'], 4.1)
            self.assertAlmostEqual(timing['saved_coverage_seconds'], 6 + 1 / 30)
            self.assertTrue(any('final saved frame' in warning for warning in metadata['quality_warnings']))
            self.assertTrue(any('does not cover' in warning for warning in metadata['quality_warnings']))
            capture = cv2.VideoCapture(str(recorder.path))
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 181)
            capture.release()

    def test_validation_prevents_unidentified_take(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = ClipRecorder(folder)
            with self.assertRaisesRegex(ValueError, 'Participant'):
                recorder.start('WAIT', 'fake', metadata=dict(IDS, participant_id=''))
            self.assertFalse(recorder.active)
            self.assertFalse(list(Path(folder).glob('*.avi')))
            self.assertFalse(list(Path(folder).glob('*.json')))

    def test_resolution_change_and_nonmonotonic_time_fail_cleanly(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = ClipRecorder(folder)
            stamp = time.perf_counter()
            recorder.start('WAIT', 'fake', metadata=IDS)
            recorder.feed(np.zeros((48, 64, 3), dtype=np.uint8), stamp)
            recorder.feed(np.zeros((60, 80, 3), dtype=np.uint8), stamp + .03)
            self.assertFalse(recorder.active)
            self.assertIn('resolution changed', recorder.meta['stop_reason'])
            self.assertIsNone(recorder.writer)
            recorder.start('PASSTHRU', 'fake', metadata=IDS)
            recorder.feed(np.zeros((48, 64, 3), dtype=np.uint8), stamp)
            recorder.feed(np.zeros((48, 64, 3), dtype=np.uint8), stamp)
            self.assertIn('timestamps did not increase', recorder.meta['stop_reason'])

    def test_camera_only_lifecycle_and_error(self):
        class FakeCamera:
            backend_name = 'fake'
            capture_metadata = {'requested_fps': 30}
            def __init__(self, fail=False):
                self.releases = 0
                self.fail = fail
            def read(self):
                time.sleep(.005)
                if self.fail:
                    raise OSError('simulated unplug')
                return True, np.zeros((48, 64, 3), dtype=np.uint8)
            def release(self):
                self.releases += 1

        with tempfile.TemporaryDirectory() as folder:
            recorder = ClipRecorder(folder)
            camera = FakeCamera()
            session = RecordingSession(camera, recorder)
            recorder.start('WAIT', 'fake', duration=.3, metadata=IDS)
            session.start()
            deadline = time.perf_counter() + 2
            while recorder.active and time.perf_counter() < deadline:
                time.sleep(.01)
            session.stop()
            self.assertFalse(session._thread.is_alive())
            self.assertEqual(camera.releases, 1)
            self.assertEqual(recorder.meta['status'], 'complete')
            self.assertGreater(recorder.meta['frames'], 1)
            bad_camera = FakeCamera(fail=True)
            bad = RecordingSession(bad_camera, recorder)
            recorder.start('WAIT', 'fake', metadata=IDS)
            bad.start()
            bad._thread.join(timeout=2)
            bad.stop()
            self.assertIn('simulated unplug', bad.error)
            self.assertIn('simulated unplug', recorder.meta['stop_reason'])
            self.assertEqual(bad_camera.releases, 1)
            self.assertFalse(recorder.active)

    def test_camera_requests_mode_and_reports_driver_fallback(self):
        class FakeCapture:
            def __init__(self):
                self.settings = {}
                self.released = False
            def isOpened(self):
                return True
            def set(self, key, value):
                self.settings[key] = value
            def get(self, key):
                return {cv2.CAP_PROP_FRAME_WIDTH: 640,
                        cv2.CAP_PROP_FRAME_HEIGHT: 480, cv2.CAP_PROP_FPS: 15}[key]
            def release(self):
                self.released = True

        device = SimpleNamespace(name='test camera', backend=cv2.CAP_DSHOW, path='device_path', index=0)
        capture = FakeCapture()
        with patch('camera_devices.enumerate_cameras', return_value=[device]), \
             patch('camera_devices.cv2.VideoCapture', return_value=capture):
            camera = NamedCamera(device, 1920, 1080, 30)
        self.assertEqual(capture.settings[cv2.CAP_PROP_FRAME_WIDTH], 1920)
        self.assertEqual(camera.capture_metadata['driver_reported_width'], 640)
        self.assertEqual(camera.capture_metadata['driver_reported_fps'], 15)
        camera.release()
        self.assertTrue(capture.released)

    def test_import_monitor_does_not_import_torch_or_open_camera(self):
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "import cv2; "
            "cv2.VideoCapture=lambda *a, **k: (_ for _ in ()).throw(AssertionError('camera opened')); "
            "import intent_monitor; "
            "assert 'torch' not in sys.modules; "
            "assert 'intent_engine_v3_1_windows' not in sys.modules"
        )
        result = subprocess.run([sys.executable, '-B', '-c', code, str(MONITOR)],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
