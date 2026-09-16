"""Camera-only capture and persistent collection identifiers; no ML dependency."""
import json
import os
import re
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def validate_identifiers(values):
    result = {}
    for key, label in [('session_id', 'Session'), ('participant_id', 'Participant'),
                       ('camera_setup_id', 'Camera setup')]:
        value = str(values.get(key, '')).strip()
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', value):
            raise ValueError(f'{label} ID is required: use letters, numbers, hyphens or underscores (up to 64).')
        result[key] = value
    result['tool'] = str(values.get('tool', '')).strip()
    if not result['tool'] or len(result['tool']) > 100:
        raise ValueError('Enter the tool name (up to 100 characters).')
    return result


class SessionStore:
    """Persist settings and allocate take numbers within a session."""
    def __init__(self, folder):
        self.path = Path(folder) / 'recording_session.json'
        self.lock = threading.Lock()
        self.data = {'settings': {'session_id': datetime.now().strftime('session_%Y%m%d'),
                                 'participant_id': '', 'camera_setup_id': '',
                                 'tool': 'screwdriver'}, 'next_takes': {}}
        if self.path.exists():
            loaded = json.loads(self.path.read_text(encoding='utf-8'))
            if not isinstance(loaded.get('settings'), dict) or not isinstance(loaded.get('next_takes'), dict):
                raise ValueError('Recording session settings are invalid; preserve the file and use a new recording folder.')
            self.data = loaded

    def settings(self):
        return dict(self.data['settings'])

    def next_take(self, session_id):
        return int(self.data['next_takes'].get(session_id, 1))

    def reserve_take(self, values):
        values = validate_identifiers(values)
        with self.lock:
            number = self.next_take(values['session_id'])
            updated = {'settings': values,
                       'next_takes': dict(self.data['next_takes'], **{values['session_id']: number + 1})}
            atomic_json(self.path, updated)
            self.data = updated
        return dict(values, take_id=f'{values["session_id"]}_take_{number:04d}', take_number=number,
                    split_group_id=values['session_id'])


class CameraPreview:
    def __init__(self, camera, recorder=None):
        self.camera = camera
        self.recorder = recorder
        self.backend_name = camera.backend_name
        self.frame = None
        self.last_frame_time = 0.0
        self.capture_metadata = getattr(camera, 'capture_metadata', {})

    def read(self):
        ok, frame = self.camera.read()
        if ok and frame is not None:
            self.frame = frame
            self.last_frame_time = time.perf_counter()
            self.capture_metadata['delivered_frame_size'] = [int(frame.shape[1]), int(frame.shape[0])]
            if self.recorder is not None:
                self.recorder.feed(frame, self.last_frame_time)
        return ok, frame

    def release(self):
        self.camera.release()


class RecordingSession:
    """Own the camera on one worker thread; never load inference weights."""
    record_only = True

    def __init__(self, camera, recorder):
        self.preview = CameraPreview(camera, recorder)
        self.camera_backend = camera.backend_name
        self.recorder = recorder
        self.error = ''
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name='camera-record-only')
        self._thread.start()

    def _run(self):
        last_good = time.perf_counter()
        try:
            while not self._stop.is_set():
                ok, frame = self.preview.read()
                if ok and frame is not None:
                    last_good = time.perf_counter()
                else:
                    if time.perf_counter() - last_good > 5:
                        raise RuntimeError('Camera stopped delivering frames. Stop and reconnect it.')
                    self._stop.wait(0.02)
        except Exception as exc:
            self.error = str(exc)
            self.recorder.stop('recording error: ' + self.error)
        finally:
            try:
                self.preview.release()
            except Exception as exc:
                self.error = self.error or ('Could not close camera: ' + str(exc))
                self.recorder.stop('recording error: ' + self.error)

    def stop(self):
        self._stop.set()
        self.recorder.stop()
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise RuntimeError('Camera driver is still stopping. Close the application before reconnecting the camera.')
        else:
            self.preview.release()
