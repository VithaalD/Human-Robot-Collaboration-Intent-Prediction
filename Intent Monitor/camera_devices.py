"""Windows camera names and backend-matched capture for Intent Monitor."""
import sys
from pathlib import Path
from collections import Counter
import cv2
import math

RESOLUTIONS = {'1920 x 1080': (1920, 1080), '1280 x 720': (1280, 720), '640 x 480': (640, 480)}

sys.path.insert(0, str(Path(__file__).resolve().parent / '_vendor'))
from cv2_enumerate_cameras import enumerate_cameras


def discover_cameras():
    # Prefer one backend so the same physical camera is not listed twice.
    devices = enumerate_cameras(cv2.CAP_DSHOW)
    return devices or enumerate_cameras(cv2.CAP_MSMF)


def identity(device):
    return (device.backend, device.path or (device.name, device.index))


def display_names(devices):
    counts = Counter(d.name for d in devices)
    seen = Counter()
    labels = []
    for d in devices:
        seen[d.name] += 1
        labels.append(d.name if counts[d.name] == 1 else f'{d.name} (device {seen[d.name]})')
    return labels


class NamedCamera:
    def __init__(self, selected, width=1920, height=1080, fps=30):
        # Resolve a fresh index by stable device path after plug/unplug events.
        matches = [d for d in enumerate_cameras(selected.backend) if identity(d) == identity(selected)]
        if len(matches) != 1:
            raise RuntimeError(f'{selected.name} is no longer available. Click Refresh and select a camera.')
        device = matches[0]
        self.backend_name = device.name
        self._cap = cv2.VideoCapture(device.index, device.backend)
        if not self._cap.isOpened():
            self._cap.release()
            raise RuntimeError(f'Cannot open {device.name}. Close other camera apps and check Windows camera permissions.')
        try:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._cap.set(cv2.CAP_PROP_FPS, fps)
            def readback(prop):
                value = float(self._cap.get(prop))
                return value if math.isfinite(value) and value > 0 else None
            self.capture_metadata = {
                'device_name': device.name, 'backend': int(device.backend),
                'requested_frame_size': [width, height], 'requested_fps': fps,
                'driver_reported_width': readback(cv2.CAP_PROP_FRAME_WIDTH),
                'driver_reported_height': readback(cv2.CAP_PROP_FRAME_HEIGHT),
                'driver_reported_fps': readback(cv2.CAP_PROP_FPS),
                'note': 'Driver readback is not measured capture rate; check per-take timing.'}
        except Exception:
            self._cap.release()
            raise

    def read(self):
        return self._cap.read()

    def release(self):
        self._cap.release()
