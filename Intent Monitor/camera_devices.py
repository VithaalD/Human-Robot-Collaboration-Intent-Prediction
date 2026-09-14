"""Windows camera names and backend-matched capture for Intent Monitor."""
import sys
from pathlib import Path
from collections import Counter
import cv2

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
    def __init__(self, selected):
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
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_FPS, 30)

    def read(self):
        return self._cap.read()

    def release(self):
        self._cap.release()
