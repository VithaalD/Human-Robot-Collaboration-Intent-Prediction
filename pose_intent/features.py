"""One causal preprocessing contract for training, replay and live inference."""
from dataclasses import asdict, dataclass
import numpy as np

LANDMARKS = ('left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
             'left_wrist', 'right_wrist', 'screwdriver_tip', 'screwdriver_handle')
LABELS = ('INTERACTION', 'PASSTHRU', 'WAIT')
FEATURE_DIM = 46


@dataclass(frozen=True)
class InputConfig:
    version: int = 1
    landmarks: tuple = LANDMARKS
    labels: tuple = LABELS
    frames: int = 16
    sample_hz: float = 10.0
    max_gap: float = 0.15
    likelihood: float = 0.6
    min_valid: float = 0.5
    max_age: float = 0.5
    workspace_center: tuple | None = None

    def __post_init__(self):
        if self.version != 1 or tuple(self.landmarks) != LANDMARKS or tuple(self.labels) != LABELS:
            raise ValueError('Unsupported input schema, landmark order or class order')
        values = [self.sample_hz, self.max_gap, self.likelihood, self.min_valid, self.max_age]
        if not np.isfinite(values).all() or self.frames < 2 or self.sample_hz <= 0 or self.max_gap <= 0 or self.max_age <= 0:
            raise ValueError('Invalid timing settings')
        if not 0 < self.likelihood <= 1 or not 0 < self.min_valid <= 1:
            raise ValueError('Likelihood and minimum validity must be in (0, 1]')
        if self.workspace_center is not None and (len(self.workspace_center) != 2 or
                not np.isfinite(self.workspace_center).all() or not all(0 <= v <= 1 for v in self.workspace_center)):
            raise ValueError('Workspace center must be normalized x,y in [0,1]')

    def dictionary(self):
        return asdict(self)


def validate_times(times):
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Need at least two finite, strictly increasing timestamps')
    return times


def sample_indices(times, end, cfg):
    """Select only observations at or before each target; never look ahead."""
    times = validate_times(times)
    targets = end - np.arange(cfg.frames - 1, -1, -1) / cfg.sample_hz
    indices = np.searchsorted(times, targets + 1e-9, side='right') - 1
    if np.any(indices < 0) or np.any(targets - times[indices] > cfg.max_gap) or np.any(np.diff(indices) <= 0):
        raise ValueError('Insufficient distinct, timely observations for this window')
    return indices


def motion_features(points, times, size, cfg):
    """Returns features, per-feature masks, likelihoods, and pose coverage.

    No interpolation/symmetric smoothing. Derivatives use actual elapsed time.
    Distances are image-plane distances, not estimates of physical contact.
    """
    times = validate_times(times)
    p = np.asarray(points, dtype=np.float64)
    if p.shape != (len(times), len(LANDMARKS), 3):
        raise ValueError('Expected (time, 8, x/y/likelihood) pose array')
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError('Invalid image dimensions')
    confidence = np.nan_to_num(p[:, :, 2], nan=0, posinf=0, neginf=0).clip(0, 1)
    valid = np.isfinite(p).all(-1) & (confidence >= cfg.likelihood)
    xy = np.nan_to_num(p[:, :, :2], nan=0, posinf=0, neginf=0)
    center = (xy[:, 0] + xy[:, 1]) / 2
    scale = np.linalg.norm(xy[:, 0] - xy[:, 1], axis=-1)
    anchor = valid[:, 0] & valid[:, 1] & (scale > 1)
    scale = np.maximum(scale, 1)
    rel = (xy - center[:, None]) / scale[:, None, None]
    rmask = np.repeat((valid & anchor[:, None])[:, :, None], 2, axis=2)
    parts, masks = [], []

    def add(value, mask):
        parts.append(value.reshape(len(times), -1))
        masks.append(np.broadcast_to(mask, value.shape).reshape(len(times), -1))

    def derivative(value, mask):
        out = np.zeros_like(value)
        vm = np.zeros_like(mask, dtype=bool)
        dt = np.diff(times)
        out[1:] = np.diff(value, axis=0) / dt.reshape((-1,) + (1,) * (value.ndim - 1))
        timely = (dt <= cfg.max_gap).reshape((-1,) + (1,) * (value.ndim - 1))
        vm[1:] = mask[1:] & mask[:-1] & timely
        return out, vm

    add(rel, rmask)
    add(*derivative(rel, rmask))
    dist = np.linalg.norm(xy[:, 4:6] - xy[:, 7:8], axis=-1) / scale[:, None]
    dm = valid[:, 4:6] & valid[:, 7:8] & anchor[:, None]
    add(dist, dm)
    add(*derivative(dist, dm))
    angles = np.zeros((len(times), 2))
    am = np.zeros_like(angles, dtype=bool)
    for side in range(2):
        a, b = xy[:, side] - xy[:, side + 2], xy[:, side + 4] - xy[:, side + 2]
        denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
        angles[:, side] = np.arccos(np.clip((a * b).sum(-1) / np.maximum(denom, 1e-6), -1, 1)) / np.pi
        am[:, side] = valid[:, side] & valid[:, side + 2] & valid[:, side + 4] & (denom > 1)
    add(angles, am)
    cn = center / np.array([width, height])
    cm = np.repeat(anchor[:, None], 2, axis=1)
    add(cn, cm)
    add(*derivative(cn, cm))
    tool = xy[:, 6] - xy[:, 7]
    length = np.linalg.norm(tool, axis=-1)
    add(tool / np.maximum(length[:, None], 1), np.repeat((valid[:, 6] & valid[:, 7] & (length > 1))[:, None], 2, axis=1))
    workspace = np.zeros_like(dist)
    wm = np.zeros_like(dm)
    if cfg.workspace_center is not None:
        target = np.array(cfg.workspace_center) * [width, height]
        workspace = np.linalg.norm(xy[:, 4:6] - target, axis=-1) / scale[:, None]
        wm = valid[:, 4:6] & anchor[:, None]
    add(workspace, wm)
    mask = np.concatenate(masks, axis=1)
    features = np.concatenate(parts, axis=1)
    mask &= np.isfinite(features)
    features = np.where(mask, features, 0).clip(-100, 100).astype('float32')
    assert features.shape[1] == FEATURE_DIM
    # Torso, wrists and tool determine whether the window is usable.
    coverage = float((valid[:, [0, 1, 4, 5, 6, 7]] & anchor[:, None]).mean())
    return features, mask.astype('float32'), confidence.astype('float32'), coverage
