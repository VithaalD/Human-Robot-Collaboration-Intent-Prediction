"""Fixed, versioned object features shared by future training and inference."""
import numpy as np

CLASS_ORDER = ('gearbox', 'human_hand', 'screwdriver')
FIELDS = ('present', 'confidence', 'center_x', 'center_y', 'width', 'height')
FEATURE_COUNT = len(CLASS_ORDER) * len(FIELDS)
SCHEMA_VERSION = 'yolo_object_features_v1'


def feature_schema():
    return {
        'version': SCHEMA_VERSION,
        'class_order': list(CLASS_ORDER),
        'fields_per_class': list(FIELDS),
        'feature_count': FEATURE_COUNT,
        'dtype': 'float32',
        'coordinates': 'xywh normalized to original image width/height',
        'missing_detection': 'six zeros',
        'instance_selection': 'highest confidence per class; first on ties',
    }


def extract_yolo_features(result):
    """Convert one Ultralytics Results object to 18 float32 values.

    Map by class NAME so differing numeric IDs cannot silently swap features.
    Input boxes must already be filtered by the caller's confidence threshold.
    Presence means detected, not proof that an undetected object is absent.
    """
    names = result.names
    names = dict(enumerate(names)) if isinstance(names, (list, tuple)) else names
    if len(names) != len(CLASS_ORDER) or set(names.values()) != set(CLASS_ORDER):
        raise ValueError(f'Expected exactly these detector classes: {CLASS_ORDER}')
    output = np.zeros((len(CLASS_ORDER), len(FIELDS)), dtype=np.float32)
    if result.boxes is None or len(result.boxes) == 0:
        return output.reshape(-1)
    boxes = result.boxes.xywhn.detach().cpu().numpy()
    confidences = result.boxes.conf.detach().cpu().numpy()
    ids = result.boxes.cls.detach().cpu().numpy()
    if boxes.shape != (len(confidences), 4) or ids.shape != confidences.shape:
        raise ValueError('Inconsistent detection array shapes')
    if not all(np.isfinite(a).all() for a in (boxes, confidences, ids)):
        raise ValueError('Detection contains non-finite values')
    if (boxes < 0).any() or (boxes > 1).any():
        raise ValueError('Normalized box coordinates must be within [0, 1]')
    if (confidences < 0).any() or (confidences > 1).any():
        raise ValueError('Confidence must be within [0, 1]')
    for box, confidence, class_id in zip(boxes, confidences, ids):
        if class_id != int(class_id) or int(class_id) not in names:
            raise ValueError('Unknown or non-integer class ID')
        slot = CLASS_ORDER.index(names[int(class_id)])
        if output[slot, 0] == 0 or confidence > output[slot, 1]:
            output[slot] = [1.0, confidence, *box]
    return output.reshape(-1)
