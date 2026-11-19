import numpy as np
import pytest
import torch
from ultralytics.engine.results import Results
from yolo_features import CLASS_ORDER, FEATURE_COUNT, extract_yolo_features


def result(rows, names=None):
    return Results(np.zeros((100, 200, 3), dtype=np.uint8), path='test',
                   names=names or dict(enumerate(CLASS_ORDER)),
                   boxes=torch.tensor(rows, dtype=torch.float32).reshape(-1, 6))


def test_empty_detections_have_fixed_zero_vector():
    features = extract_yolo_features(result([]))
    assert features.shape == (FEATURE_COUNT,)
    assert features.dtype == np.float32
    assert not features.any()


def test_pixel_normalization_missing_class_and_strongest_instance():
    features = extract_yolo_features(result([
        [20, 10, 60, 30, .4, 1],
        [100, 20, 180, 60, .9, 1],
        [0, 0, 200, 100, .8, 0],
    ])).reshape(3, 6)
    np.testing.assert_allclose(features[0], [1, .8, .5, .5, 1, 1])
    np.testing.assert_allclose(features[1], [1, .9, .7, .4, .4, .4])
    np.testing.assert_array_equal(features[2], np.zeros(6))


def test_names_determine_slots_not_numeric_ids():
    names = {0: 'screwdriver', 1: 'gearbox', 2: 'human_hand'}
    features = extract_yolo_features(result([[0, 0, 100, 50, .8, 0]], names))
    assert not features[:12].any()
    np.testing.assert_allclose(features[12:], [1, .8, .25, .25, .5, .5])


def test_wrong_taxonomy_is_rejected_even_without_detections():
    with pytest.raises(ValueError, match='Expected exactly'):
        extract_yolo_features(result([], {0: 'person'}))


def test_nonfinite_detection_is_rejected():
    with pytest.raises(ValueError, match='non-finite'):
        extract_yolo_features(result([[0, 0, 100, 50, float('nan'), 0]]))
