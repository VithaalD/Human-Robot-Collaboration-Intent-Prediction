import numpy as np
import pytest
from paired_features import PairedFeatureStream


def extractor(frame):
    return np.full(300, frame, dtype=np.float32), np.full(18, frame + 100, dtype=np.float32)


def test_sampling_pairs_and_window_order():
    calls = []
    def observed(frame):
        calls.append(frame)
        return extractor(frame)
    stream = PairedFeatureStream(observed)
    for i in range(32):
        stream.push(i, i / 32, i, 'clip-a')
    window = stream.window()
    assert calls == list(range(0, 32, 4))
    assert window['features'].shape == (8, 318)
    assert window['features'].dtype == np.float32
    np.testing.assert_allclose(window['timestamps'], np.arange(8) / 8)
    assert window['frame_ids'] == calls
    np.testing.assert_array_equal(window['features'][:, 0], calls)
    np.testing.assert_array_equal(window['features'][:, 300], np.array(calls) + 100)
    assert stream.push(999, 31 / 32, 99, 'clip-a') is None


@pytest.mark.parametrize('timestamp,source', [(5.0, 'a'), (0.0, 'a'), (1.0, 'b')])
def test_no_windows_across_gap_rewind_or_source_change(timestamp, source):
    stream = PairedFeatureStream(extractor)
    for i in range(8):
        stream.push(i, i / 8, i, 'a')
    assert stream.window() is not None
    stream.push(10, timestamp, 10, source)
    assert stream.window() is None
    assert len(stream.frames) == 1


def test_failure_clears_partial_sequence():
    stream = PairedFeatureStream(extractor)
    stream.push(1, 0., 0, 'a')
    stream.extractor = lambda frame: (np.zeros(299), np.zeros(18))
    with pytest.raises(ValueError, match='300 CNN'):
        stream.push(2, .125, 1, 'a')
    assert len(stream.frames) == 0
    assert stream.source_id is None
