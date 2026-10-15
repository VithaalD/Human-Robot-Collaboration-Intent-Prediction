import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import cv2
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from frame_sampling import FrameSampler, recording_timestamps, validate_training_timing
from test_temporal import engine, LABELS, TREES, training_tree


def load_prepare():
    spec = importlib.util.spec_from_file_location('prepare8', ROOT / 'tools/prepare_8fps.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def recording(tmp_path, name='test', session='s1'):
    folder = tmp_path / 'INTERACTION'
    (folder / 'videos').mkdir(parents=True, exist_ok=True)
    (folder / 'metadata').mkdir(exist_ok=True)
    path = folder / 'videos' / (name + '.avi')
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 30, (32, 24))
    for i in range(24): writer.write(np.full((24,32,3), 40+i*3, dtype=np.uint8))
    writer.release()
    metadata = dict(frame_timestamps_seconds=[i*.1251 for i in range(24)], scenario='INTERACTION',
                    source_recording_id=name, session_id=session, split_group_id=session,
                    participant_id='p1', status='incomplete', stop_reason='stopped by operator')
    (folder / 'metadata' / (name + '.json')).write_text(json.dumps(metadata))
    return path


def test_jitter_does_not_halve_eight_fps():
    sampler = FrameSampler(8)
    assert sum(sampler.accept(i*.1251) for i in range(80)) == 80
    sampler = FrameSampler(8)
    assert sum(sampler.accept(i/30) for i in range(300)) == 80


def test_gaps_and_loops_reset():
    s = FrameSampler(8)
    for t in [0,.125,.25]: assert s.accept(t)
    assert s.accept(2) and s.discontinuity
    assert s.accept(0) and s.discontinuity


def test_capture_timestamps_override_header_and_invalid_is_error(tmp_path):
    p = recording(tmp_path)
    stamps, meta = recording_timestamps(p,24,30)
    assert stamps[-1] == pytest.approx(2.8773)
    with pytest.raises(ValueError, match='count mismatch'): recording_timestamps(p,23,30)


def test_video_playback_uses_capture_intervals(engine, tmp_path):
    camera = engine._VideoFileCamera(str(recording(tmp_path)))
    with patch.object(engine.time, 'sleep') as sleep, patch.object(engine.time, 'monotonic', return_value=100):
        for i in range(24):
            assert camera.read()[0]
            assert camera.sample_timestamp == pytest.approx(i*.1251)
        assert all(c.args[0] == pytest.approx(.1251) for c in sleep.call_args_list)
        assert camera.read()[0]
        assert camera.sample_timestamp > 10
    camera.release()


def test_checkpoint_rate_auto_and_mismatch(engine):
    cfg = engine.EngineConfig()
    engine._validate_checkpoint_input({'input_config': {'sample_fps': 8}}, cfg, LABELS)
    assert cfg.sample_fps == 8
    with pytest.raises(ValueError, match='sample_fps'):
        engine._validate_checkpoint_input({'input_config': {'sample_fps': 8}}, engine.EngineConfig(sample_fps=30), LABELS)
    with pytest.raises(ValueError, match='finite'):
        engine._validate_checkpoint_input({}, engine.EngineConfig(sample_fps=float('nan')), LABELS)


def test_live_sampling_and_gap_flush(engine):
    import collections, threading
    e = engine.IntentEngine.__new__(engine.IntentEngine)
    e._sampler = FrameSampler(8);e._frame_lock=threading.Lock();e._frame_generation=0;e._sequence_epoch=0
    e._raw_frames=collections.deque(maxlen=16);e._pred_history=collections.deque()
    frame=np.full((24,32,3),128,dtype=np.uint8)
    for i in range(60): e.push_frame(frame,i/30)
    assert e._frame_generation == 16
    e._pred_history.append(1)
    e.push_frame(frame,10)
    assert len(e._raw_frames)==1 and not e._pred_history


def test_prepare_preserves_short_takes_and_requires_review(tmp_path):
    p=recording(tmp_path/'raw')
    prep=load_prepare();out=tmp_path/'prepared'
    result=prep.prepare(tmp_path/'raw',out)
    assert result['clips']['test']['frames']==24
    assert not (out/'annotations.json').exists()
    rows=list(csv.DictReader((out/'review.csv').open()))
    assert rows[0]['confirmed']=='false'
    with pytest.raises(ValueError,match='No confirmed'): prep.prepare(tmp_path/'raw',tmp_path/'bad',review=out/'review.csv')
    with pytest.raises(ValueError,match='already exists'): prep.prepare(tmp_path/'raw',out)
    rows[0].update(confirmed='true',subset='training',start_seconds='.5',end_seconds='2')
    with (out/'review.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    reviewed=tmp_path/'reviewed';prep.prepare(tmp_path/'raw',reviewed,review=out/'review.csv')
    data=json.loads((reviewed/'annotations.json').read_text())
    assert data['annotations_confirmed']
    with pytest.raises(ValueError,match='separate training and validation'):
        validate_training_timing(SimpleNamespace(sample_fps=8,video_path=reviewed/'frames',annotation_path=reviewed/'annotations.json'))


@pytest.mark.parametrize('base',TREES)
def test_training_saves_rate_and_rejects_unprepared(base,tmp_path):
    with training_tree(base) as trainer:
        from test_temporal import options
        opt=options(tmp_path,16);opt.sample_fps=8
        with pytest.raises(ValueError,match='sampling.json'): trainer.get_loaders(opt)
        opt.sample_fps=None
        training,_=trainer.get_loaders(opt)
        opt.sample_fps=8
        assert trainer.build_input_config(opt,training.dataset)['sample_fps']==8


def test_duplicate_timestamp_is_not_a_new_sample():
    s=FrameSampler(8)
    assert s.accept(0)
    assert not s.accept(0)
    assert not s.discontinuity
    assert s.accept(.125)


@pytest.mark.parametrize('base', TREES)
def test_reviewed_export_loads_for_training_with_matching_rate(base, tmp_path):
    raw=tmp_path/'raw'
    recording(raw,'one','s1');recording(raw,'two','s2')
    prep=load_prepare();initial=tmp_path/'initial';prep.prepare(raw,initial)
    with (initial/'review.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    for row,subset in zip(rows,['training','validation']):
        row.update(confirmed='true',subset=subset)
    with (initial/'review.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    out=tmp_path/'reviewed';prep.prepare(raw,out,review=initial/'review.csv')
    with training_tree(base) as trainer:
        from test_temporal import options
        opt=options(tmp_path,16)
        opt.video_path=str(out/'frames');opt.annotation_path=str(out/'annotations.json');opt.sample_fps=8
        train,val=trainer.get_loaders(opt)
        assert next(iter(train))[0].shape==(1,16,3,8,8)
        assert next(iter(val))[0].shape==(1,16,3,8,8)
        assert trainer.build_input_config(opt,train.dataset)['sample_fps']==8
        opt.sample_fps=30
        with pytest.raises(ValueError,match='does not match'):trainer.get_loaders(opt)
