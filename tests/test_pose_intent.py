"""Synthetic contract tests; no accuracy claim and no DLC weights downloaded."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
import cv2
from pose_intent.features import InputConfig, LANDMARKS, FEATURE_DIM, sample_indices, motion_features
from pose_intent.models import IntentModel, RGB_CONFIG, file_hash
from pose_intent.data import prepare, read_manifest, WindowDataset
from pose_intent.training import train, evaluate
from pose_intent.dlc import read_pose_table
from pose_intent.runtime import FramePacket, PoseWorker, PoseIntentRuntime
from pose_intent.replay import replay


def poses(n):
    xy = np.array([[30, 20], [70, 20], [25, 40], [75, 40], [20, 60], [80, 60], [80, 75], [80, 65]])
    result = np.ones((n, 8, 3), dtype=float)
    result[:, :, :2] = xy
    result[:, 4, 0] += np.arange(n) * 0.2
    return result


def save_checkpoint(path, cfg, modality='motion'):
    torch.manual_seed(7)
    model = IntentModel(modality)
    torch.save(dict(format='pose-intent-v1', rgb_config=RGB_CONFIG, input_config=cfg.dictionary(),
                    hidden=64, modality=modality, state_dict=model.state_dict(), pose_identity={'test': 'pose'},
                    rgb_sha256='wrong'), path)
    return model


def test_sampling_never_looks_ahead_or_repeats():
    cfg = InputConfig(frames=4)
    times = np.arange(30) / 30
    ids = sample_indices(times, 0.81, cfg)
    assert np.all(times[ids] <= 0.81 - np.arange(3, -1, -1) / 10 + 1e-9)
    assert np.all(np.diff(ids) > 0)
    with pytest.raises(ValueError):
        sample_indices(np.array([0, 0.8, 1]), 1, cfg)
    with pytest.raises(ValueError):
        sample_indices(np.array([0, 0.1, 0.1, 0.2]), 0.2, cfg)


def test_features_scale_invariance_missing_and_causality():
    cfg = InputConfig(frames=4)
    p, times = poses(4), np.arange(4) / 10
    f, m, c, quality = motion_features(p, times, (100, 100), cfg)
    scaled = p.copy()
    scaled[:, :, :2] *= 2
    f2, m2, _, _ = motion_features(scaled, times, (200, 200), cfg)
    np.testing.assert_allclose(f, f2)
    assert f.shape == (4, FEATURE_DIM) and quality == 1
    changed = p.copy()
    changed[-1, :, :2] += 20
    f3, _, _, _ = motion_features(changed, times, (100, 100), cfg)
    np.testing.assert_allclose(f[:-1], f3[:-1])
    p[:, 4] = np.nan
    bad, mask, confidence, quality = motion_features(p, times, (100, 100), cfg)
    assert np.isfinite(bad).all() and np.isfinite(confidence).all()
    assert np.all(bad[mask == 0] == 0) and quality < 1
    assert np.all(mask[:, 8:10] == 0)


@pytest.mark.parametrize('modality', ['motion', 'fusion'])
def test_pose_is_used_and_can_train(modality):
    torch.manual_seed(1)
    model = IntentModel(modality).eval()
    features = torch.randn(2, 4, FEATURE_DIM, requires_grad=True)
    mask = torch.ones_like(features)
    confidence = torch.ones(2, 4, 8)
    rgb = torch.randn(2, 128) if modality == 'fusion' else None
    a = model(features, mask, confidence, rgb)
    b = model(features + 1, mask, confidence, rgb)
    assert not torch.allclose(a, b)
    a.sum().backward()
    assert features.grad.abs().sum() > 0
    if modality == 'fusion':
        assert not torch.allclose(a, model(features, mask, confidence, rgb + 1))


def test_stale_low_quality_and_fusion_hash(tmp_path):
    cfg = InputConfig(frames=4)
    checkpoint = tmp_path / 'motion.pt'
    save_checkpoint(checkpoint, cfg)
    runtime = PoseIntentRuntime(checkpoint)
    p = poses(4)
    obs = [(FramePacket(i, i / 10, np.zeros((100, 100, 3), np.uint8)), p[i]) for i in range(4)]
    assert runtime.predict(obs, 0.3)['status'] == 'OK'
    assert runtime.predict(obs, 2)['status'] == 'UNAVAILABLE'
    missing = [(packet, np.zeros((8, 3))) for packet, _ in obs]
    result = runtime.predict(missing, 0.3)
    assert result['status'] == 'UNAVAILABLE' and result['probabilities'] is None
    save_checkpoint(checkpoint, cfg, 'fusion')
    with pytest.raises(ValueError, match='exact RGB'):
        PoseIntentRuntime(checkpoint)


def test_worker_preserves_identity_and_bounds_backlog():
    worker = PoseWorker(lambda rgb: np.full((8, 3), rgb[0, 0, 0]), capacity=3)
    try:
        for i in range(20):
            worker.submit(FramePacket(i, i / 10, np.full((10, 10, 3), i, np.uint8)))
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            result, error = worker.snapshot()
            if result and result[-1][0].frame_id == 19:
                break
            time.sleep(0.01)
        assert not error and result[-1][0].frame_id == 19
        assert len(result) <= 3
        for packet, pose in result:
            assert pose[0, 0] == packet.frame_id
            assert packet.original_size == (10, 10)
    finally:
        worker.close()
    assert not worker.thread.is_alive()


@pytest.fixture
def fixture_manifest(tmp_path):
    takes = []
    for k, split in enumerate(('train', 'val', 'test')):
        video = tmp_path / f'{split}.avi'
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (100, 100))
        assert writer.isOpened()
        for i in range(40):
            writer.write(np.full((100, 100, 3), k * 60 + i, np.uint8))
        writer.release()
        csv = tmp_path / f'{split}.csv'
        cols = pd.MultiIndex.from_product([['fixture'], LANDMARKS, ['x', 'y', 'likelihood']])
        pd.DataFrame(poses(40).reshape(40, -1), columns=cols).to_csv(csv)
        csv.with_suffix('.pose.json').write_text(json.dumps(dict(video_sha256=file_hash(video), pose_sha256=file_hash(csv), pose_identity={'test': 'pose'})))
        metadata = dict(take_id=split, session_id=split, participant_id=split, camera_setup_id='camera1',
                        frame_timestamps_seconds=(np.arange(40) / 10).tolist())
        meta_path = tmp_path / f'{split}.json'
        meta_path.write_text(json.dumps(metadata))
        takes.append(dict(video=str(video), pose=str(csv), metadata=str(meta_path), split=split,
                          **{key: metadata[key] for key in ('take_id', 'session_id', 'participant_id', 'camera_setup_id')},
                          annotations_confirmed=True, segments=[dict(start=0, end=1.3, label='WAIT'),
                          dict(start=1.3, end=2.6, label='INTERACTION'), dict(start=2.6, end=4, label='PASSTHRU')]))
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(dict(schema_version=1, takes=takes)))
    return path


def test_split_leakage_and_pose_rows(fixture_manifest):
    manifest = json.loads(fixture_manifest.read_text())
    manifest['takes'][1]['session_id'] = 'train'
    altered = fixture_manifest.parent / 'leak.json'
    altered.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='leakage'):
        read_manifest(altered)
    pose_path = manifest['takes'][0]['pose']
    df = pd.read_csv(pose_path, header=[0, 1, 2], index_col=0)
    df.index = np.arange(1, 41)
    df.to_csv(pose_path)
    with pytest.raises(ValueError, match='frame IDs'):
        read_pose_table(pose_path)


def test_prepare_train_evaluate_and_replay(fixture_manifest, tmp_path):
    torch.set_num_threads(2)
    cfg = InputConfig(frames=4)
    folder, run = tmp_path / 'windows', tmp_path / 'run'
    result = prepare(fixture_manifest, folder, cfg)
    assert result['windows'] > 30
    train(folder, run, epochs=2, batch_size=8)
    report, rows = evaluate(folder, run / 'best.pt')
    assert report['windows'] == len(rows) > 0 and report['interaction_events']
    take = read_manifest(fixture_manifest)['takes'][2]
    output = tmp_path / 'replay.jsonl'
    result = replay(run / 'best.pt', take['video'], take['metadata'], take['pose'], output)
    assert result['available'] > 0
    replay_rows = [json.loads(line) for line in output.read_text().splitlines()]
    for row in rows:
        match = next(r for r in replay_rows if abs(r['timestamp'] - row['end']) < 1e-7)
        np.testing.assert_allclose(match['probabilities'], row['probabilities'], atol=1e-6)
    # Test data cannot update training normalization or selected weights.
    saved = torch.load(run / 'best.pt', weights_only=True)
    assert saved['train_sessions'] == ['train'] and saved['validation_sessions'] == ['val']
    index_path = folder / 'index.json'
    index = json.loads(index_path.read_text())
    index['input_config']['sample_hz'] = 5
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match='configuration'):
        evaluate(folder, run / 'best.pt')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('modality', ['motion', 'fusion'])
def test_gpu_gradient_step(modality):
    model = IntentModel(modality).cuda()
    optimizer = torch.optim.Adam(model.parameters())
    f = torch.randn(2, 16, FEATURE_DIM, device='cuda')
    rgb = torch.randn(2, 128, device='cuda') if modality == 'fusion' else None
    loss = model(f, torch.ones_like(f), torch.ones(2, 16, 8, device='cuda'), rgb).square().mean()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    assert torch.isfinite(loss)
