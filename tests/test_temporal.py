"""Regression checks for duration propagation and checkpoint input contracts."""
import importlib
import importlib.util
import json
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TREES = [ROOT / 'intentpredictionattempt4-main', ROOT / 'test code and model weights for mecha lab-20260904T185537Z-1-001/test code and model weights for mecha lab/cnn-lstm-master']
ENGINES = [ROOT / 'Intent Monitor/intent_engine_v3_1_windows.py', ROOT / 'Intent Monitor/intent-monitor/intent_engine_v3_1_windows.py', TREES[0] / 'intent_engine_v3_1_windows.py']
LABELS = ['INTERACTION', 'PASSTHRU', 'WAIT']


@contextmanager
def training_tree(base):
    names = {'main', 'train', 'validation', 'opts', 'model', 'models', 'dataset', 'datasets', 'mean', 'spatial_transforms', 'temporal_transforms', 'target_transforms', 'utils'}
    prior = {key: value for key, value in sys.modules.items() if key.split('.')[0] in names}
    for key in prior:
        sys.modules.pop(key)
    sys.path.insert(0, str(base))
    prior_tensorboard = sys.modules.get('tensorboardX')
    sys.modules['tensorboardX'] = types.ModuleType('tensorboardX')
    try:
        # tensorboardX is irrelevant to data loading; no training is performed.
        yield importlib.import_module('main')
    finally:
        if prior_tensorboard is None:
            sys.modules.pop('tensorboardX', None)
        else:
            sys.modules['tensorboardX'] = prior_tensorboard
        sys.path.remove(str(base))
        for key in list(sys.modules):
            if key.split('.')[0] in names:
                sys.modules.pop(key)
        sys.modules.update(prior)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=ENGINES, ids=['monitor', 'compatibility', 'terminal'])
def engine(request):
    return load_module(request.param, 'tested_engine_' + str(ENGINES.index(request.param)))


def options(tmp_path, duration):
    root = tmp_path / 'frames'
    entries = {}
    for name, subset, count in [('short_train', 'training', 3), ('long_train', 'training', 21), ('short_val', 'validation', 3), ('long_val', 'validation', 21)]:
        folder = root / LABELS[0] / name
        folder.mkdir(parents=True)
        (folder / 'n_frames').write_text(str(count))
        for index in range(1, count + 1):
            Image.new('RGB', (10, 10), (index, 30, 60)).save(folder / f'image_{index:05d}.jpg')
        entries[name] = {'subset': subset, 'annotations': {'label': LABELS[0]}}
    annotation = tmp_path / 'labels.json'
    annotation.write_text(json.dumps({'labels': LABELS, 'database': entries}))
    return SimpleNamespace(dataset='ucf101', video_path=str(root), annotation_path=str(annotation), sample_duration=duration, sample_size=8, n_val_samples=1, norm_value=1, mean_dataset='activitynet', no_mean_norm=False, std_norm=False, batch_size=2, num_workers=0, n_classes=3)


@pytest.mark.parametrize('base', TREES, ids=['primary', 'archived'])
@pytest.mark.parametrize('duration', [8, 16])
def test_real_train_and_validation_loaders_honor_duration(base, duration, tmp_path):
    with training_tree(base) as trainer:
        opt = options(tmp_path, duration)
        train, val = trainer.get_loaders(opt)
        original = [list(item['frame_indices']) for item in val.dataset.data]
        for loader in (train, val):
            clips, labels = next(iter(loader))
            assert clips.shape == (2, duration, 3, 8, 8)
            assert labels.shape == (2,)
        assert original == [item['frame_indices'] for item in val.dataset.data]
        metadata = trainer.build_input_config(opt, train.dataset)
        assert metadata['sample_duration'] == duration
        assert metadata['class_labels'] == LABELS
        assert metadata['sample_size'] == 8
        saved = tmp_path / 'metadata.pth'
        torch.save({'input_config': metadata}, saved)
        assert torch.load(saved, weights_only=True)['input_config'] == metadata


@pytest.mark.parametrize('base', TREES, ids=['primary', 'archived'])
@pytest.mark.parametrize('duration', [8, 16])
def test_both_dataset_constructors_receive_duration(base, duration):
    with training_tree(base):
        dataset = importlib.import_module('dataset')
        opt = SimpleNamespace(dataset='ucf101', video_path='frames', annotation_path='labels.json', sample_duration=duration, n_val_samples=3)
        with mock.patch.object(dataset, 'UCF101') as constructor:
            dataset.get_training_set(opt, None, None, None)
            assert constructor.call_args.kwargs['sample_duration'] == duration
            dataset.get_validation_set(opt, None, None, None)
            assert constructor.call_args.kwargs['sample_duration'] == duration


@pytest.mark.parametrize('base', TREES, ids=['primary', 'archived'])
@pytest.mark.parametrize('duration', [8, 16])
def test_temporal_crops_exact_length_no_mutation(base, duration):
    module = load_module(base / 'temporal_transforms.py', 'temporal_test')
    for kind in (module.LoopPadding, module.TemporalBeginCrop, module.TemporalCenterCrop, module.TemporalRandomCrop):
        for count in (1, 3, duration, duration + 5):
            source = list(range(count))
            result = kind(duration)(source)
            assert len(result) == duration
            assert source == list(range(count))
            assert all(index in source for index in result)
        with pytest.raises(ValueError, match='without frames'):
            kind(duration)([])
        with pytest.raises(ValueError, match='positive'):
            kind(0)
    with mock.patch.object(module.random, 'randint', return_value=4) as choose:
        assert module.TemporalRandomCrop(duration)(list(range(duration + 4))) == list(range(4, duration + 4))
        choose.assert_called_once_with(0, 4)


@pytest.mark.parametrize('duration', [8, 16])
def test_legacy_checkpoint_accepts_explicit_duration(engine, duration, caplog):
    cfg = engine.EngineConfig(sample_duration=duration)
    engine._validate_checkpoint_input({'state_dict': {}}, cfg, LABELS)
    assert 'historical frame duration is unknown' in caplog.text
    assert engine.EngineConfig().sample_duration == 16


@pytest.mark.parametrize('key,saved_value', [('sample_duration', 8), ('sample_size', 112), ('class_labels', list(reversed(LABELS))), ('norm_value', 255), ('mean', [0, 0, 0]), ('std', [2, 2, 2])])
def test_metadata_mismatches_are_actionable(engine, key, saved_value):
    with pytest.raises(ValueError, match='Checkpoint input mismatch: ' + key):
        engine._validate_checkpoint_input({'input_config': {key: saved_value}}, engine.EngineConfig(), LABELS)


def test_matching_metadata_accepted(engine):
    cfg = engine.EngineConfig()
    engine._validate_checkpoint_input({'input_config': {'sample_duration': 16, 'sample_size': 150, 'class_labels': LABELS, 'norm_value': 1, 'mean': cfg.mean, 'std': cfg.std, 'color_order': 'RGB', 'n_classes': 3}}, cfg, LABELS)


def test_engine_rejects_mismatch_before_model_allocation(engine, tmp_path):
    path = tmp_path / 'mismatch.pth'
    torch.save({'state_dict': {}, 'input_config': {'sample_duration': 8}}, path)
    with mock.patch.object(engine, 'CNNLSTM') as model:
        with pytest.raises(ValueError, match='--sample_duration 8'):
            engine.IntentEngine(engine.EngineConfig(resume_path=str(path), class_labels=LABELS))
        model.assert_not_called()


@pytest.mark.parametrize('base', TREES, ids=['primary', 'archived'])
def test_resume_checks_duration_and_restores_on_cpu(base, tmp_path):
    with training_tree(base) as trainer:
        model = torch.nn.Linear(2, 3)
        optimizer = torch.optim.Adam(model.parameters())
        path = tmp_path / 'resume.pth'
        torch.save({'epoch': 4, 'state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'input_config': {'sample_duration': 16}}, path)
        opt = SimpleNamespace(resume_path=str(path), input_config={'sample_duration': 16})
        assert trainer.resume_model(opt, model, optimizer) == 5
        opt.input_config['sample_duration'] = 8
        with pytest.raises(ValueError, match='Cannot resume.*sample_duration'):
            trainer.resume_model(opt, model, optimizer)
