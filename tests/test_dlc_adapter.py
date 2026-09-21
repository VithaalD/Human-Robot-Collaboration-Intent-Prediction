"""Optional real DLC adapter smoke test with random weights, not trained poses."""
import numpy as np
import pytest
import torch
from pose_intent.features import LANDMARKS
from pose_intent.dlc import DLCPredictor


def test_actual_dlc_runner(tmp_path):
    pytest.importorskip('deeplabcut')
    from deeplabcut.core.config import ProjectConfig
    from deeplabcut.pose_estimation_pytorch.config.pose import PoseConfig
    from deeplabcut.pose_estimation_pytorch.models import PoseModel
    project_path, model_path = tmp_path / 'config.yaml', tmp_path / 'pytorch_config.yaml'
    project = ProjectConfig.from_dict(dict(Task='smoke', scorer='tester', date='Sep21',
                                           project_path=str(tmp_path), video_sets={},
                                           bodyparts=list(LANDMARKS), multianimalproject=False))
    project.to_yaml(project_path)
    model_cfg = PoseConfig.build(project, model_path, top_down=False, net_type='resnet_50', save=True)
    model = PoseModel.build(model_cfg['model'], pretrained_backbone=False)
    snapshot = tmp_path / 'snapshot.pt'
    torch.save({'model': model.state_dict()}, snapshot)
    predictor = DLCPredictor(project_path, model_path, snapshot,
                             'cuda:0' if torch.cuda.is_available() else 'cpu')
    output = predictor(np.zeros((128, 128, 3), dtype=np.uint8))
    assert output.shape == (8, 3) and np.isfinite(output).all()
