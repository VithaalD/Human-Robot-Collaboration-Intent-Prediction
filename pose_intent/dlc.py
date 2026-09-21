"""DLC 3.0.1 adapter; imports DLC only when extraction/live pose is requested."""
import numpy as np
from .features import LANDMARKS


def read_pose_table(path):
    import pandas as pd
    if str(path).lower().endswith('.csv'):
        df = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    else:
        df = pd.read_hdf(path)
    if df.columns.nlevels != 3 or len(df.columns.get_level_values(0).unique()) != 1:
        raise ValueError('Use one scorer and one subject, with standard DLC x/y/likelihood columns')
    if not np.array_equal(np.asarray(df.index), np.arange(len(df))):
        raise ValueError('DLC rows must be contiguous original video frame IDs starting at zero')
    scorer = df.columns.get_level_values(0)[0]
    expected = [(scorer, name, coord) for name in LANDMARKS for coord in ('x', 'y', 'likelihood')]
    if df.columns.duplicated().any() or any(key not in df.columns for key in expected):
        raise ValueError('Missing or duplicate landmark columns; see the required naming schema')
    return df.loc[:, expected].to_numpy(dtype=float).reshape(-1, len(LANDMARKS), 3)


class DLCPredictor:
    """Full-frame, single-subject bottom-up PyTorch inference (no crop offsets)."""
    def __init__(self, project_config, model_config, snapshot, device='cpu'):
        import deeplabcut
        import yaml
        from deeplabcut.core.config import read_config_as_dict
        from deeplabcut.pose_estimation_pytorch import get_pose_inference_runner
        if deeplabcut.__version__ != '3.0.1':
            raise ValueError('This adapter is verified against DeepLabCut 3.0.1')
        with open(project_config, encoding='utf-8') as stream:
            project = yaml.safe_load(stream)
        if project.get('multianimalproject') or project.get('cropping'):
            raise ValueError('Use a single-subject project with cropping disabled')
        names = project.get('bodyparts', [])
        if len(names) != len(LANDMARKS) or set(names) != set(LANDMARKS):
            raise ValueError('DLC project must define exactly the eight documented landmarks')
        self.order = [names.index(name) for name in LANDMARKS]
        model = read_config_as_dict(model_config)
        if str(model.get('method')).upper() not in ('BU', 'BOTTOMUP') or model.get('metadata', {}).get('bodyparts') != names:
            raise ValueError('Use a bottom-up (bu) model with the same landmark order as its project')
        self.runner = get_pose_inference_runner(model, snapshot, device=device, batch_size=1)

    def __call__(self, rgb):
        predictions = self.runner.inference(images=[np.ascontiguousarray(rgb)])
        if len(predictions) != 1 or 'bodyparts' not in predictions[0]:
            raise ValueError('Unsupported DLC prediction format')
        points = np.asarray(predictions[0]['bodyparts'])
        if points.shape != (1, len(LANDMARKS), 3):
            raise ValueError('Expected exactly one tracked person and eight landmarks')
        return points[0, self.order].copy()
