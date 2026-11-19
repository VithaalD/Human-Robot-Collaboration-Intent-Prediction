"""Temporal intent head for frozen CNN (300) + YOLO (18) features."""
import torch
from torch import nn
from torch.nn import functional as F

LABELS = ('INTERACTION', 'PASSTHRU', 'WAIT')
ARCHITECTURE = {'name': 'paired_intent_lstm_v1', 'input_size': 318,
                'cnn_features': 300, 'object_features': 18, 'hidden_size': 256,
                'num_layers': 3, 'head_size': 128, 'num_classes': 3,
                'sequence_length': 8, 'batch_first': True, 'labels': list(LABELS)}


class FusedIntentLSTM(nn.Module):
    """Input [batch,8,318]; output unnormalized logits [batch,3].

    CNN and YOLO are separate frozen extractors. The first 300 inputs must be
    CNN features; the last 18 follow the cache's object-feature schema.
    """
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(318, 256, num_layers=3, batch_first=True)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, features):
        if features.ndim != 3 or features.shape[1:] != (8, 318):
            raise ValueError('Expected features shaped [batch,8,318]')
        if not features.is_floating_point() or not torch.isfinite(features).all():
            raise ValueError('Expected finite floating-point features')
        sequence, _ = self.lstm(features)
        return self.fc2(F.relu(self.fc1(sequence[:, -1])))

    @classmethod
    def from_baseline(cls, checkpoint):
        """Copy temporal weights; zero the new object-input columns.

        This preserves baseline behavior at initialization. New columns remain
        trainable and receive gradients; the model learns object use in step 10.
        """
        config = checkpoint.get('input_config', {})
        if config.get('class_labels') != list(LABELS) or config.get('sample_duration') != 8:
            raise ValueError('Baseline must use eight frames and the expected label order')
        source = checkpoint['state_dict']
        source = {k.removeprefix('module.'): v for k, v in source.items()}
        model = cls()
        copied = {}
        for key, target in model.state_dict().items():
            if key not in source:
                raise ValueError(f'Missing baseline parameter: {key}')
            value = source[key].detach().cpu()
            if key == 'lstm.weight_ih_l0':
                if value.shape != (1024, 300):
                    raise ValueError('Baseline LSTM input weights do not match 300 features')
                target.zero_()
                target[:, :300].copy_(value)
                copied[key] = target
            else:
                if value.shape != target.shape:
                    raise ValueError(f'Incompatible baseline parameter: {key}')
                copied[key] = value.clone()
        model.load_state_dict(copied, strict=True)
        return model


def load_fused_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('architecture') != ARCHITECTURE:
        raise ValueError('Unsupported fusion checkpoint architecture')
    model = FusedIntentLSTM()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.to(device), checkpoint
