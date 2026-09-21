"""A frozen legacy RGB encoder and a small trainable motion/fusion classifier."""
import hashlib
from pathlib import Path
import torch
from torch import nn
from torchvision.models import resnet101
from torchvision import transforms
from PIL import Image
from .features import FEATURE_DIM, LANDMARKS, LABELS, InputConfig

RGB_CONFIG = dict(sample_size=150, norm_value=1.0,
                  mean=[114.7748, 107.7354, 99.4750], std=[1.0, 1.0, 1.0], color_order='RGB')


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class VideoEncoder(nn.Module):
    """Names/shapes match the supplied ResNet101-LSTM checkpoint exactly."""
    def __init__(self):
        super().__init__()
        self.resnet = resnet101(weights=None)
        self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 300))
        self.lstm = nn.LSTM(300, 256, 3)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, len(LABELS))

    def forward(self, frames):
        hidden = None
        for frame in frames.unbind(1):
            out, hidden = self.lstm(self.resnet(frame).unsqueeze(0), hidden)
        embedding = torch.relu(self.fc1(out[-1]))
        return embedding, self.fc2(embedding)


def load_video_encoder(path, cfg, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    metadata = checkpoint.get('input_config')
    if metadata:
        expected = dict(RGB_CONFIG, sample_duration=cfg.frames, n_classes=3, class_labels=list(LABELS))
        for key, value in expected.items():
            if key in metadata and metadata[key] != value:
                raise ValueError(f'RGB checkpoint mismatch: {key}')
    else:
        import warnings
        warnings.warn('Legacy RGB checkpoint has unknown training settings; validate the new sampling rate.')
    state = checkpoint.get('state_dict', checkpoint)
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    model = VideoEncoder()
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    return model.to(device).eval()


def image_tensor(rgb):
    transform = transforms.Compose([transforms.Resize((150, 150)), transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x * 255),
                                    transforms.Normalize(RGB_CONFIG['mean'], RGB_CONFIG['std'])])
    return transform(Image.fromarray(rgb))


class IntentModel(nn.Module):
    def __init__(self, modality='motion', hidden=64):
        super().__init__()
        if modality not in ('motion', 'fusion'):
            raise ValueError('Choose motion or fusion')
        self.modality = modality
        self.hidden = hidden
        self.register_buffer('feature_mean', torch.zeros(FEATURE_DIM))
        self.register_buffer('feature_std', torch.ones(FEATURE_DIM))
        self.motion = nn.GRU(FEATURE_DIM * 2 + len(LANDMARKS), hidden, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(hidden + (128 if modality == 'fusion' else 0), 64),
                                        nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 3))

    def forward(self, features, mask, likelihood, rgb=None):
        normalized = ((features - self.feature_mean) / self.feature_std).clamp(-10, 10) * mask
        _, hidden = self.motion(torch.cat([normalized, mask, likelihood], dim=-1))
        vector = hidden[-1]
        if self.modality == 'fusion':
            if rgb is None or rgb.shape[-1] != 128:
                raise ValueError('Fusion requires synchronized 128-dimensional RGB features')
            vector = torch.cat([vector, rgb], dim=-1)
        return self.classifier(vector)


def load_intent(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format') != 'pose-intent-v1' or checkpoint.get('rgb_config') != RGB_CONFIG:
        raise ValueError('Unsupported pose-intent checkpoint')
    cfg = InputConfig(**checkpoint['input_config'])
    model = IntentModel(checkpoint['modality'], checkpoint['hidden'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.to(device).eval(), cfg, checkpoint
