"""Clip-level dataset for cached paired features, preserving baseline crop policy."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class PairedCacheDataset(Dataset):
    def __init__(self, cache_directory, subset):
        if subset not in ('training', 'validation'):
            raise ValueError('Only training/validation are cached; reserved test stays unopened')
        self.root = Path(cache_directory)
        manifest = json.loads((self.root / 'manifest.json').read_text())
        if manifest['status'] != 'complete':
            raise ValueError('Cache build is not complete')
        self.subset = subset
        self.contract = manifest['contract']
        self.items = [(name, entry) for name, entry in manifest['clips'].items()
                      if entry['subset'] == subset]
        if not self.items or any(e['status'] != 'cached' for _, e in self.items):
            raise ValueError('Requested split has missing caches')
        self.labels = [entry['label'] for _, entry in self.items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        _, entry = self.items[index]
        with np.load(self.root / entry['cache_path'], allow_pickle=False) as data:
            starts = data['window_starts']
            if self.subset == 'training':
                start = int(starts[torch.randint(len(starts), (1,)).item()])
            else:
                center = (len(data['features']) - 8) // 2
                start = int(starts[np.abs(starts - center).argmin()])
            features = data['features'][start:start+8].copy()
            label = int(data['label'])
        if features.shape != (8, 318) or not np.isfinite(features).all() or label != entry['label']:
            raise ValueError('Invalid cached sample')
        return torch.from_numpy(features), torch.tensor(label, dtype=torch.long)
