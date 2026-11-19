"""Verify all completed paired caches and exercise the training data loader."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paired_cache_dataset import PairedCacheDataset
from tools.build_paired_cache import verify_cache, sha


def main():
    cache = ROOT / '.runtime/paired-cache-v1'
    manifest = json.loads((cache/'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    selection = json.loads((ROOT.parent/'Prototype Training Runs/active_prototype.json').read_text())
    sampling_path = Path(selection['dataset'])/'frames/sampling.json'
    ann_path = Path(selection['run_directory'])/'annotations.json'
    assert sha(sampling_path) == manifest['contract']['sampling_sha256']
    assert sha(ann_path) == manifest['contract']['annotations_sha256']
    assert sha(selection['checkpoint']) == manifest['contract']['intent_sha256']
    assert sha(ROOT/'models/custom_yolo11.pt') == manifest['contract']['yolo_sha256']
    ann = json.loads(ann_path.read_text())
    sampling = json.loads(sampling_path.read_text())
    assert set(manifest['clips']) == set(ann['database'])
    source_splits = {}
    total_frames = total_windows = 0
    for name, entry in manifest['clips'].items():
        annotation = ann['database'][name]
        assert entry['subset'] == annotation['subset']
        assert entry['label_name'] == annotation['annotations']['label']
        assert entry['label'] == ann['labels'].index(entry['label_name'])
        source = entry['source_recording_id']
        assert source not in source_splits
        source_splits[source] = entry['subset']
        if entry['subset'] == 'testing':
            assert entry['status'] == 'reserved_unprocessed' and 'cache_path' not in entry
            continue
        path = cache/entry['cache_path']
        assert sha(path) == entry['sha256'], path
        clip = sampling['clips'][name]
        verify_cache(path,clip['source_frame_indices'],clip['source_timestamps_seconds'],entry['label'])
        total_frames += entry['frame_count']; total_windows += entry['window_count']
    assert total_frames == 7674 and total_windows == 6673
    assert len(list(cache.rglob('*.npz'))) == 143
    torch.manual_seed(20261117)
    loaded = {}
    for subset, expected in [('training',127),('validation',16)]:
        dataset = PairedCacheDataset(cache,subset)
        assert len(dataset) == expected
        count = 0
        for x,y in DataLoader(dataset,batch_size=8,shuffle=False,num_workers=0):
            assert x.shape[1:] == (8,318) and x.dtype == torch.float32
            assert y.dtype == torch.int64 and torch.isfinite(x).all()
            count += len(y)
        assert count == expected
        loaded[subset] = count
    report = {'status':'passed','cache_directory':str(cache),'summary':manifest['summary'],
              'verified_cache_files':143,'verified_frames':total_frames,'available_overlapping_windows':total_windows,
              'all_cache_hashes_match':True,'labels_splits_timestamps_verified':True,
              'loader_recordings_checked':loaded,'sample_shape':[8,318],
              'source_recording_overlap':False,'reserved_test_processed':False,
              'batch_vs_single_max_error':manifest.get('batch_vs_single_max_error'),
              'limitations':manifest['limitations'],'new_intent_model_trained':False}
    (ROOT/'paired-cache-verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__': main()
