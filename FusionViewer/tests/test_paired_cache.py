import json
import numpy as np
import pytest
from paired_cache_dataset import PairedCacheDataset
from tools.build_paired_cache import valid_starts


def test_windows_do_not_cross_time_gaps():
    stamps = list(np.arange(8)/8) + list(3+np.arange(8)/8)
    assert valid_starts(stamps) == [0,8]
    with pytest.raises(ValueError):
        valid_starts([0,0,.125])


def test_loader_preserves_split_label_and_center_window(tmp_path):
    features = np.repeat(np.arange(12,dtype=np.float32)[:,None],318,axis=1)
    np.savez(tmp_path/'clip.npz',features=features,label=np.array(2),window_starts=np.arange(5))
    manifest={'status':'complete','contract':{},'clips':{'clip':{'subset':'validation','status':'cached','cache_path':'clip.npz','label':2}}}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    ds=PairedCacheDataset(tmp_path,'validation')
    x,y=ds[0]
    assert len(ds)==1 and x.shape==(8,318) and y.item()==2
    np.testing.assert_array_equal(x[:,0].numpy(),np.arange(2,10))
    with pytest.raises(ValueError,match='reserved test'):
        PairedCacheDataset(tmp_path,'testing')
    manifest['status']='building'
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='not complete'):
        PairedCacheDataset(tmp_path,'validation')
