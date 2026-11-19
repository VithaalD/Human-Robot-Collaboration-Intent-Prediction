"""Cache the existing reviewed frame selections without changing dataset splits."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paired_features import PairedFeatureExtractor
from frame_sampling import recording_timestamps


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def valid_starts(timestamps):
    t = np.asarray(timestamps, dtype=np.float64)
    if t.ndim != 1 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('Invalid selected timestamps')
    return [i for i in range(len(t)-7) if np.all(np.diff(t[i:i+8]) <= 2.5/8)]


def verify_cache(path, indices, timestamps, label):
    with np.load(path, allow_pickle=False) as d:
        n = len(indices)
        assert d['features'].shape == (n, 318)
        assert d['features'].dtype == np.float32
        assert np.isfinite(d['features']).all()
        np.testing.assert_array_equal(d['frame_ids'], indices)
        np.testing.assert_allclose(d['timestamps'], timestamps, atol=1e-8, rtol=0)
        assert int(d['label']) == label
        np.testing.assert_array_equal(d['window_starts'], valid_starts(timestamps))
        assert len(d['window_starts']) > 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selection', default=str(ROOT.parent/'Prototype Training Runs/active_prototype.json'))
    ap.add_argument('--output', default=str(ROOT/'.runtime/paired-cache-v1'))
    ap.add_argument('--batch-size', type=int, default=8)
    args = ap.parse_args()
    if args.batch_size < 1:
        raise ValueError('batch-size must be positive')
    torch.set_num_threads(4)
    selection = json.loads(Path(args.selection).read_text())
    run = Path(selection['run_directory'])
    dataset = Path(selection['dataset'])
    annotations_path = run/'annotations.json'
    sampling_path = dataset/'frames/sampling.json'
    ann = json.loads(annotations_path.read_text())
    sampling = json.loads(sampling_path.read_text())
    split = json.loads((run/'split_manifest.json').read_text())
    labels = ['INTERACTION','PASSTHRU','WAIT']
    assert ann['labels'] == labels and ann['sample_fps'] == sampling['sample_fps'] == 8
    assert ann['sample_duration'] == 8
    sources, sessions, paths = {}, defaultdict(set), {}
    for name, item in ann['database'].items():
        c = sampling['clips'][name]; sid = item['source_recording_id']; subset = item['subset']
        assert subset in ('training','validation','testing')
        assert sid == c['source_recording_id'] and split['source_assignments'][sid] == subset
        assert sid not in sources, 'Repeated source recording in annotations'
        sources[sid] = subset
        video = str(Path(c['source_video']).resolve())
        assert video not in paths, 'Repeated video path'
        paths[video] = subset
        assert Path(video).is_file(), video
        assert len(c['source_frame_indices']) == len(c['source_timestamps_seconds']) >= 8
        assert all(b > a for a,b in zip(c['source_frame_indices'], c['source_frame_indices'][1:]))
        assert min(c['source_frame_indices']) >= 0
        assert valid_starts(c['source_timestamps_seconds'])
        sessions[c['session_id']].add(subset)
    extractor = PairedFeatureExtractor.from_checkpoints(selection['checkpoint'], ROOT/'models/custom_yolo11.pt')
    contract = {'schema_version': 'paired_cache_v1', 'input': extractor.metadata(),
                'intent_sha256': sha(selection['checkpoint']), 'yolo_sha256': sha(ROOT/'models/custom_yolo11.pt'),
                'annotations_sha256': sha(annotations_path), 'sampling_sha256': sha(sampling_path),
                'split_sha256': sha(run/'split_manifest.json'), 'labels': labels,
                'torch': str(torch.__version__), 'batch_size': args.batch_size}
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    contract_path = out/'contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError('Existing cache uses a different contract; choose a new output directory')
    contract_path.write_text(json.dumps(contract,indent=2))
    manifest = {'status':'building','signature':signature,'contract':contract,'clips':{},
                'limitations':['Existing within-session split: sessions/participants overlap across subsets.',
                               'YOLO training-source overlap with intent evaluation splits is unverified.',
                               'CNN projection comes from the previously validation-selected baseline.',
                               'Supplied labels and previously reviewed cores are retained, not independently relabeled.'],
                'shared_sessions':{k:sorted(v) for k,v in sessions.items() if len(v)>1},
                'reserved_test_processed':False,'intent_predictions_computed':False}
    begin=time.monotonic(); completed=0; total=sum(i['subset']!='testing' for i in ann['database'].values())
    parity_checked=False
    for name,item in ann['database'].items():
        c=sampling['clips'][name]; subset=item['subset']; label=labels.index(item['annotations']['label'])
        indices=c['source_frame_indices']; stamps=c['source_timestamps_seconds']
        entry={'subset':subset,'label':label,'label_name':labels[label],
               'source_recording_id':item['source_recording_id'],'source_video':c['source_video'],
               'session_id':c['session_id'],'frame_count':len(indices),'window_count':len(valid_starts(stamps)),
               'label_source':item.get('label_source','supplied'), 'status':'reserved_unprocessed'}
        manifest['clips'][name]=entry
        if subset=='testing': continue
        folder=out/subset; folder.mkdir(exist_ok=True)
        target=folder/(name+'.npz'); sidecar=target.with_suffix('.json')
        source_hash=sha(c['source_video'])
        provenance={'contract_signature':signature,'source_sha256':source_hash}
        if target.exists() and sidecar.exists() and json.loads(sidecar.read_text())==provenance:
            verify_cache(target,indices,stamps,label)
        else:
            cap=cv2.VideoCapture(c['source_video']); rows=[]; batch=[]; selected=0
            try:
                if not cap.isOpened(): raise RuntimeError('Unable to open '+c['source_video'])
                all_stamps,_=recording_timestamps(c['source_video'],int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),cap.get(cv2.CAP_PROP_FPS))
                np.testing.assert_allclose(np.asarray(all_stamps)[indices],stamps,atol=1e-6,rtol=0)
                for frame_id in range(indices[-1]+1):
                    ok,frame=cap.read()
                    if not ok: raise RuntimeError('Early end of video')
                    if selected>=len(indices) or frame_id!=indices[selected]: continue
                    if selected==0:
                        prepared=dataset/'frames'/labels[label]/name/'image_00001.png'
                        # Resolve numbering from the existing prepared frame list.
                        prepared=sorted(prepared.parent.glob('image_*.png'))[0]
                        with Image.open(prepared) as im: expected=extractor.engine._transform(im.convert('RGB'))
                        actual=extractor.engine._transform(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
                        assert torch.equal(expected,actual), 'Prepared/original preprocessing mismatch'
                    batch.append(frame); selected+=1
                    if len(batch)==args.batch_size or selected==len(indices):
                        cnn,objects=extractor.extract_batch(batch)
                        if not parity_checked:
                            one_cnn,one_objects=extractor(batch[0])
                            np.testing.assert_allclose(cnn[0],one_cnn,rtol=1e-4,atol=1e-4)
                            np.testing.assert_allclose(objects[0],one_objects,rtol=1e-3,atol=1e-3)
                            manifest['batch_vs_single_max_error']={'cnn':float(np.abs(cnn[0]-one_cnn).max()),'objects':float(np.abs(objects[0]-one_objects).max())}
                            parity_checked=True
                        rows.append(np.concatenate((cnn,objects),axis=1)); batch=[]
                assert selected==len(indices)
            finally: cap.release()
            features=np.concatenate(rows).astype(np.float32)
            partial=target.with_suffix('.partial.npz')
            np.savez_compressed(partial, features=features, frame_ids=np.array(indices,dtype=np.int64),
                                timestamps=np.array(stamps,dtype=np.float64), label=np.array(label,dtype=np.int64),
                                window_starts=np.array(valid_starts(stamps),dtype=np.int64))
            verify_cache(partial,indices,stamps,label); partial.replace(target)
            sidecar.write_text(json.dumps(provenance,indent=2))
        entry.update(status='cached',cache_path=target.relative_to(out).as_posix(),sha256=sha(target),source_sha256=source_hash)
        completed+=1
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
        print(f'{completed}/{total}: {subset} {name} ({len(indices)} frames), elapsed {time.monotonic()-begin:.1f}s',flush=True)
    for name,entry in manifest['clips'].items():
        if entry['status']=='cached':
            c=sampling['clips'][name]
            verify_cache(out/entry['cache_path'],c['source_frame_indices'],c['source_timestamps_seconds'],entry['label'])
            assert sha(out/entry['cache_path'])==entry['sha256']
    manifest['status']='complete'
    manifest['summary']={s:{'recordings':sum(e['subset']==s for e in manifest['clips'].values()),
                            'cached_frames':sum(e['frame_count'] for e in manifest['clips'].values() if e['subset']==s and e['status']=='cached'),
                            'available_windows':sum(e['window_count'] for e in manifest['clips'].values() if e['subset']==s and e['status']=='cached')}
                         for s in ('training','validation','testing')}
    manifest['elapsed_seconds']=time.monotonic()-begin
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest['summary'],indent=2),flush=True)


if __name__=='__main__': main()
