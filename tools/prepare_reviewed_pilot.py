"""Materialize reviewed ranges for an explicitly marked within-session pilot."""
from pathlib import Path
import sys,json,csv,hashlib,shutil,collections,argparse
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from frame_sampling import FrameSampler,recording_timestamps
def prepare(output,window):
    output=Path(output)
    if output.exists():raise ValueError('Output already exists; choose a new output to preserve earlier exports')
    review=ROOT/'Prototype Dataset Review/review'
    inventory=json.loads((review/'inventory.json').read_text())
    decisions=json.loads((review/'reviewed_segments.json').read_text())
    items={i['take']:i for i in inventory['items']}
    hashes=json.loads((review/'frame_sha256.json').read_text())
    source={};correspondence=[]
    # Verify every imported image against original recording order, as well as ZIP hashes.
    for take,i in items.items():
        meta_path=Path(i['metadata'])
        video=meta_path.parent.parent/'videos'/(meta_path.stem+'.avi')
        cap=cv2.VideoCapture(str(video))
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stamps,meta=recording_timestamps(video,count,cap.get(cv2.CAP_PROP_FPS))
        if count!=i['frames'] or len(stamps)!=len(i['timestamps']):raise ValueError('Frame-count mismatch')
        if not np.allclose(stamps,i['timestamps'],atol=1e-8):raise ValueError('Timestamp correspondence changed')
        errors=[]
        for f in range(1,count+1):
            ok,raw=cap.read()
            if not ok:raise ValueError('Source decode failed')
            path=Path(i['folder'])/f'image_{f:05d}.jpg'
            key=str(path.relative_to(review.parent/'imported_original'))
            if hashlib.sha256(path.read_bytes()).hexdigest()!=hashes[key]:raise ValueError('Imported frame changed')
            extracted=cv2.imread(str(path))
            if extracted is None:raise ValueError('JPEG decode failed')
            h,w=extracted.shape[:2]
            # Two common resize methods; compression/resize differences are expected.
            err=min(float(np.abs(cv2.resize(raw,(w,h),interpolation=method).astype(float)-extracted).mean())
                    for method in [cv2.INTER_AREA,cv2.INTER_LINEAR,cv2.INTER_CUBIC])
            errors.append(err)
        cap.release()
        if max(errors)>12:raise ValueError(f'Frame correspondence needs inspection: take {take}, MAE {max(errors)}')
        source[take]=(stamps,meta,video)
        correspondence.append(dict(take=take,frames=count,max_resize_pixel_mae=max(errors),mean_resize_pixel_mae=float(np.mean(errors))))
    labels=['INTERACTION','PASSTHRU','WAIT']
    annotation={'labels':labels,'sample_fps':8.0,'annotations_confirmed':True,'reviewer':'AI visual review, not independent human ground truth',
                'split_protocol':'within_session_trial_pilot','sample_duration':window,'database':{}}
    timing={'sample_fps':8.0,'split_protocol':'within_session_trial_pilot','selection':'reviewed source ranges; shared FrameSampler at 8 fps','clips':{}}
    jobs=[];held=[];used={};source_subsets={}
    for k,s in enumerate(decisions['segments']):
        t=s['take'];i=items[t];stamps,meta,video=source[t];start=s['start_frame_inclusive'];end=s['end_frame_inclusive']
        if s['subset']!=i['subset']:raise ValueError('Review split changed from original')
        sid=meta['source_recording_id']
        if sid in source_subsets and source_subsets[sid]!=s['subset']:raise ValueError('Source trial split leak')
        source_subsets[sid]=s['subset']
        sampler=FrameSampler(8);indices=[]
        for f in range(start,end+1):
            if (t,f) in used:raise ValueError('Overlapping reviewed ranges')
            used[t,f]=dict(label=s['reviewed_label'],included=False)
            accepted=sampler.accept(stamps[f-1])
            if sampler.discontinuity:raise ValueError('Timing discontinuity inside reviewed range')
            if accepted:indices.append(f-1)
        if len(indices)<window:
            held.append(dict(take=t,start=start,end=end,label=s['reviewed_label'],sampled_frames=len(indices),reason=f'fewer than {window} real sampled frames'))
            continue
        name=f'{i["name"]}_review_{k:04d}'
        for f in indices:used[t,f+1]['included']=True
        jobs.append((s,i,meta,video,stamps,indices,name))
    output.mkdir(parents=True)
    for s,i,meta,video,stamps,indices,name in jobs:
        label=s['reviewed_label'];folder=output/'frames'/label/name;folder.mkdir(parents=True)
        for out_index,source_index in enumerate(indices,1):
            shutil.copy2(Path(i['folder'])/f'image_{source_index+1:05d}.jpg',folder/f'image_{out_index:05d}.jpg')
        (folder/'n_frames').write_text(str(len(indices)))
        group=meta.get('split_group_id') or meta['session_id']
        clip=dict(source_recording_id=meta['source_recording_id'],session_id=meta['session_id'],split_group_id=group,
                  participant_id=meta['participant_id'],source_video=str(video),source_frame_indices=indices,
                  source_timestamps_seconds=[stamps[f] for f in indices],sample_fps=8.0,frames=len(indices),
                  reviewed_source_range_inclusive=[s['start_frame_inclusive'],s['end_frame_inclusive']])
        timing['clips'][name]=clip
        annotation['database'][name]=dict(subset=s['subset'],annotations={'label':label},
                    source_recording_id=meta['source_recording_id'],split_group_id=group)
    (output/'frames/sampling.json').write_text(json.dumps(timing,indent=2))
    (output/'annotations.json').write_text(json.dumps(annotation,indent=2))
    (output/'held_out_short_segments.json').write_text(json.dumps(held,indent=2))
    (output/'frame_correspondence.json').write_text(json.dumps(correspondence,indent=2))
    with (output/'per_frame_labels.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['take','source_frame_1based','source_seconds','reviewed_label','included_in_export','subset'])
        for t,i in items.items():
            for n,stamp in enumerate(i['timestamps'],1):
                decision=used.get((t,n),{'label':'UNASSIGNED','included':False})
                writer.writerow([t,n,stamp,decision['label'],decision['included'],i['subset']])
    counts={subset:dict(collections.Counter(v['annotations']['label'] for v in annotation['database'].values() if v['subset']==subset))
            for subset in ['training','validation','testing']}
    summary=dict(source_trials=len(items),exported_source_trials=len({s['take'] for s,*_ in jobs}),
                 exported_segments=len(jobs),exported_frames=sum(len(j[5]) for j in jobs),
                 held_short_segments=len(held),unassigned_frames=sum((t,n) not in used for t,i in items.items() for n in range(1,i['frames']+1)),
                 segments_by_split_and_class=counts,sample_fps=8,sample_duration=window,
                 sessions=sorted({m['session_id'] for _,m,_ in source.values()}),
                 frame_correspondence_max_mae=max(c['max_resize_pixel_mae'] for c in correspondence),
                 review_sha256=hashlib.sha256((review/'reviewed_segments.json').read_bytes()).hexdigest(),
                 annotation_sha256=hashlib.sha256((output/'annotations.json').read_bytes()).hexdigest())
    (output/'preparation_summary.json').write_text(json.dumps(summary,indent=2))
    shutil.copy2(review/'reviewed_segments.json',output/'reviewed_ranges_snapshot.json')
    config=dict(sample_duration=window,sample_fps=8,video_path=str(output.resolve()/'frames'),annotation_path=str(output.resolve()/'annotations.json'),
                n_classes=3,sample_size=150,batch_size=1,num_workers=0,n_val_samples=1,allow_within_session_pilot=True)
    (output/'pilot_config.json').write_text(json.dumps(config,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--window',type=int,choices=[8,16],default=8);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();prepare(args.output,args.window)
