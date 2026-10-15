"""Prepare the October 15 archive roster from matching original video frames."""
from pathlib import Path
import sys,json,hashlib,random,time,collections,copy
from datetime import datetime
import cv2,numpy as np,torch
from PIL import Image
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'Intent Monitor'))
from prototype_preprocessing import resize_rgb,build_transform
from intent_engine_v3_1_windows import IntentEngine,EngineConfig
from frame_sampling import FrameSampler,validate_training_timing

def main():
 start=time.monotonic();torch.set_num_threads(4);base=ROOT/'Full Dataset 20261015';imported=base/'imported/data';out=base/'prepared_original_frames';assert not out.exists()
 supplied=json.loads((imported/'annotation/new_dataset.json').read_text());labels=['INTERACTION','PASSTHRU','WAIT'];assert supplied['labels']==labels
 records={}
 for p in (ROOT/'Camera Recordings').glob('*/metadata/*.json'):
  m=json.loads(p.read_text());sid=m['source_recording_id'];assert sid[:8] not in records;v=ROOT/'Camera Recordings'/m['video_relative_path'];assert v.is_file();records[sid[:8]]=(v,m)
 old=json.loads((ROOT/'Prototype Dataset Review/review/reviewed_segments.json').read_text());review=collections.defaultdict(list)
 for s in old['segments']:review[s['trial'].split('_')[-1]].append(s)
 groups=collections.defaultdict(list);all_hashes={}
 for name,a in supplied['database'].items():
  folder=imported/'image_data'/a['annotations']['label']/name;paths=sorted(folder.glob('image_*.jpg'));assert len(paths)==int((folder/'n_frames').read_text())
  assert [p.name for p in paths]==[f'image_{i:05d}.jpg' for i in range(1,len(paths)+1)]
  hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths];content=hashlib.sha256(''.join(hashes).encode()).hexdigest()
  groups[name.split('_')[-1]].append({'name':name,'label':a['annotations']['label'],'folder':str(folder),'frames':len(paths),'content_sha256':content})
 assert len(groups)==159;roster=[];excluded=[]
 for short,items in groups.items():
  video,m=records[short]
  if len(items)>1:
   assert len({x['content_sha256'] for x in items})==1
   selected=[x for x in items if x['label']==m['scenario']];assert len(selected)==1
   chosen=selected[0];excluded += [x for x in items if x is not chosen]
  else:chosen=items[0]
  assert chosen['label']==m['scenario'];assert chosen['frames']==m['frames']==len(m['frame_timestamps_seconds'])
  sid=m['source_recording_id'];frame_start=0;frame_end=m['frames']-1;source='supplied_clip_label_not_new_frame_review'
  if short in review:
   valid=[s for s in review[short] if s['reviewed_label']==chosen['label'] and s['frame_count']>=8];assert valid,(short,chosen['label'])
   segment=max(valid,key=lambda s:s['frame_count']);frame_start=segment['start_frame_inclusive']-1;frame_end=segment['end_frame_inclusive']-1;source='previously_reviewed_same_class_core'
  sampler=FrameSampler(8);indices=[]
  for i in range(frame_start,frame_end+1):
   accepted=sampler.accept(m['frame_timestamps_seconds'][i]);assert not sampler.discontinuity
   if accepted:indices.append(i)
  assert len(indices)>=8
  roster.append(dict(chosen,source_recording_id=sid,video=str(video),participant=m['participant_id'],session_id='session_'+datetime.fromisoformat(m['created']).strftime('%Y%m%d'),indices=indices,timestamps=[m['frame_timestamps_seconds'][i] for i in indices],label_source=source))
 assert collections.Counter(x['label'] for x in roster)=={'INTERACTION':59,'PASSTHRU':60,'WAIT':40}
 # Check image-identical sources cannot leak across groups.
 contentgroups=collections.defaultdict(list)
 for x in roster:contentgroups[x['content_sha256']].append(x['source_recording_id'])
 assert all(len(v)==1 for v in contentgroups.values()), 'Other duplicated recording contents require review'
 rng=random.Random(20261015)
 for label in labels:
  for participant in sorted({x['participant'] for x in roster if x['label']==label}):
   xs=sorted([x for x in roster if x['label']==label and x['participant']==participant],key=lambda x:x['name']);rng.shuffle(xs);nval=3 if label!='WAIT' else 2;ntest=nval
   assert len(xs)>nval+ntest
   for j,x in enumerate(xs):x['subset']='validation' if j<nval else 'testing' if j<nval+ntest else 'training'
 assert collections.Counter(x['subset'] for x in roster)=={'training':127,'validation':16,'testing':16}
 out.mkdir();(out/'frames').mkdir();ann={'labels':labels,'sample_fps':8.,'sample_duration':8,'annotations_confirmed':True,'reviewer':'Supplied clip labels; prior reviewed cores retained where available. Not independently reviewed frame labels for new clips.','split_protocol':'within_session_trial_pilot','database':{}};timing={'sample_fps':8.,'split_protocol':'within_session_trial_pilot','clips':{}}
 mean=[.485,.456,.406];std=[.229,.224,.225];pre=build_transform(150,255,mean,std);live=IntentEngine._build_transform(SimpleNamespace(cfg=EngineConfig(sample_size=150,norm_value=255,mean=mean,std=std)))
 total=0;maxmae=0.;proof=[]
 for number,x in enumerate(roster,1):
  name=x['name'];label=x['label'];dest=out/'frames'/label/name;dest.mkdir(parents=True);wanted={i:j for j,i in enumerate(x['indices'],1)};cap=cv2.VideoCapture(x['video']);i=0;written=0;compare_indices={0,x['frames']//2,x['frames']-1};maes=[]
  while True:
   ok,bgr=cap.read()
   if not ok:break
   if i in compare_indices:
    jpg=cv2.imread(str(Path(x['folder'])/f'image_{i+1:05d}.jpg'));h,w=jpg.shape[:2];mae=float(np.abs(cv2.resize(bgr,(w,h),interpolation=cv2.INTER_AREA).astype(float)-jpg).mean());assert mae<12,(name,i,mae);maes.append(mae)
   if i in wanted:
    rgb=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));p=dest/f'image_{wanted[i]:05d}.png';resize_rgb(rgb,150).save(p)
    with Image.open(p) as im:assert torch.equal(pre(im.convert('RGB')),live(rgb))
    written+=1;total+=1
   i+=1
  cap.release();assert i==x['frames'] and written==len(wanted);(dest/'n_frames').write_text(str(written));maxmae=max(maxmae,max(maes));sid=x['source_recording_id']
  ann['database'][name]={'subset':x['subset'],'annotations':{'label':label},'source_recording_id':sid,'split_group_id':x['session_id'],'label_source':x['label_source']}
  timing['clips'][name]={'source_recording_id':sid,'session_id':x['session_id'],'split_group_id':x['session_id'],'participant_id':x['participant'],'source_video':x['video'],'source_frame_indices':x['indices'],'source_timestamps_seconds':x['timestamps'],'sample_fps':8.,'frames':written}
  proof.append({'name':name,'video_sha256':hashlib.sha256(Path(x['video']).read_bytes()).hexdigest(),'archive_frame_check_mae':maes,'tensor_parity_frames':written})
  if number%20==0:print(f'Prepared and parity-checked {number}/159 recordings ({total} frames)',flush=True)
 (out/'annotations.json').write_text(json.dumps(ann,indent=2));(out/'frames/sampling.json').write_text(json.dumps(timing,indent=2));validate_training_timing(SimpleNamespace(sample_fps=8.,video_path=str(out/'frames'),annotation_path=str(out/'annotations.json'),allow_within_session_pilot=True))
 summary={'status':'passed','source_archive':r'C:\Users\vitha\Downloads\data-20261015T152832Z-1-001.zip','archive_sha256':hashlib.sha256(Path(r'C:\Users\vitha\Downloads\data-20261015T152832Z-1-001.zip').read_bytes()).hexdigest(),'archive_folders':163,'unique_recordings':159,'class_counts':dict(collections.Counter(x['label'] for x in roster)),'split_counts':{s:dict(collections.Counter(x['label'] for x in roster if x['subset']==s)) for s in ['training','validation','testing']},'exported_frames':total,'tensor_parity_max_error':0.,'source_alignment_spot_checks':159*3,'source_alignment_max_mae':maxmae,'excluded_duplicate_copies':excluded,'label_sources':dict(collections.Counter(x['label_source'] for x in roster)),'roster':roster,'source_verification':proof,'elapsed_seconds':time.monotonic()-start,'test_scored':False,'limitations':['Supplied new-clip labels are not newly frame-reviewed; transitions may have other labels','Participant-stratified clip split, not a held-out recording-session evaluation','Some recordings have appeared in earlier prototype experiments; test is not a wholly unseen research benchmark']}
 (out/'rebuild_verification.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k not in ['roster','source_verification','excluded_duplicate_copies']},indent=2),flush=True)
if __name__=='__main__':main()
