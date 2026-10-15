"""Rebuild reviewed ranges from original video with exact live-input tensor parity."""
from pathlib import Path
import sys,json,hashlib,copy,time
from collections import defaultdict,Counter
import cv2,torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'Intent Monitor'))
from prototype_preprocessing import resize_rgb,build_transform,PREPROCESSING_VERSION
from intent_engine_v3_1_windows import EngineConfig,IntentEngine
from frame_sampling import validate_training_timing
from types import SimpleNamespace

def main():
 started=time.monotonic();torch.set_num_threads(4)
 old=json.loads((ROOT/'Prototype Training Runs/latest_step6.json').read_text());run=Path(old['run_directory']);out=ROOT/'Prototype Original Frame Pilot'
 assert not out.exists(), 'Keep existing exports; choose a different output name.'
 ann=json.loads((run/'annotations.json').read_text());timing=json.loads((ROOT/'Prototype 8 Frame Pilot/frames/sampling.json').read_text());original_ann=copy.deepcopy(ann)
 recordings={}
 for p in (ROOT/'Camera Recordings').glob('*/metadata/*.json'):
  d=json.loads(p.read_text());recordings[d['source_recording_id']]=(ROOT/'Camera Recordings'/d['video_relative_path'],d)
 grouped=defaultdict(list)
 for n,a in ann['database'].items():grouped[a['source_recording_id']].append(n)
 assert len(grouped)==59 and len(ann['database'])==93
 out.mkdir();(out/'frames').mkdir()
 contract=old['input_config'];pre=build_transform(150,255,contract['mean'],contract['std'])
 # Obtain the transform through the actual inference engine API, without loading weights.
 proxy=SimpleNamespace(cfg=EngineConfig(sample_size=150,norm_value=255,mean=contract['mean'],std=contract['std']))
 live=IntentEngine._build_transform(proxy)
 records=[];frame_count=0;maxerr=0.;inputhash=hashlib.sha256()
 for sid,names in grouped.items():
  video,meta=recordings[sid];assert video.is_file();sha=hashlib.sha256(video.read_bytes()).hexdigest();jobs=defaultdict(list)
  for name in names:
   clip=timing['clips'][name];indices=clip['source_frame_indices'];assert len(indices)>=8 and len(set(indices))==len(indices)
   assert clip['session_id']==meta['session_id'];clip['source_video']=str(video)
   assert all(abs(meta['frame_timestamps_seconds'][i]-t)<1e-8 for i,t in zip(indices,clip['source_timestamps_seconds']))
   folder=out/'frames'/ann['database'][name]['annotations']['label']/name;folder.mkdir(parents=True)
   (folder/'n_frames').write_text(str(len(indices)))
   for j,i in enumerate(indices,1):jobs[i].append(folder/f'image_{j:05d}.png')
  cap=cv2.VideoCapture(str(video));i=0;count=0
  while True:
   ok,bgr=cap.read()
   if not ok:break
   if i in jobs:
    rgb=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));small=resize_rgb(rgb,150);expected=live(rgb)
    for p in jobs[i]:
     small.save(p,format='PNG')
     with Image.open(p) as im:actual=pre(im.convert('RGB'))
     error=float((expected-actual).abs().max());assert torch.equal(expected,actual),(p,error)
     maxerr=max(maxerr,error);frame_count+=1;inputhash.update(p.read_bytes());count+=1
   i+=1
  cap.release();assert count==sum(len(x) for x in jobs.values());assert i==meta['frames']
  records.append({'source_recording_id':sid,'video':str(video),'sha256':sha,'frames_decoded':i,'frames_exported':count})
 assert frame_count==2413 and ann==original_ann
 timing['preprocessing']={'version':PREPROCESSING_VERSION,'storage':'lossless RGB PNG after shared bilinear 150x150 resize','original_frame_pixel_parity_verified':True}
 (out/'frames/sampling.json').write_text(json.dumps(timing,indent=2));(out/'annotations.json').write_text(json.dumps(ann,indent=2));(out/'split_manifest.json').write_bytes((run/'split_manifest.json').read_bytes())
 validate_training_timing(SimpleNamespace(sample_fps=8.,video_path=str(out/'frames'),annotation_path=str(out/'annotations.json'),allow_within_session_pilot=True))
 result={'status':'passed','dataset':str(out),'source_run':str(run),'source_recordings':59,'segments':93,'exported_frames':frame_count,'pixel_tensor_parity_checked_frames':frame_count,'max_tensor_absolute_difference':maxerr,'annotations_and_split_unchanged':True,'preprocessing_version':PREPROCESSING_VERSION,'prepared_files_sha256':inputhash.hexdigest(),'source_video_hashes':records,'elapsed_seconds':time.monotonic()-started,'test_frames_rebuilt_for_preprocessing_check_only':True,'test_predictions_computed':False}
 (out/'rebuild_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='source_video_hashes'},indent=2),flush=True)
if __name__=='__main__':main()
