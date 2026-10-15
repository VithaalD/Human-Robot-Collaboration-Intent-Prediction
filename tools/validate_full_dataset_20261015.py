"""Validate the full-dataset checkpoint and original-video inference parity."""
from pathlib import Path
import sys,json,time,hashlib
from types import SimpleNamespace
import cv2,numpy as np,torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'Intent Monitor'));sys.path.insert(0,str(ROOT))
from intent_monitor import config,make_inference_engine
from intent_engine_v3_1_windows import IntentEngine
from clip_recorder import ClipRecorder

def main():
 torch.set_num_threads(4);r=json.loads((ROOT/'Prototype Training Runs/latest_full_dataset_training.json').read_text());run=Path(r['run_directory']);source=Path(r['dataset']);weights=Path(r['checkpoint']);sha=hashlib.sha256(weights.read_bytes()).hexdigest()
 engine=IntentEngine(config(weights,sample_duration=8));timing=json.loads((source/'frames/sampling.json').read_text())['clips'];rows=[];stamp=0
 for item in r['best_validation']['predictions']:
  name=item['segment'];c=timing[name];paths=sorted((source/'frames'/item['true']/name).glob('image_*.png'));start=(len(paths)-8)//2;indices=c['source_frame_indices'][start:start+8];prepared=[]
  for p in paths[start:start+8]:
   with Image.open(p) as im:prepared.append(engine._transform(im.convert('RGB')))
  cap=cv2.VideoCapture(c['source_video']);raw=[];i=0
  while i<=max(indices):
   ok,frame=cap.read();assert ok
   if i in indices:raw.append(frame)
   i+=1
  cap.release();assert len(raw)==8
  direct=[engine._transform(Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))) for f in raw];assert torch.equal(torch.stack(prepared),torch.stack(direct))
  with torch.inference_mode():expected=engine._model(torch.stack(prepared).unsqueeze(0).to(engine.device)).softmax(1)[0].cpu().numpy()
  stamp+=10
  for f in raw:engine.push_frame(f,stamp);stamp+=.125
  engine._inference_step();state=engine.get_state();err=float(np.abs(expected-state['probs']).max());assert err<1e-6
  assert state['label']==item['predicted']
  rows.append({'segment':name,'true':item['true'],'predicted':state['label'],'tensor_max_error':0.,'probability_max_error':err})
 replay_video=next(x['source_video'] for x in timing.values())
 replay=make_inference_engine(config(weights,video=replay_video,sample_duration=8),ClipRecorder(ROOT/'Camera Recordings'));states=[];start=time.perf_counter()
 try:
  replay.start()
  while time.perf_counter()-start<10:
   s=replay.get_state()
   if s['probs'].sum()>0:states.append({'elapsed':time.perf_counter()-start,'label':s['label'],'confidence':s['confidence']})
   time.sleep(.1)
  assert states and replay.preview.frame is not None
 finally:replay.stop()
 assert not replay._cam_thread.is_alive() and not replay._infer_thread.is_alive()
 assert hashlib.sha256(weights.read_bytes()).hexdigest()==sha
 result={'status':'passed','validation_windows':len(rows),'correct':sum(x['true']==x['predicted'] for x in rows),'all_prepared_and_original_tensors_identical':True,'all_monitor_predictions_match_training_validation':True,'max_probability_error':max(x['probability_max_error'] for x in rows),'windows':rows,'checkpoint_sha256':sha,'test_predictions_computed':False,'continuous_replay':{'video':replay_video,'seconds':10,'first_prediction_seconds':states[0]['elapsed'],'labels_seen':sorted(set(s['label'] for s in states)),'note':'Engineering check, not a continuous labeled accuracy evaluation.'},'live_lab_validation_completed':False}
 (run/'live_input_validation.json').write_text(json.dumps(result,indent=2));(run/'checkpoint_sha256.txt').write_text(sha+'  best_checkpoint.pth\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
