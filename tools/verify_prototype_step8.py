from pathlib import Path
import sys,json,time
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'Intent Monitor'));sys.path.insert(0,str(ROOT))
from intent_monitor import config,prototype_profile,make_inference_engine
from clip_recorder import ClipRecorder
from camera_devices import discover_cameras
from intent_engine_v3_1_windows import IntentEngine

def main():
 torch.set_num_threads(4)
 weights,video,frames=prototype_profile();cfg=config(weights,sample_duration=frames)
 assert cfg.sample_duration==8 and cfg.sample_fps==8 and cfg.norm_value==255 and cfg.mean==[.485,.456,.406]
 try:config(weights,sample_duration=16)
 except ValueError:mismatch=True
 else:raise AssertionError('Wrong frame duration accepted')
 report=json.loads((ROOT/'Prototype Training Runs/latest_step6.json').read_text());run=Path(report['run_directory'])
 engine=IntentEngine(cfg);observed=[];times=[];stamp=0.
 for item in report['best_validation']['predictions']:
  folder=ROOT/'Prototype 8 Frame Pilot/frames'/item['true']/item['segment'];paths=sorted(folder.glob('image_*.jpg'));start=(len(paths)-8)//2
  stamp+=10
  for p in paths[start:start+8]:engine.push_frame(cv2.imread(str(p)),stamp);stamp+=.125
  torch.cuda.synchronize();begin=time.perf_counter();engine._inference_step();torch.cuda.synchronize();times.append((time.perf_counter()-begin)*1000)
  state=engine.get_state();assert state['label']==item['predicted'];assert np.isfinite(state['probs']).all() and abs(float(state['probs'].sum())-1)<1e-5
  observed.append(dict(segment=item['segment'],expected=item['predicted'],actual=state['label'],flag=state['flag']))
 cfg=config(weights,video=video,sample_duration=8)
 replay=make_inference_engine(cfg,ClipRecorder(ROOT/'Camera Recordings'));states=[];began=time.perf_counter();first=None
 try:
  replay.start()
  while time.perf_counter()-began<10:
   state=replay.get_state()
   if state['probs'].sum()>0:
    if first is None:first=time.perf_counter()-began
    states.append({'label':state['label'],'confidence':state['confidence'],'flag':state['flag']})
   time.sleep(.1)
  assert states and replay.preview.frame is not None
 finally:replay.stop()
 assert not replay._cam_thread.is_alive() and not replay._infer_thread.is_alive()
 result={'status':'passed','checkpoint':str(weights),'input_config_loaded':True,'wrong_frame_count_rejected':mismatch,'validation_prediction_parity':observed,'inference_ms':{'first':times[0],'median_excluding_first':float(np.median(times[1:])),'max':max(times)},'recorded_video_replay':{'video':video,'seconds':10,'first_prediction_seconds':first,'observed_labels':sorted(set(x['label'] for x in states)),'preview_frames_received':True,'shutdown_threads_joined':True},'cameras_detected':[d.name for d in discover_cameras()],'lab_realsense_available':False,'live_workspace_accuracy_tested':False,'note':'Replay and per-window timing are engineering checks, not new accuracy scores. RealSense lab scene not connected.'}
 (run/'step8_verification.json').write_text(json.dumps(result,indent=2));(ROOT/'Prototype Training Runs/latest_step8.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
