from pathlib import Path
import sys,json,hashlib
import cv2,numpy as np,torch
from PIL import Image
root=Path.cwd();sys.path.insert(0,str(root/'Intent Monitor'))
from intent_monitor import config,prototype_profile
from intent_engine_v3_1_windows import IntentEngine
w,_,_=prototype_profile();torch.set_num_threads(4);engine=IntentEngine(config(w,sample_duration=8));train=json.loads((root/'Prototype Training Runs/latest_step6.json').read_text());run=Path(train['run_directory']);timing=json.loads((root/'Prototype 8 Frame Pilot/frames/sampling.json').read_text())['clips'];labels=engine.class_labels
variants=['extracted','original','linear427','area427','cubic427','pil_bilinear427','pil_lanczos427','linear427_jpeg95','area427_jpeg95','cubic427_jpeg95']
rows=[];errors={v:[] for v in variants};alignment=[];dims=set();seek_equal=True
for item in train['best_validation']['predictions']:
 name=item['segment'];c=timing[name];folder=root/'Prototype 8 Frame Pilot/frames'/item['true']/name;paths=sorted(folder.glob('image_*.jpg'));start=(len(paths)-8)//2;ix=c['source_frame_indices'][start:start+8]
 cap=cv2.VideoCapture(c['source_video']);raw=[]
 while True:
  ok,f=cap.read()
  if not ok:break
  raw.append(f)
 cap.release();cap=cv2.VideoCapture(c['source_video'])
 tensors={v:[] for v in variants}
 for idx,p in zip(ix,paths[start:start+8]):
  cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,seek=cap.read();assert ok;seek_equal=seek_equal and np.array_equal(raw[idx],seek)
  with Image.open(p) as im: ext=np.array(im.convert('RGB'));width,height=im.size
  original=cv2.cvtColor(raw[idx],cv2.COLOR_BGR2RGB);dims.add((original.shape[1],original.shape[0],width,height))
  versions={'extracted':ext,'original':original}
  for kind,method in [('linear',cv2.INTER_LINEAR),('area',cv2.INTER_AREA),('cubic',cv2.INTER_CUBIC)]:
   resized=cv2.resize(original,(width,height),interpolation=method);versions[kind+'427']=resized
   ok,enc=cv2.imencode('.jpg',cv2.cvtColor(resized,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,95]);versions[kind+'427_jpeg95']=cv2.cvtColor(cv2.imdecode(enc,1),cv2.COLOR_BGR2RGB)
  versions['pil_bilinear427']=np.array(Image.fromarray(original).resize((width,height),Image.Resampling.BILINEAR));versions['pil_lanczos427']=np.array(Image.fromarray(original).resize((width,height),Image.Resampling.LANCZOS))
  for v,im in versions.items():
   tensors[v].append(engine._transform(Image.fromarray(im)))
   if im.shape==ext.shape:errors[v].append(float(np.abs(im.astype(float)-ext).mean()))
  offset_errors={}
  for delta in [-2,-1,0,1,2]:
   if 0<=idx+delta<len(raw):
    rgb=cv2.cvtColor(raw[idx+delta],cv2.COLOR_BGR2RGB);small=cv2.resize(rgb,(width,height),interpolation=cv2.INTER_LINEAR);offset_errors[delta]=float(np.abs(small.astype(float)-ext).mean())
  alignment.append({'frame':idx,'segment':name,'best_offset':min(offset_errors,key=offset_errors.get),'offset_mae':offset_errors})
 cap.release();preds={}
 with torch.inference_mode():
  for v in variants:
   logits=engine._model(torch.stack(tensors[v]).unsqueeze(0).to(engine.device));prob=logits.softmax(1)[0].cpu().numpy();preds[v]={'label':labels[int(prob.argmax())],'probabilities':dict(zip(labels,map(float,prob))),'input_tensor_mae_vs_extracted':float((torch.stack(tensors[v])-torch.stack(tensors['extracted'])).abs().mean())}
 rows.append({'segment':name,'true':item['true'],'source_indices':ix,'predictions':preds})
summary={v:{'correct':sum(r['predictions'][v]['label']==r['true'] for r in rows),'prediction_agreement_with_extracted':sum(r['predictions'][v]['label']==r['predictions']['extracted']['label'] for r in rows),'pixel_mae_at_extracted_size':float(np.mean(errors[v])) if errors[v] else None} for v in variants}
result={'validation_only':True,'test_set_used':False,'checkpoint_modified':False,'dimensions':list(dims),'random_seek_matches_sequential_decode':seek_equal,'offset_match_counts':{str(i):sum(a['best_offset']==i for a in alignment) for i in range(-2,3)},'summary':summary,'windows':rows,'alignment':alignment}
out=run/'frame_path_investigation';out.mkdir(exist_ok=False);(out/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['windows','alignment']},indent=2))
