from pathlib import Path
import sys,json,io
import cv2,numpy as np,torch
from PIL import Image
root=Path.cwd();sys.path.insert(0,str(root/'Intent Monitor'))
from intent_monitor import config,prototype_profile
from intent_engine_v3_1_windows import IntentEngine
torch.set_num_threads(4);w,_,_=prototype_profile();e=IntentEngine(config(w,sample_duration=8));r=json.loads((root/'Prototype Training Runs/latest_step6.json').read_text());run=Path(r['run_directory']);timing=json.loads((root/'Prototype 8 Frame Pilot/frames/sampling.json').read_text())['clips'];variants=['area_matched_jpeg','linear_matched_jpeg','pil_bilinear_matched_jpeg','pil_lanczos_matched_jpeg'];rows=[];errors={v:[] for v in variants};tables=set()
for item in r['best_validation']['predictions']:
 name=item['segment'];c=timing[name];paths=sorted((root/'Prototype 8 Frame Pilot/frames'/item['true']/name).glob('image_*.jpg'));start=(len(paths)-8)//2;cap=cv2.VideoCapture(c['source_video']);tensors={v:[] for v in variants}
 for idx,p in zip(c['source_frame_indices'][start:start+8],paths[start:start+8]):
  cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,raw=cap.read();assert ok;raw=cv2.cvtColor(raw,cv2.COLOR_BGR2RGB)
  with Image.open(p) as im:ext=np.array(im.convert('RGB'));size=im.size;qt=im.quantization;tables.add(json.dumps(qt,sort_keys=True))
  smalls={'area_matched_jpeg':Image.fromarray(cv2.resize(raw,size,interpolation=cv2.INTER_AREA)),'linear_matched_jpeg':Image.fromarray(cv2.resize(raw,size,interpolation=cv2.INTER_LINEAR)),'pil_bilinear_matched_jpeg':Image.fromarray(raw).resize(size,Image.Resampling.BILINEAR),'pil_lanczos_matched_jpeg':Image.fromarray(raw).resize(size,Image.Resampling.LANCZOS)}
  for v,im in smalls.items():
   buf=io.BytesIO();im.save(buf,format='JPEG',qtables=qt,subsampling=2);buf.seek(0)
   with Image.open(buf) as decoded:decoded=decoded.convert('RGB');errors[v].append(float(np.abs(np.array(decoded).astype(float)-ext).mean()));tensors[v].append(e._transform(decoded))
 cap.release();pred={}
 with torch.inference_mode():
  for v in variants:
   probs=e._model(torch.stack(tensors[v]).unsqueeze(0).to(e.device)).softmax(1)[0].cpu().tolist();pred[v]={'label':e.class_labels[int(np.argmax(probs))],'probabilities':dict(zip(e.class_labels,probs))}
 rows.append({'segment':name,'true':item['true'],'extracted_prediction':item['predicted'],'predictions':pred})
summary={v:{'correct':sum(x['true']==x['predictions'][v]['label'] for x in rows),'agreement_with_extracted':sum(x['extracted_prediction']==x['predictions'][v]['label'] for x in rows),'pixel_mae':float(np.mean(errors[v]))} for v in variants}
out={'method':'427x240 resize then JPEG encode/decode with quantization tables read from corresponding extracted JPEG and 4:2:0 subsampling; fixed model and validation indices','distinct_extracted_quantization_tables':len(tables),'summary':summary,'windows':rows}
(run/'frame_path_investigation/compression_results.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='windows'},indent=2))
