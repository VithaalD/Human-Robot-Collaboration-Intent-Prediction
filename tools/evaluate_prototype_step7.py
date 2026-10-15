"""Evaluate the frozen step-6 checkpoint once on the predefined reserved test split."""
from pathlib import Path
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys,json,hashlib
from collections import Counter,defaultdict
from types import SimpleNamespace
import torch
from PIL import Image
from torchvision import transforms
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'intentpredictionattempt4-main'));sys.path.insert(0,str(ROOT))
from model import generate_model
from frame_sampling import validate_training_timing

def main():
 training=json.loads((ROOT/'Prototype Training Runs/latest_step6.json').read_text());run=Path(training['run_directory']);output=run/'test_report.json'
 if output.exists():
  print(output.read_text());return
 checkpoint=Path(training['checkpoint']);digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
 assert digest==(run/'checkpoint_sha256.txt').read_text().split()[0]
 ann=json.loads((run/'annotations.json').read_text());manifest=json.loads((run/'split_manifest.json').read_text());source=ROOT/'Prototype 8 Frame Pilot/frames'
 validate_training_timing(SimpleNamespace(sample_fps=8.,video_path=str(source),annotation_path=str(run/'annotations.json'),allow_within_session_pilot=True))
 selected=[(n,x) for n,x in ann['database'].items() if x['subset']=='testing'];assert len(selected)==10
 groups={x['source_recording_id'] for _,x in selected};assert len(groups)==6
 assert all(manifest['source_assignments'][g]=='testing' for g in groups)
 torch.set_num_threads(4);torch.manual_seed(20261010);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
 saved=torch.load(checkpoint,map_location=device,weights_only=True);cfg=saved['input_config'];labels=cfg['class_labels'];assert saved['epoch']==training['best_epoch'] and labels==ann['labels']
 assert cfg['sample_duration']==8 and cfg['sample_fps']==8 and cfg['norm_value']==255
 model=generate_model(SimpleNamespace(model='cnnlstm',n_classes=3),device);model.load_state_dict(saved['state_dict'],strict=True);model.eval()
 for p in model.parameters():p.requires_grad=False
 transform=transforms.Compose([transforms.Resize((cfg['sample_size'],cfg['sample_size'])),transforms.ToTensor(),transforms.Normalize(cfg['mean'],cfg['std'])])
 confusion=torch.zeros(3,3,dtype=torch.long);predictions=[];total_loss=0.;source_results=defaultdict(list)
 with torch.inference_mode():
  for name,item in selected:
   folder=source/item['annotations']['label']/name;paths=sorted(folder.glob('image_*.jpg'));assert len(paths)>=8
   start=(len(paths)-8)//2;images=[]
   for p in paths[start:start+8]:
    with Image.open(p) as im:images.append(transform(im.convert('RGB')))
   logits=model(torch.stack(images).unsqueeze(0).to(device));assert torch.isfinite(logits).all()
   true=labels.index(item['annotations']['label']);pred=int(logits.argmax(1));probs=logits.softmax(1)[0].cpu().tolist()
   total_loss+=torch.nn.functional.cross_entropy(logits,torch.tensor([true],device=device)).item();confusion[true,pred]+=1
   source_results[item['source_recording_id']].append(true==pred)
   predictions.append(dict(segment=name,source_recording_id=item['source_recording_id'],true=labels[true],predicted=labels[pred],probabilities=dict(zip(labels,probs)),export_frame_indices_1based=list(range(start+1,start+9))))
   print(name+' '+labels[true]+' -> '+labels[pred],flush=True)
 metrics={}
 for c,label in enumerate(labels):
  tp=int(confusion[c,c]);support=int(confusion[c].sum());precision=tp/max(int(confusion[:,c].sum()),1);recall=tp/max(support,1)
  metrics[label]=dict(precision=precision,recall=recall,f1=2*precision*recall/max(precision+recall,1e-12),support=support,correct=tp)
 correct=int(confusion.diag().sum());assert int(confusion.sum())==10
 assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
 report=dict(status='completed',evaluation_date='2026-10-10',checkpoint=str(checkpoint),checkpoint_sha256=digest,checkpoint_epoch=saved['epoch'],test_recordings=6,test_segments=10,correct_segments=correct,accuracy=correct/10,loss=total_loss/10,macro_f1=sum(v['f1'] for v in metrics.values())/3,balanced_accuracy=sum(v['recall'] for v in metrics.values())/3,per_class=metrics,confusion_matrix=confusion.tolist(),matrix_order=labels,majority_wait_baseline_accuracy=.6,predictions=predictions,source_recordings_all_segments_correct=sum(all(v) for v in source_results.values()),input_config=cfg,selection='Fixed best validation checkpoint and one center eight-frame window per reserved segment; no test-based tuning',checkpoint_unchanged=True,limitations=['Only 10 segments from 6 source recordings; segments within a recording are correlated','All test footage is from the same September 29 session and known participants','Only two test segments each for INTERACTION and PASSTHRU','AI-reviewed labels, not independently confirmed human ground truth','Not a live latency, robot safety, new-person or new-session test'])
 output.write_text(json.dumps(report,indent=2));(ROOT/'Prototype Training Runs/latest_step7.json').write_text(json.dumps(report,indent=2))
 print(json.dumps({k:v for k,v in report.items() if k not in ['predictions','input_config']},indent=2),flush=True)
if __name__=='__main__':main()
