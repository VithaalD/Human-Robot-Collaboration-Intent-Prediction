"""Train the deduplicated October 15 full dataset with a stratified 80/10/10 split."""
from pathlib import Path
import json, sys, random, hashlib, time, copy
from collections import Counter, defaultdict
import os
import argparse
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'intentpredictionattempt4-main'));sys.path.insert(0,str(ROOT))
from model import generate_model
from types import SimpleNamespace

def main():
 parser=argparse.ArgumentParser();parser.add_argument("--learning-rate",type=float,default=.001);parser.add_argument("--patience",type=int,default=15);args=parser.parse_args()
 start=time.monotonic(); seed=20261015
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.use_deterministic_algorithms(True)
 assert torch.cuda.is_available(), 'CUDA required for this run'
 device=torch.device('cuda:0')
 source=ROOT/'Full Dataset 20261015/prepared_original_frames'
 ann=json.loads((source/'annotations.json').read_text());timing=json.loads((source/'frames/sampling.json').read_text())
 assert json.loads((source/'rebuild_verification.json').read_text())['status']=='passed'
 labels=ann['labels'];subset={}
 for item in ann['database'].values():
  sid=item['source_recording_id']
  if sid in subset:assert subset[sid]==item['subset']
  subset[sid]=item['subset']
 assert Counter(subset.values())=={'training':127,'validation':16,'testing':16}
 run=ROOT/'Prototype Training Runs'/time.strftime('full_dataset_80_10_10_%Y%m%d_%H%M%S');run.mkdir(parents=True,exist_ok=False)
 (run/'annotations.json').write_text(json.dumps(ann,indent=2))
 from frame_sampling import validate_training_timing
 validate_training_timing(SimpleNamespace(sample_fps=8.,video_path=str(source/'frames'),annotation_path=str(run/'annotations.json'),allow_within_session_pilot=True))
 manifest={'seed':seed,'ratio_requested':[80,10,10],'source_recordings':{s:sum(v==s for v in subset.values()) for s in set(subset.values())},'segments':{s:dict(Counter(x['annotations']['label'] for x in ann['database'].values() if x['subset']==s)) for s in ['training','validation','testing']},'source_assignments':subset,'source_annotation_sha256':hashlib.sha256((source/'annotations.json').read_bytes()).hexdigest(),'test_evaluated':False}
 (run/'split_manifest.json').write_text(json.dumps(manifest,indent=2))
 print('RUN '+str(run),flush=True);print(json.dumps(manifest['segments']),flush=True)
 contract={'sample_duration':8,'sample_fps':8.0,'sample_size':150,'norm_value':255,'mean':[.485,.456,.406],'std':[.229,.224,.225],'color_order':'RGB','n_classes':3,'class_labels':labels,'validation_window':'center 8 consecutive sampled frames','training_window':'random 8 consecutive sampled frames','resize':'bilinear 150x150','preprocessing_version':'original_rgb_bilinear_v1'}
 config={'seed':seed,'max_epochs':100,'early_stopping_patience':args.patience,'minimum_improvement':1e-4,'batch_size':8,'head_lr':args.learning_rate,'weight_decay':.001,'class_balancing':'inverse-frequency replacement sampling, training segments only','backbone':'frozen ImageNet ResNet101, including batch normalization statistics','trainable':'resnet.fc projection, 3-layer LSTM, fc1, fc2','checkpoint_selection':'lowest unweighted validation cross entropy','input_config':contract,'torch_version':str(torch.__version__),'device':torch.cuda.get_device_name(0),'label_policy':'archive clip labels; previous reviewed cores retained for 59 older recordings; 100 supplied clips not independently frame-reviewed'}
 (run/'run_config.json').write_text(json.dumps(config,indent=2))
 model=generate_model(SimpleNamespace(model='cnnlstm',n_classes=3),device)
 for p in model.resnet.parameters():p.requires_grad=False
 for p in model.resnet.fc.parameters():p.requires_grad=True
 backbone=torch.nn.Sequential(*list(model.resnet.children())[:-1]).to(device).eval()
 from prototype_preprocessing import build_transform,PREPROCESSING_VERSION
 pre=build_transform(150,255,contract['mean'],contract['std'])
 cache={}; targets={}; frames_hash=hashlib.sha256()
 # Test images are deliberately not loaded or encoded.
 for name,item in ann['database'].items():
  if item['subset']=='testing':continue
  folder=source/'frames'/item['annotations']['label']/name
  paths=sorted(folder.glob('image_*.png'));assert len(paths)>=8
  feats=[]
  for offset in range(0,len(paths),32):
   images=[]
   for p in paths[offset:offset+32]:
    frames_hash.update(p.read_bytes())
    with Image.open(p) as im:images.append(pre(im.convert('RGB')))
   with torch.no_grad():feats.append(backbone(torch.stack(images).to(device)).flatten(1).cpu())
  cache[name]=torch.cat(feats);targets[name]=labels.index(item['annotations']['label'])
 print('Cached frozen image features for '+str(len(cache))+' train/validation segments.',flush=True)
 def forward(x):
  b,t,d=x.shape; z=model.resnet.fc(x.reshape(b*t,d)).reshape(b,t,300).transpose(0,1)
  z,_=model.lstm(z);return model.fc2(torch.relu(model.fc1(z[-1])))
 train=[n for n,x in ann['database'].items() if x['subset']=='training'];val=[n for n,x in ann['database'].items() if x['subset']=='validation']
 class_counts=Counter(targets[n] for n in train);weights=torch.tensor([1/class_counts[targets[n]] for n in train],dtype=torch.double)
 def batch(names,random_crop):
  clips=[]
  for n in names:
   f=cache[n];lo=random.randrange(len(f)-7) if random_crop else (len(f)-8)//2;clips.append(f[lo:lo+8])
  return torch.stack(clips).to(device),torch.tensor([targets[n] for n in names],device=device)
 def evaluate():
  model.eval();cm=torch.zeros(3,3,dtype=torch.long);loss=0.;predictions=[]
  with torch.no_grad():
   for offset in range(0,len(val),8):
    names=val[offset:offset+8];x,y=batch(names,False);out=forward(x);assert torch.isfinite(out).all()
    loss+=torch.nn.functional.cross_entropy(out,y,reduction='sum').item()
    for name,a,b in zip(names,y.tolist(),out.argmax(1).tolist()):cm[a,b]+=1;predictions.append({'segment':name,'true':labels[a],'predicted':labels[b]})
  per={}
  for c,lab in enumerate(labels):
   tp=int(cm[c,c]);support=int(cm[c].sum());precision=tp/max(int(cm[:,c].sum()),1);recall=tp/max(support,1)
   per[lab]={'precision':precision,'recall':recall,'f1':2*precision*recall/max(precision+recall,1e-12),'support':support}
  return {'loss':loss/len(val),'accuracy':float(cm.diag().sum()/cm.sum()),'macro_f1':sum(v['f1'] for v in per.values())/3,'per_class':per,'confusion_matrix':cm.tolist(),'predictions':predictions}
 optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.learning_rate,weight_decay=.001)
 scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=5,factor=.5)
 baseline=evaluate();history=[];bestloss=float('inf');stale=0
 for epoch in range(1,101):
  model.train();model.resnet.eval();total_loss=0.;correct=0
  indices=torch.multinomial(weights,len(train),replacement=True).tolist()
  for offset in range(0,len(indices),8):
   names=[train[i] for i in indices[offset:offset+8]];x,y=batch(names,True);optimizer.zero_grad(set_to_none=True);out=forward(x);loss=torch.nn.functional.cross_entropy(out,y)
   assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.);assert torch.isfinite(norm);optimizer.step()
   total_loss+=loss.item()*len(names);correct+=(out.argmax(1)==y).sum().item()
  metrics=evaluate();scheduler.step(metrics['loss']);history.append({'epoch':epoch,'train_loss':total_loss/len(train),'sampled_train_accuracy':correct/len(train),'validation':metrics,'lr':optimizer.param_groups[0]['lr']})
  if metrics['loss']<bestloss-1e-4:
   bestloss=metrics['loss'];stale=0
   torch.save({'epoch':epoch,'state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'input_config':contract,'training_config':config,'validation':metrics,'purpose':'small_prototype_not_test_evaluated'},run/'best_checkpoint.pth')
  else:stale+=1
  (run/'history.json').write_text(json.dumps(history,indent=2))
  print(f'Epoch {epoch}: train loss {total_loss/len(train):.4f}, validation loss {metrics["loss"]:.4f}, accuracy {metrics["accuracy"]:.3f}, macro F1 {metrics["macro_f1"]:.3f}',flush=True)
  if stale>=args.patience:break
 saved=torch.load(run/'best_checkpoint.pth',map_location=device,weights_only=True);model.load_state_dict(saved['state_dict']);model.eval();final=evaluate()
 assert abs(final['loss']-saved['validation']['loss'])<1e-6
 # Verify cached-feature training predicts identically through the original full model.
 name=val[0];folder=source/'frames'/ann['database'][name]['annotations']['label']/name;paths=sorted(folder.glob('image_*.png'));lo=(len(paths)-8)//2;images=[]
 for p in paths[lo:lo+8]:
  with Image.open(p) as im:images.append(pre(im.convert('RGB')))
 with torch.no_grad():
  full=model(torch.stack(images).unsqueeze(0).to(device));cached=forward(batch([name],False)[0]);error=float((full-cached).abs().max())
 assert torch.allclose(full,cached,atol=1e-4,rtol=1e-4),error
 report={'status':'completed','run_directory':str(run),'checkpoint':str(run/'best_checkpoint.pth'),'epochs_completed':len(history),'best_epoch':saved['epoch'],'baseline_validation':baseline,'best_validation':final,'split':manifest,'input_config':contract,'dataset':str(source),'full_model_vs_cached_max_error':error,'train_validation_frames_sha256':frames_hash.hexdigest(),'test_evaluated':False,'elapsed_seconds':time.monotonic()-start,'limitations':['159 unique recordings from supplied archive; one fewer INTERACTION than requested','Both participants and recording sessions occur across subsets','New clip labels supplied by partner, not independently frame-reviewed','Four exact conflicting PASSTHRU copies excluded; corrected WAIT copies kept','Some source recordings were used in previous prototype experiments','Validation used for checkpoint selection; test not scored in this run','Shared original-frame preprocessing replaces compressed extracted JPEGs; use checkpoint input_config']}
 (run/'training_report.json').write_text(json.dumps(report,indent=2));(ROOT/'Prototype Training Runs/latest_full_dataset_training.json').write_text(json.dumps(report,indent=2))
 print('COMPLETED '+str(run),flush=True)
if __name__=='__main__':main()
