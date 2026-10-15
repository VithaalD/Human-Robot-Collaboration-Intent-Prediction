"""Small model smoke test; never evaluates the reserved test split."""
from pathlib import Path
import sys,json,random,time,math,hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'intentpredictionattempt4-main'));sys.path.insert(0,str(ROOT))
from opts import parse_opts
from main import get_loaders,build_input_config,resume_model
from model import generate_model
from train import train_epoch
from validation import val_epoch
def main():
 started=time.monotonic()
 run=ROOT/'Prototype Training Runs'/time.strftime('smoke_%Y%m%d_%H%M%S');run.mkdir(parents=True,exist_ok=False)
 cfg=json.loads((ROOT/'Prototype 8 Frame Pilot/pilot_config.json').read_text())
 sys.argv=['smoke']
 for k,v in cfg.items():
  if isinstance(v,bool):
   if v:sys.argv.append('--'+k)
  else:sys.argv.extend(['--'+k,str(v)])
 opt=parse_opts();random.seed(7);np.random.seed(7);torch.manual_seed(7)
 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
 torch.set_num_threads(4)
 train,val=get_loaders(opt)
 def balanced(ds,per_class):
  indices=[]
  for c in range(3):
   chosen=[i for i,x in enumerate(ds.data) if x['label']==c][:per_class]
   assert len(chosen)==per_class
   indices+=chosen
  return indices
 ti=balanced(train.dataset,2);vi=balanced(val.dataset,1)
 tl=torch.utils.data.DataLoader(torch.utils.data.Subset(train.dataset,ti),batch_size=1,shuffle=True,num_workers=0)
 vl=torch.utils.data.DataLoader(torch.utils.data.Subset(val.dataset,vi),batch_size=1,shuffle=False,num_workers=0)
 opt.input_config=build_input_config(opt,train.dataset)
 print('Initializing pretrained ResNet101 + LSTM on '+str(device),flush=True)
 model=generate_model(opt,device)
 optimizer=torch.optim.Adam(model.parameters(),lr=opt.lr_rate,weight_decay=opt.weight_decay)
 criterion=torch.nn.CrossEntropyLoss()
 before=model.fc2.weight.detach().clone()
 counts={'forwards':0,'gradient_checks':0}
 def check_out(module,inputs,outputs):
  assert outputs.shape==(inputs[0].shape[0],3) and torch.isfinite(outputs).all()
  counts['forwards']+=1
 def check_grad(g):
  assert torch.isfinite(g).all()
  counts['gradient_checks']+=1
  return g
 forward_hook=model.register_forward_hook(check_out)
 hooks=[p.register_hook(check_grad) for p in model.parameters() if p.requires_grad]
 results=[]
 for epoch in [1,2]:
  loss,acc=train_epoch(model,tl,criterion,optimizer,epoch,1,device)
  vloss,vacc=val_epoch(model,vl,criterion,device)
  assert math.isfinite(loss) and math.isfinite(vloss)
  results.append({'epoch':epoch,'training_loss':loss,'validation_loss':vloss})
  print(json.dumps(results[-1]),flush=True)
 delta=float((model.fc2.weight.detach()-before).abs().max())
 assert delta>0 and counts['gradient_checks']>0
 model.eval();sample=next(iter(vl))[0].to(device)
 with torch.no_grad():expected=model(sample).detach().cpu()
 checkpoint=run/'smoke_checkpoint.pth'
 torch.save({'epoch':2,'state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'input_config':opt.input_config,'purpose':'smoke_test_only_not_validated_model'},checkpoint)
 forward_hook.remove()
 for h in hooks:h.remove()
 del model,optimizer
 if device.type=='cuda':torch.cuda.empty_cache()
 restored=generate_model(opt,device);restored_optimizer=torch.optim.Adam(restored.parameters(),lr=opt.lr_rate,weight_decay=opt.weight_decay)
 opt.resume_path=str(checkpoint);next_epoch=resume_model(opt,restored,restored_optimizer);assert next_epoch==3
 restored.eval()
 with torch.no_grad():actual=restored(sample).detach().cpu()
 error=float((expected-actual).abs().max());assert torch.allclose(expected,actual,atol=1e-6,rtol=1e-5)
 opt.input_config=dict(opt.input_config,sample_duration=16)
 try:resume_model(opt,restored,restored_optimizer)
 except ValueError as e:
  assert 'sample_duration' in str(e);mismatch_rejected=True
 else:raise AssertionError('Mismatched checkpoint input accepted')
 report={'status':'passed','purpose':'infrastructure smoke test, not an accuracy evaluation','device':str(device),'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,
 'model':'ImageNet-pretrained ResNet101 + newly initialized LSTM/classifier','sample_duration':8,'sample_fps':8,
 'training_segments':6,'validation_segments':3,'epochs':2,'optimizer_steps':12,'reserved_test_evaluated':False,
 'losses':results,'classifier_max_weight_change':delta,'checks':counts,'checkpoint':str(checkpoint),
 'reload_max_prediction_difference':error,'resume_next_epoch':next_epoch,'16_frame_mismatch_rejected':mismatch_rejected,
 'elapsed_seconds':time.monotonic()-started,'peak_gpu_allocated_mb':torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None,
 'training_trials':[train.dataset.data[i]['video_id'] for i in ti],'validation_trials':[val.dataset.data[i]['video_id'] for i in vi]}
 (run/'smoke_report.json').write_text(json.dumps(report,indent=2))
 (ROOT/'Prototype 8 Frame Pilot/latest_smoke_test.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
