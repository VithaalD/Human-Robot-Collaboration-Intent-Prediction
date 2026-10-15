"""Check the pilot loaders, or explicitly start training with --train."""
from pathlib import Path
import argparse,json,sys,os,datetime,collections
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--dataset',type=Path,default=ROOT/'Prototype 8 Frame Pilot')
 p.add_argument('--train',action='store_true')
 p.add_argument('--epochs',type=int,default=10)
 args=p.parse_args()
 if args.epochs<1:p.error('epochs must be positive')
 cfg=json.loads((args.dataset/'pilot_config.json').read_text())
 sys.path.insert(0,str(ROOT/'intentpredictionattempt4-main'));sys.path.insert(0,str(ROOT))
 from opts import parse_opts
 from main import get_loaders,build_input_config,main_worker
 sys.argv=['pilot']
 for k,v in cfg.items():
  if isinstance(v,bool):
   if v:sys.argv.append('--'+k)
  else:sys.argv.extend(['--'+k,str(v)])
 sys.argv.extend(['--n_epochs',str(args.epochs)])
 import torch
 if torch.cuda.is_available():sys.argv.extend(['--use_cuda','--gpu','0'])
 opt=parse_opts()
 train,val=get_loaders(opt)
 observed={}
 for name,loader in [('training',train),('validation',val)]:
  counts=collections.Counter()
  for clips,labels in loader:
   assert clips.shape[1:]==(cfg['sample_duration'],3,cfg['sample_size'],cfg['sample_size'])
   assert torch.isfinite(clips).all()
   counts.update(int(y) for y in labels)
  observed[name]={'samples':sum(counts.values()),'class_indices':dict(counts)}
 from datasets.ucf101 import UCF101
 # Validate test loading without model predictions or score-based decisions.
 test=UCF101(cfg['video_path'],cfg['annotation_path'],'testing',
             spatial_transform=val.dataset.spatial_transform,temporal_transform=val.dataset.temporal_transform,
             target_transform=val.dataset.target_transform,sample_duration=cfg['sample_duration'])
 for x,y in test:
  assert x.shape==(cfg['sample_duration'],3,cfg['sample_size'],cfg['sample_size'])
 observed['testing']={'samples':len(test)}
 report={'status':'passed','loaders':observed,'input_config':build_input_config(opt,train.dataset),
         'cuda_available':torch.cuda.is_available(),'training_started':False}
 (args.dataset/'loader_verification.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2),flush=True)
 if args.train:
  run=ROOT/'Prototype Training Runs'/datetime.datetime.now().strftime('pilot_%Y%m%d_%H%M%S')
  run.mkdir(parents=True,exist_ok=False);os.chdir(run)
  (run/'run_config.json').write_text(json.dumps({'config':cfg,'epochs':args.epochs},indent=2))
  main_worker()
if __name__=='__main__':main()
