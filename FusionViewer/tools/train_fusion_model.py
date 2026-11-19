"""Train the fusion temporal head on frozen features; never open reserved test data."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fusion_model import ARCHITECTURE, LABELS, load_fused_checkpoint
from paired_cache_dataset import PairedCacheDataset


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def metrics(logits,targets,names):
    probabilities=logits.softmax(1); predictions=logits.argmax(1)
    cm=torch.zeros(3,3,dtype=torch.int64)
    for a,b in zip(targets.tolist(),predictions.tolist()):cm[a,b]+=1
    per={}
    for c,label in enumerate(LABELS):
        tp=int(cm[c,c]); support=int(cm[c].sum())
        precision=tp/max(int(cm[:,c].sum()),1); recall=tp/max(support,1)
        per[label]={'precision':precision,'recall':recall,'f1':2*precision*recall/max(precision+recall,1e-12),'support':support}
    return {'loss':float(nn.functional.cross_entropy(logits,targets)),
            'accuracy':float((predictions==targets).float().mean()),
            'correct':int((predictions==targets).sum()),'total':len(targets),
            'macro_f1':sum(v['f1'] for v in per.values())/3,
            'confusion_matrix':cm.tolist(),'per_class':per,
            'predictions':[{'recording':n,'true':LABELS[a],'predicted':LABELS[b],'probabilities':p}
                           for n,a,b,p in zip(names,targets.tolist(),predictions.tolist(),probabilities.tolist())]}


def evaluate(model,loader,device,zero_objects=False):
    model.eval(); outputs=[]; labels=[]
    with torch.inference_mode():
        for x,y in loader:
            if zero_objects:x[:,:,300:]=0
            outputs.append(model(x.to(device)).cpu()); labels.append(y)
    logits=torch.cat(outputs); targets=torch.cat(labels)
    assert torch.isfinite(logits).all()
    return metrics(logits,targets,[n for n,_ in loader.dataset.items]),logits


def save_model(path,model,initial,config,epoch,validation):
    payload={'architecture':ARCHITECTURE,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
             'cache_contract':initial['cache_contract'],'baseline_sha256':initial['baseline_sha256'],
             'purpose':'trained_fusion_temporal_head','trained':True,'epoch':epoch,
             'training_config':config,'validation':validation}
    temporary=path.with_suffix('.partial.pth'); torch.save(payload,temporary); temporary.replace(path)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--epochs',type=int,default=100)
    ap.add_argument('--patience',type=int,default=20)
    ap.add_argument('--learning-rate',type=float,default=1e-4)
    args=ap.parse_args()
    if args.epochs<1 or args.patience<1 or args.learning_rate<=0:raise ValueError('Invalid training settings')
    seed=20261117; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(4); torch.backends.cudnn.allow_tf32=False; torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cache=ROOT/'.runtime/paired-cache-v1'; init_path=ROOT/'models/intent_yolo_initialized.pth'
    model,initial=load_fused_checkpoint(init_path,device)
    if initial.get('trained'):raise ValueError('Expected the untrained initializer')
    train=PairedCacheDataset(cache,'training'); val=PairedCacheDataset(cache,'validation')
    assert train.contract==val.contract==initial['cache_contract']
    manifest=json.loads((cache/'manifest.json').read_text())
    for entry in manifest['clips'].values():
        if entry['subset']=='testing':assert entry['status']=='reserved_unprocessed'; continue
        assert sha(cache/entry['cache_path'])==entry['sha256']
    selection={'checkpoint': str(ROOT/'models/intent_baseline.pth')}
    protected={str(init_path):sha(init_path),str(ROOT/'models/custom_yolo11.pt'):sha(ROOT/'models/custom_yolo11.pt'),selection['checkpoint']:sha(selection['checkpoint'])}
    assert protected[str(ROOT/'models/custom_yolo11.pt')]==train.contract['yolo_sha256']
    assert protected[selection['checkpoint']]==train.contract['intent_sha256']
    run=ROOT/'.runtime/fusion-training'/datetime.now().strftime('%Y%m%d_%H%M%S'); run.mkdir(parents=True,exist_ok=False)
    config={'seed':seed,'maximum_epochs':args.epochs,'patience':args.patience,'learning_rate':args.learning_rate,
            'batch_size':8,'optimizer':'AdamW','weight_decay':.001,'gradient_clip':5.,
            'selection':'lowest unweighted validation cross-entropy; minimum improvement 0.0001',
            'scheduler':'ReduceLROnPlateau factor 0.5 patience 5',
            'sampling':'inverse-frequency replacement by training recording; random eight-frame crop',
            'validation':'center valid eight-frame crop per validation recording',
            'frozen':'CNN and YOLO (cached features)','trainable':'LSTM and classification head',
            'device':str(device),'torch':str(torch.__version__),'cache_signature':manifest['signature']}
    (run/'config.json').write_text(json.dumps(config,indent=2))
    counts=Counter(train.labels)
    sampler=WeightedRandomSampler([1/counts[y] for y in train.labels],len(train),replacement=True)
    train_loader=DataLoader(train,batch_size=8,sampler=sampler,num_workers=0)
    val_loader=DataLoader(val,batch_size=8,shuffle=False,num_workers=0)
    baseline,_=evaluate(model,val_loader,device)
    (run/'baseline_validation.json').write_text(json.dumps(baseline,indent=2))
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=.001)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=5,factor=.5)
    best_loss=float('inf'); stale=0; history=[]; started=time.monotonic(); best_path=run/'best_checkpoint.pth'
    print('RUN '+str(run),flush=True)
    print(f'Baseline validation: {baseline["correct"]}/{baseline["total"]}, loss {baseline["loss"]:.6f}',flush=True)
    for epoch in range(1,args.epochs+1):
        model.train(); total_loss=0.; total_correct=0; seen=0
        for x,y in train_loader:
            x=x.to(device); y=y.to(device); optimizer.zero_grad(set_to_none=True)
            logits=model(x); loss=nn.functional.cross_entropy(logits,y)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
            loss.backward(); norm=nn.utils.clip_grad_norm_(model.parameters(),5.)
            if not torch.isfinite(norm):raise RuntimeError('Non-finite gradients')
            optimizer.step(); total_loss+=float(loss.detach())*len(y)
            total_correct+=int((logits.argmax(1)==y).sum()); seen+=len(y)
        validation,_=evaluate(model,val_loader,device); scheduler.step(validation['loss'])
        improved=validation['loss']<best_loss-1e-4
        if improved:
            best_loss=validation['loss']; stale=0
            save_model(best_path,model,initial,config,epoch,validation)
        else:stale+=1
        history.append({'epoch':epoch,'sampled_training_loss':total_loss/seen,
                        'sampled_training_accuracy':total_correct/seen,'validation':validation,
                        'learning_rate':optimizer.param_groups[0]['lr'],'selected':improved})
        (run/'history.json').write_text(json.dumps(history,indent=2))
        torch.save({'epoch':epoch,'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),
                    'scheduler':scheduler.state_dict(),'torch_rng':torch.get_rng_state(),
                    'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    'training_config':config},run/'last_training_state.pth')
        print(f'Epoch {epoch}: train loss {total_loss/seen:.4f}, val loss {validation["loss"]:.6f}, correct {validation["correct"]}/16'+(' BEST' if improved else ''),flush=True)
        if stale>=args.patience:break
    best,checkpoint=load_fused_checkpoint(best_path,device)
    final,final_logits=evaluate(best,val_loader,device)
    assert abs(final['loss']-checkpoint['validation']['loss'])<1e-7
    zeroed,zeroed_logits=evaluate(best,val_loader,device,True)
    object_norm=float(best.lstm.weight_ih_l0[:,300:].detach().norm())
    assert object_norm>0
    output=ROOT/'models/intent_yolo_fusion.pth'
    save_model(output,best,initial,config,checkpoint['epoch'],final)
    restored,_=load_fused_checkpoint(output,device)
    _,reloaded_logits=evaluate(restored,val_loader,device)
    torch.testing.assert_close(reloaded_logits,final_logits,rtol=0,atol=0)
    for path,digest in protected.items():assert sha(path)==digest
    report={'status':'complete','run_directory':str(run),'checkpoint':str(output),'checkpoint_sha256':sha(output),
            'epochs_completed':len(history),'best_epoch':checkpoint['epoch'],'elapsed_seconds':time.monotonic()-started,
            'baseline_validation':baseline,'selected_validation':final,'zero_object_ablation':zeroed,
            'object_weight_norm':object_norm,'zero_object_max_logit_change':float((final_logits-zeroed_logits).abs().max()),
            'checkpoint_reload_exact':True,'protected_checkpoints_unchanged':True,
            'reserved_test_processed':False,'trained':True,'training_config':config,'limitations':manifest['limitations']+
            ['Only 16 validation recordings; used for model selection, not final unbiased test accuracy.',
             'Zero-object ablation is a sensitivity check, not a separately trained no-YOLO control.']}
    (run/'training_report.json').write_text(json.dumps(report,indent=2))
    (ROOT/'fusion-training-report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ('status','epochs_completed','best_epoch','checkpoint','object_weight_norm','zero_object_max_logit_change','elapsed_seconds')},indent=2),flush=True)
    print('Final validation: '+str(final['correct'])+'/16; baseline '+str(baseline['correct'])+'/16',flush=True)


if __name__=='__main__': main()
