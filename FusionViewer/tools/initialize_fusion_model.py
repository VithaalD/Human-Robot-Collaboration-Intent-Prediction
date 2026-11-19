"""Initialize the expanded temporal model and verify it without training."""
from pathlib import Path
import hashlib
import json
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fusion_model import FusedIntentLSTM, ARCHITECTURE, load_fused_checkpoint
from paired_cache_dataset import PairedCacheDataset


def main():
    torch.set_num_threads(4); torch.manual_seed(20261117)
    torch.backends.cudnn.allow_tf32=False; torch.backends.cuda.matmul.allow_tf32=False
    cache=ROOT/'.runtime/paired-cache-v1'
    contract=json.loads((cache/'contract.json').read_text())
    selected={'checkpoint': str(ROOT/'models/intent_baseline.pth')}
    baseline_path=Path(selected['checkpoint'])
    before=hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    assert before==contract['intent_sha256']
    saved=torch.load(baseline_path,map_location='cpu',weights_only=True)
    model=FusedIntentLSTM.from_baseline(saved)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model=model.to(device).eval()
    # Replicate the baseline temporal forward loop, using cached CNN features.
    recurrent=nn.LSTM(300,256,3,batch_first=False).to(device).eval()
    fc1=nn.Linear(256,128).to(device).eval(); fc2=nn.Linear(128,3).to(device).eval()
    state={k.removeprefix('module.'):v for k,v in saved['state_dict'].items()}
    for prefix,module in [('lstm',recurrent),('fc1',fc1),('fc2',fc2)]:
        module.load_state_dict({k[len(prefix)+1:]:v for k,v in state.items() if k.startswith(prefix+'.')},strict=True)
    checked=0; max_error=0.
    with torch.inference_mode():
        for x,_ in DataLoader(PairedCacheDataset(cache,'validation'),batch_size=8):
            x=x.to(device); hidden=None
            for t in range(8):
                out,hidden=recurrent(x[:,t,:300].unsqueeze(0),hidden)
            expected=fc2(torch.relu(fc1(out[-1])))
            actual=model(x)
            torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-4)
            assert torch.equal(actual.argmax(1),expected.argmax(1))
            max_error=max(max_error,float((actual-expected).abs().max()))
            checked+=len(x)
    assert checked==16
    model.train(); x,y=next(iter(DataLoader(PairedCacheDataset(cache,'training'),batch_size=8)))
    loss=nn.functional.cross_entropy(model(x.to(device)),y.to(device)); loss.backward()
    gradient=model.lstm.weight_ih_l0.grad[:,300:]
    assert torch.isfinite(gradient).all() and gradient.abs().sum()>0
    gradient_norm=float(gradient.norm()); model.zero_grad(set_to_none=True)
    assert not model.lstm.weight_ih_l0[:,300:].any()
    output=ROOT/'models/intent_yolo_initialized.pth'
    payload={'architecture':ARCHITECTURE,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
             'cache_contract':contract,'baseline_sha256':before,
             'purpose':'initialized_fusion_head_not_trained','trained':False,
             'initialization':'copy baseline temporal/head weights; zero 18 new input columns'}
    torch.save(payload,output)
    restored,metadata=load_fused_checkpoint(output,device)
    restored.eval(); model.eval()
    with torch.inference_mode():
        torch.testing.assert_close(restored(x.to(device)),model(x.to(device)),rtol=0,atol=0)
    assert hashlib.sha256(baseline_path.read_bytes()).hexdigest()==before
    report={'status':'passed','architecture':ARCHITECTURE,'checkpoint':str(output),
            'checkpoint_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
            'parameters':sum(p.numel() for p in model.parameters()),
            'validation_sequences_checked':checked,'baseline_logit_max_error':max_error,'baseline_predicted_labels_match':True,'parity_tolerance':{'rtol':1e-4,'atol':1e-4},
            'object_input_gradient_norm':gradient_norm,'checkpoint_reload_exact':True,
            'baseline_unchanged':True,'optimizer_steps_performed':0,'trained':False,
            'reserved_test_processed':False,'device':str(device),
            'note':'Architecture/initialization verification only; object features start at zero influence and require step 10 training.'}
    (ROOT/'fusion-model-verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()

