import pytest
import torch
from torch import nn
from fusion_model import FusedIntentLSTM, LABELS


def baseline():
    state={}
    for prefix,module in [('lstm',nn.LSTM(300,256,3,batch_first=True)),
                          ('fc1',nn.Linear(256,128)),('fc2',nn.Linear(128,3))]:
        state.update({prefix+'.'+k:v for k,v in module.state_dict().items()})
    return {'input_config':{'sample_duration':8,'class_labels':list(LABELS)},'state_dict':state}


def test_expansion_preserves_old_weights_and_can_learn_objects():
    torch.manual_seed(42)
    checkpoint=baseline(); model=FusedIntentLSTM.from_baseline(checkpoint)
    torch.testing.assert_close(model.lstm.weight_ih_l0[:,:300],checkpoint['state_dict']['lstm.weight_ih_l0'])
    assert not model.lstm.weight_ih_l0[:,300:].any()
    x=torch.randn(2,8,318)
    y=model(x)
    changed=x.clone(); changed[:,:,300:]+=2
    torch.testing.assert_close(y,model(changed))
    loss=nn.functional.cross_entropy(y,torch.tensor([0,1]))
    loss.backward()
    assert torch.isfinite(model.lstm.weight_ih_l0.grad).all()
    assert model.lstm.weight_ih_l0.grad[:,300:].abs().sum()>0
    optimizer=torch.optim.SGD(model.parameters(),lr=.1); optimizer.step()
    assert model.lstm.weight_ih_l0[:,300:].abs().sum()>0
    assert not torch.allclose(model(x),model(changed),rtol=0,atol=1e-8)


def test_bad_shapes_and_labels_rejected():
    model=FusedIntentLSTM()
    with pytest.raises(ValueError,match='shaped'):
        model(torch.zeros(1,8,300))
    with pytest.raises(ValueError,match='finite'):
        model(torch.full((1,8,318),float('nan')))
    checkpoint=baseline(); checkpoint['input_config']['class_labels'].reverse()
    with pytest.raises(ValueError,match='label order'):
        FusedIntentLSTM.from_baseline(checkpoint)
