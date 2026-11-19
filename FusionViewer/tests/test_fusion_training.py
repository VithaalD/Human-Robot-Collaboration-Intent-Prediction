import pytest
import torch
from tools.train_fusion_model import metrics


def test_validation_metrics_use_all_three_classes_and_correct_confusion_order():
    logits=torch.tensor([[8.,0.,0.],[0.,0.,8.],[0.,0.,8.]])
    result=metrics(logits,torch.tensor([0,1,2]),['interaction','passing','waiting'])
    assert result['correct']==2 and result['total']==3
    assert result['confusion_matrix']==[[1,0,0],[0,0,1],[0,0,1]]
    assert result['macro_f1']==pytest.approx(5/9)
    assert result['per_class']['PASSTHRU']['recall']==0
    assert result['predictions'][1]['true']=='PASSTHRU'
    assert result['predictions'][1]['predicted']=='WAIT'
