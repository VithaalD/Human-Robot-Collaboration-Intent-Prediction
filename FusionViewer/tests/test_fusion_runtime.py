from types import SimpleNamespace
import time
import numpy as np
import torch
import pytest
from paired_features import PairedFeatureStream
from fusion_runtime import FusionPredictor, annotate
from tools.run_fusion_viewer import LatestCamera


def fake_predictor():
    predictor=FusionPredictor.__new__(FusionPredictor)
    predictor.device=torch.device('cpu')
    predictor.model=lambda x:torch.tensor([[2.,1.,0.]])
    predictor.stream=PairedFeatureStream(lambda f:(np.zeros(300,np.float32),np.zeros(18,np.float32)))
    predictor.last_prediction=None; predictor.accepted=0
    return predictor


def test_runtime_warmup_prediction_and_gap_reset():
    predictor=fake_predictor(); frame=np.zeros((100,200,3),np.uint8)
    for i in range(7):assert predictor.process(frame,i/8,i,'a')['prediction'] is None
    result=predictor.process(frame,7/8,7,'a')
    assert result['prediction']['label']=='INTERACTION'
    assert sum(result['prediction']['probabilities'])==pytest.approx(1)
    assert result['prediction']['frame_ids']==list(range(8))
    result=predictor.process(frame,3,8,'a')
    assert result['prediction'] is None and result['buffered']==1
    predictor.reset(); assert predictor.last_prediction is None


def test_overlay_does_not_modify_frame():
    frame=np.zeros((100,200,3),np.uint8)
    objects=np.zeros(18,np.float32); objects[:6]=[1,.9,.5,.5,.5,.5]
    rendered=annotate(frame,{'objects':objects,'prediction':None,'buffered':2})
    assert rendered.shape==(250,200,3)
    assert not frame.any() and rendered.any()


def test_capture_keeps_latest_and_closes_driver():
    class Camera:
        count=0; released=False
        def read(self):
            time.sleep(.002); self.count+=1
            return True,np.full((2,2,3),self.count%255,np.uint8)
        def release(self):self.released=True
    camera=Camera(); capture=LatestCamera(camera)
    deadline=time.monotonic()+2
    while camera.count<6 and time.monotonic()<deadline:time.sleep(.01)
    assert capture.close() and camera.released
    packet,error=capture.snapshot()
    assert error is None and packet[0]>=6
    assert int(packet[3][0,0,0])==packet[0]%255
