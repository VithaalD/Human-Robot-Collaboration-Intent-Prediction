"""Preprocessing consistency across live decoding, lossless storage and legacy inputs."""
import io,sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prototype_preprocessing import build_transform,resize_rgb

def test_lossless_storage_matches_direct_live_input():
    rng=np.random.default_rng(123)
    pre=build_transform(150,255,[.485,.456,.406],[.229,.224,.225])
    for shape in [(1080,1920,3),(240,427,3)]:
        original=Image.fromarray(rng.integers(0,256,shape,dtype=np.uint8))
        buf=io.BytesIO();resize_rgb(original,150).save(buf,format='PNG');buf.seek(0)
        with Image.open(buf) as prepared:
            assert torch.equal(pre(original),pre(prepared))

def test_shared_transform_preserves_existing_inference_contract():
    original=Image.fromarray(np.random.default_rng(44).integers(0,256,(240,427,3),dtype=np.uint8))
    for norm,mean,std in [(255,[.485,.456,.406],[.229,.224,.225]),(1,[114.7748,107.7354,99.4750],[1,1,1])]:
        old=transforms.Compose([transforms.Resize((150,150)),transforms.ToTensor(),transforms.Lambda(lambda x:x*(255./norm)),transforms.Normalize(mean,std)])
        assert torch.equal(old(original),build_transform(150,norm,mean,std)(original))
