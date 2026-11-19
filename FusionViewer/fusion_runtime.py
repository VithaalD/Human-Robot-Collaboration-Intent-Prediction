"""Display-only fusion prediction. No robot control interfaces are used."""
from pathlib import Path
import hashlib
import time
import cv2
import numpy as np
import torch
from fusion_model import LABELS, load_fused_checkpoint
from paired_features import PairedFeatureExtractor, PairedFeatureStream
from yolo_features import CLASS_ORDER


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


class FusionPredictor:
    def __init__(self,intent_weights,yolo_weights,fusion_weights):
        device='cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model,self.checkpoint=load_fused_checkpoint(fusion_weights,device)
        if not self.checkpoint.get('trained'): raise ValueError('Use the trained fusion checkpoint')
        contract=self.checkpoint['cache_contract']
        if file_hash(intent_weights)!=contract['intent_sha256'] or file_hash(yolo_weights)!=contract['yolo_sha256']:
            raise ValueError('Feature-extractor weights do not match training')
        settings=contract['input']['detector']
        self.extractor=PairedFeatureExtractor.from_checkpoints(intent_weights,yolo_weights,
                              confidence=settings['confidence'],image_size=settings['imgsz'])
        if self.extractor.metadata()!=contract['input']:
            raise ValueError('Live preprocessing does not match training contract')
        self.device=self.extractor.device; self.model.eval()
        self.stream=PairedFeatureStream(self.extractor)
        self.last_prediction=None; self.accepted=0

    def warmup(self):
        cnn,objects=self.extractor(np.zeros((480,640,3),dtype=np.uint8))
        with torch.inference_mode():
            self.model(torch.from_numpy(np.concatenate((cnn,objects))).to(self.device).repeat(1,8,1))
        self.reset()

    def reset(self):
        self.stream.reset(); self.last_prediction=None

    def process(self,frame,timestamp,frame_id,source_id):
        started=time.perf_counter()
        try:
            row=self.stream.push(frame,timestamp,frame_id,source_id)
            window=self.stream.window()
            if window is None: self.last_prediction=None
            if row is not None:
                self.accepted+=1
                if window is not None:
                    tensor=torch.from_numpy(window['features']).unsqueeze(0).to(self.device)
                    with torch.inference_mode(), torch.backends.cudnn.flags(allow_tf32=False):
                        probabilities=self.model(tensor).softmax(1)[0].cpu().numpy()
                    self.last_prediction={'label':LABELS[int(probabilities.argmax())],
                                          'probabilities':probabilities.tolist(),'timestamp':float(timestamp),
                                          'frame_ids':window['frame_ids']}
            return {'prediction':self.last_prediction,'objects':None if row is None else row.objects,
                    'accepted':row is not None,'buffered':len(self.stream.frames),
                    'processing_ms':(time.perf_counter()-started)*1000}
        except Exception:
            self.reset(); raise


def annotate(frame,result,status=None,age=None):
    canvas=frame.copy(); h,w=canvas.shape[:2]
    objects=result.get('objects')
    if objects is not None:
        for name,row in zip(CLASS_ORDER,objects.reshape(3,6)):
            present,confidence,cx,cy,bw,bh=row
            if not present: continue
            x1=max(0,int((cx-bw/2)*w)); y1=max(0,int((cy-bh/2)*h))
            x2=min(w-1,int((cx+bw/2)*w)); y2=min(h-1,int((cy+bh/2)*h))
            cv2.rectangle(canvas,(x1,y1),(x2,y2),(70,220,140),2)
            cv2.putText(canvas,f'{name} {confidence:.2f}',(x1,max(18,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.6,(70,220,140),2)
    scale=min(1.,1100/w,740/h); canvas=cv2.resize(canvas,(int(w*scale),int(h*scale)))
    panel=np.zeros((150,canvas.shape[1],3),np.uint8)
    prediction=result.get('prediction')
    lines=['YOLO + CNN-LSTM | experimental fusion | display only',
           status or (f'Intent: {prediction["label"]}' if prediction else f'Collecting frames: {result.get("buffered",0)}/8')]
    if prediction and status is None:
        lines.append('  '.join(f'{name}: {p:.1%}' for name,p in zip(LABELS,prediction['probabilities'])))
    lines.append(f'Processing: {result.get("processing_ms",0):.1f} ms'+(f' | frame age: {age*1000:.0f} ms' if age is not None else ''))
    lines.append('Q / Esc: quit | R: reset sequence')
    for i,line in enumerate(lines):cv2.putText(panel,line,(12,24+i*27),cv2.FONT_HERSHEY_SIMPLEX,.55,(235,235,235),1,cv2.LINE_AA)
    return np.vstack((panel,canvas))
