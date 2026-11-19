"""Run the trained fusion viewer on recorded video, webcam, or RealSense RGB."""
import argparse
import json
from pathlib import Path
import sys
import threading
import time
import cv2
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fusion_runtime import FusionPredictor, annotate
from frame_sampling import recording_timestamps


class LatestCamera:
    """Capture independently; retain only the latest frame, never a growing queue."""
    def __init__(self,camera):
        self.camera=camera; self.lock=threading.Lock(); self.packet=None
        self.error=None; self.stop_event=threading.Event()
        self.thread=threading.Thread(target=self._run,daemon=True)
        self.thread.start()

    def _run(self):
        number=0; bad=0
        try:
            while not self.stop_event.is_set():
                ok,frame=self.camera.read(); captured=time.monotonic()
                if not ok:
                    bad+=1
                    with self.lock:self.packet=None; self.error='Camera frame unavailable'
                    if bad>=30:break
                    self.stop_event.wait(.03); continue
                bad=0; number+=1
                stamp=getattr(self.camera,'timestamp',captured)
                with self.lock:self.packet=(number,stamp,captured,frame.copy()); self.error=None
        except Exception as exc:
            with self.lock:self.packet=None; self.error=str(exc)
        finally:self.camera.release()

    def snapshot(self):
        with self.lock:return self.packet,self.error

    def close(self):
        self.stop_event.set(); self.thread.join(timeout=3)
        return not self.thread.is_alive()


class RealSenseColor:
    def __init__(self):
        try:import pyrealsense2 as rs
        except ImportError as exc:raise RuntimeError('RealSense SDK Python package is unavailable; use --camera INDEX for its RGB webcam interface.') from exc
        self.pipeline=rs.pipeline(); cfg=rs.config()
        cfg.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30)
        self.pipeline.start(cfg); self.timestamp=0.

    def read(self):
        try:frames=self.pipeline.wait_for_frames(timeout_ms=250)
        except RuntimeError:return False,None
        frame=frames.get_color_frame()
        if not frame:return False,None
        self.timestamp=frame.get_timestamp()/1000
        return True,np.asanyarray(frame.get_data()).copy()

    def release(self):self.pipeline.stop()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    source=ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--video'); source.add_argument('--demo',action='store_true')
    source.add_argument('--camera',type=int); source.add_argument('--realsense',action='store_true')
    ap.add_argument('--intent-weights'); ap.add_argument('--yolo-weights',default=str(ROOT/'models/custom_yolo11.pt'))
    ap.add_argument('--fusion-weights',default=str(ROOT/'models/intent_yolo_fusion.pth'))
    ap.add_argument('--headless',action='store_true'); ap.add_argument('--max-frames',type=int,default=0)
    ap.add_argument('--max-seconds',type=float,default=0)
    ap.add_argument('--report',default=str(ROOT/'.runtime/fusion-viewer-report.json'))
    ap.add_argument('--save-preview'); ap.add_argument('--stop-file'); args=ap.parse_args()
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False

    intent=args.intent_weights or str(ROOT/'models/intent_baseline.pth')
    video=str(ROOT/'demo/lab.avi') if args.demo else args.video
    print('Loading and warming up fusion model...',flush=True)
    predictor=FusionPredictor(intent,args.yolo_weights,args.fusion_weights); predictor.warmup()
    live=not bool(video); cap=None; camera=None; capture=None
    frames_seen=0; predictions=0; timings=[]; latest_canvas=None; first_wall=None; last_prediction=None
    started=time.monotonic(); interrupted=False; shutdown=True; status='complete'
    title='Fusion Intent Viewer'
    try:
        if video:
            cap=cv2.VideoCapture(video)
            if not cap.isOpened():raise RuntimeError('Cannot open video: '+video)
            count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if count<=0:raise ValueError('Video frame count unavailable')
            stamps,metadata=recording_timestamps(video,count,cap.get(cv2.CAP_PROP_FPS))
            source_id=str(Path(video).resolve()); index=0; replay_start=time.monotonic()
        else:
            if args.realsense:camera=RealSenseColor(); source_id='realsense'
            else:
                sys.path.insert(0,str(ROOT/'Intent Monitor'))
                from intent_engine_v3_1_windows import _OpenCVCamera
                camera=_OpenCVCamera(index=args.camera); source_id=f'camera:{args.camera}'
            capture=LatestCamera(camera); last_id=None
        if not args.headless:cv2.namedWindow(title,cv2.WINDOW_NORMAL)
        while True:
            if args.stop_file and Path(args.stop_file).exists():break
            if args.max_seconds and time.monotonic()-started>=args.max_seconds:break
            if args.max_frames and frames_seen>=args.max_frames:break
            if video:
                if index>=len(stamps):break
                ok,frame=cap.read()
                if not ok:raise RuntimeError('Video ended earlier than its timing metadata')
                stamp=stamps[index]; frame_id=index; index+=1; captured=None
                # Recorded playback preserves source time, even on slower hardware.
                if not args.headless:
                    while time.monotonic()<replay_start+stamp:
                        if args.stop_file and Path(args.stop_file).exists():interrupted=True; break
                        key=cv2.waitKey(10)&255
                        if key in (27,ord('q')):interrupted=True; break
                    if interrupted:break
            else:
                packet,error=capture.snapshot()
                if packet is None:
                    predictor.reset()
                    blank=np.zeros((480,640,3),np.uint8)
                    latest_canvas=annotate(blank,{},status=error or 'Waiting for camera')
                    if not args.headless:
                        cv2.imshow(title,latest_canvas)
                        if cv2.waitKey(20)&255 in (27,ord('q')):break
                    else:time.sleep(.02)
                    if not capture.thread.is_alive():raise RuntimeError(error or 'Camera stopped')
                    continue
                frame_id,stamp,captured,frame=packet
                if frame_id==last_id:
                    if time.monotonic()-captured>.5:
                        predictor.reset(); latest_canvas=annotate(frame,{},status='STALE CAMERA FRAME')
                        if not args.headless:cv2.imshow(title,latest_canvas)
                    if not args.headless:
                        if cv2.waitKey(10)&255 in (27,ord('q')):break
                    else:time.sleep(.005)
                    continue
                last_id=frame_id
                if time.monotonic()-captured>.5:
                    predictor.reset(); continue
            result=predictor.process(frame,stamp,frame_id,source_id); frames_seen+=1
            age=None if captured is None else time.monotonic()-captured
            stale=age is not None and age>.5
            if stale:
                predictor.reset(); result['prediction']=None
            prediction=result['prediction']
            if prediction and result['accepted']:
                predictions+=1; last_prediction=prediction
                if first_wall is None:first_wall=time.monotonic()-started
            if result['accepted']:timings.append(result['processing_ms'])
            latest_canvas=annotate(frame,result,status='STALE RESULT - recollecting frames' if stale else None,age=age)
            if not args.headless:
                cv2.imshow(title,latest_canvas); key=cv2.waitKey(1)&255
                if key in (27,ord('q')):break
                if key==ord('r'):predictor.reset()
                if cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)<1:break
    except Exception:
        status='error'; raise
    finally:
        if capture is not None:shutdown=capture.close()
        if cap is not None:cap.release()
        if not args.headless:cv2.destroyAllWindows()
        if args.save_preview and latest_canvas is not None:
            dest=Path(args.save_preview); dest.parent.mkdir(parents=True,exist_ok=True)
            if not cv2.imwrite(str(dest),latest_canvas):raise RuntimeError('Could not save preview')
        report={'status':status,'source':video or ('realsense' if args.realsense else f'camera:{args.camera}'),
                'frames_processed':frames_seen,'sampled_frames':predictor.accepted,'predictions':predictions,
                'first_prediction_wall_seconds':first_wall,'last_prediction':last_prediction,
                'processing_ms_median':float(np.median(timings)) if timings else None,
                'capture_thread_stopped':shutdown,'live_camera_tested':live and frames_seen>0,
                'note':'Display-only engineering run; no accuracy evaluation or robot commands.'}
        dest=Path(args.report); dest.parent.mkdir(parents=True,exist_ok=True); dest.write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2),flush=True)
        if not shutdown:print('WARNING: camera driver did not finish shutdown within 3 seconds.',flush=True)


if __name__=='__main__':main()


