"""
Real-time Intent Detection — RealSense Camera + CNN-LSTM
Watches a live camera feed and predicts whether a person is
reaching for a screwdriver to interact with the workspace.

When intent is detected above the confidence threshold, a STOP
signal is triggered. The signal output is modular — swap in your
robot arm's communication method in the `send_stop_signal()` function.

Requirements:
    pip install pyrealsense2 opencv-python torch torchvision

Usage:
    python realtime_intent.py --resume_path ./snapshots/best_model.pth

Controls (while window is open):
    Q — quit
    S — save current frame buffer as a test clip
"""

import os
import time
import argparse
import collections
import threading
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2

# RealSense — graceful fallback to webcam if not available
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("[WARNING] pyrealsense2 not installed. Falling back to webcam.")
    print("          Install with: pip install pyrealsense2\n")


# ─────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_path',     type=str,   required=True)
    parser.add_argument('--n_frames',        type=int,   default=16,   help='Frames in each inference window')
    parser.add_argument('--img_size',        type=int,   default=112)
    parser.add_argument('--hidden_size',     type=int,   default=256)
    parser.add_argument('--n_layers',        type=int,   default=2)
    parser.add_argument('--confidence_threshold', type=float, default=0.85,
                        help='Confidence above which STOP signal fires (0.0-1.0)')
    parser.add_argument('--inference_interval',   type=int,   default=8,
                        help='Run inference every N frames (lower = more responsive, higher = faster)')
    parser.add_argument('--cooldown_seconds',     type=float, default=2.0,
                        help='Minimum seconds between consecutive STOP signals')
    parser.add_argument('--camera_index',    type=int,   default=0,    help='Webcam index if RealSense unavailable')
    parser.add_argument('--display_width',   type=int,   default=848)
    parser.add_argument('--display_height',  type=int,   default=480)
    parser.add_argument('--save_dir',        type=str,   default='./saved_clips')
    return parser.parse_args()


# ─────────────────────────────────────────────
# Model (must match train_intent.py exactly)
# ─────────────────────────────────────────────

class CNNLSTM(nn.Module):
    def __init__(self, n_classes, hidden_size=256, n_layers=2):
        super().__init__()
        resnet       = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.cnn     = nn.Sequential(*list(resnet.children())[:-1])
        self.cnn_dim = 2048
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.lstm = nn.LSTM(
            input_size=self.cnn_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.3 if n_layers > 1 else 0.0
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size, n_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        with torch.no_grad():
            feats = self.cnn(x)
        feats = feats.view(B, T, self.cnn_dim)
        out, _ = self.lstm(feats)
        last   = out[:, -1, :]
        return self.classifier(last)


# ─────────────────────────────────────────────
# Stop signal — edit this function to integrate
# with your robot arm
# ─────────────────────────────────────────────

def send_stop_signal(label, confidence):
    """
    Called whenever intent is detected above the confidence threshold.

    Current behaviour: prints to console.

    TO INTEGRATE WITH YOUR ROBOT ARM, replace the body with one of:

    ── Serial/USB ──────────────────────────────────────────
    import serial
    ser = serial.Serial('COM3', 9600)   # adjust port
    ser.write(b'STOP\\n')

    ── ROS topic ───────────────────────────────────────────
    import rospy
    from std_msgs.msg import String
    pub = rospy.Publisher('/robot/stop', String, queue_size=1)
    pub.publish('STOP')

    ── TCP socket ──────────────────────────────────────────
    import socket
    s = socket.socket()
    s.connect(('192.168.1.100', 5005))  # adjust IP/port
    s.send(b'STOP')
    s.close()
    """
    print(f"\n{'='*50}")
    print(f"  !! STOP SIGNAL FIRED !!")
    print(f"  Intent : {label}")
    print(f"  Confidence : {confidence*100:.1f}%")
    print(f"{'='*50}\n")


# ─────────────────────────────────────────────
# Frame preprocessing
# ─────────────────────────────────────────────

TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────
# Camera abstraction
# ─────────────────────────────────────────────

class RealSenseCamera:
    def __init__(self, width, height):
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
        self.pipeline.start(cfg)
        # Warmup
        for _ in range(10):
            self.pipeline.wait_for_frames()
        print("RealSense camera started.")

    def read(self):
        frames      = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return False, None
        frame = np.asanyarray(color_frame.get_data())
        return True, frame

    def release(self):
        self.pipeline.stop()


class WebcamCamera:
    def __init__(self, index, width, height):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        print(f"Webcam {index} started.")

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


# ─────────────────────────────────────────────
# Inference worker (runs in background thread)
# ─────────────────────────────────────────────

class InferenceWorker:
    def __init__(self, model, device, classes, n_frames, confidence_threshold, cooldown_seconds):
        self.model                = model
        self.device               = device
        self.classes              = classes
        self.n_frames             = n_frames
        self.confidence_threshold = confidence_threshold
        self.cooldown_seconds     = cooldown_seconds

        self.result     = {'label': None, 'confidence': 0.0, 'probs': {}}
        self.busy       = False
        self.last_fired = 0.0
        self.lock       = threading.Lock()

    def infer_async(self, frame_buffer):
        if self.busy:
            return
        frames = list(frame_buffer)
        t = threading.Thread(target=self._run, args=(frames,), daemon=True)
        t.start()

    def _run(self, frames):
        self.busy = True
        try:
            tensors = torch.stack([TRANSFORM(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames])
            tensors = tensors.unsqueeze(0).to(self.device)  # (1, T, C, H, W)

            with torch.no_grad():
                logits = self.model(tensors)
                probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

            pred_idx   = int(probs.argmax())
            pred_label = self.classes[pred_idx]
            confidence = float(probs[pred_idx])
            prob_dict  = {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}

            with self.lock:
                self.result = {'label': pred_label, 'confidence': confidence, 'probs': prob_dict}

            # Fire stop signal if threshold met and cooldown elapsed
            if (pred_label == 'reaches_for_screwdriver'
                    and confidence >= self.confidence_threshold
                    and (time.time() - self.last_fired) > self.cooldown_seconds):
                self.last_fired = time.time()
                send_stop_signal(pred_label, confidence)

        except Exception as e:
            print(f"[Inference error] {e}")
        finally:
            self.busy = False


# ─────────────────────────────────────────────
# HUD overlay
# ─────────────────────────────────────────────

def draw_hud(frame, result, frame_count, inference_interval, threshold):
    h, w = frame.shape[:2]
    label      = result.get('label', 'warming up...')
    confidence = result.get('confidence', 0.0)
    probs      = result.get('probs', {})

    # Background panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (340, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    # Title
    cv2.putText(frame, "INTENT DETECTION", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

    # Current prediction
    alert = (label == 'reaches_for_screwdriver' and confidence >= threshold)
    color = (0, 60, 255) if alert else (0, 220, 100)
    cv2.putText(frame, f"Intent: {label}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
    cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    # STOP banner
    if alert:
        cv2.rectangle(frame, (0, h-50), (w, h), (0, 0, 200), -1)
        cv2.putText(frame, "!! STOP SIGNAL ACTIVE !!", (w//2 - 160, h-18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # Probability bars
    y = 125
    for cls, prob in probs.items():
        short = cls.replace('_', ' ')[:22]
        bar_w = int(prob * 200)
        bar_color = (0, 60, 255) if cls == 'reaches_for_screwdriver' else (80, 180, 80)
        cv2.rectangle(frame, (10, y), (10 + bar_w, y + 14), bar_color, -1)
        cv2.rectangle(frame, (10, y), (210, y + 14), (120, 120, 120), 1)
        cv2.putText(frame, f"{short} {prob*100:.0f}%", (215, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        y += 22

    # Threshold line on first bar
    tx = 10 + int(threshold * 200)
    cv2.line(frame, (tx, 122), (tx, 122 + 16), (255, 255, 0), 1)

    # Frame counter
    cv2.putText(frame, f"Frame {frame_count}", (10, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)

    # Controls hint
    cv2.putText(frame, "Q=quit  S=save clip", (10, h - 10 if not alert else h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1)

    return frame


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────

def main():
    args   = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    ckpt         = torch.load(args.resume_path, map_location=device)
    classes      = ckpt['classes']
    print(f"Classes: {classes}")

    model = CNNLSTM(n_classes=len(classes), hidden_size=args.hidden_size, n_layers=args.n_layers).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # Camera
    if REALSENSE_AVAILABLE:
        cam = RealSenseCamera(args.display_width, args.display_height)
    else:
        cam = WebcamCamera(args.camera_index, args.display_width, args.display_height)

    # Rolling frame buffer
    frame_buffer = collections.deque(maxlen=args.n_frames)

    # Inference worker
    worker = InferenceWorker(
        model, device, classes, args.n_frames,
        args.confidence_threshold, args.cooldown_seconds
    )

    os.makedirs(args.save_dir, exist_ok=True)
    frame_count = 0

    print("\nCamera feed started. Press Q to quit, S to save clip.\n")

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                continue

            frame_buffer.append(frame.copy())
            frame_count += 1

            # Run inference every N frames once buffer is full
            if len(frame_buffer) == args.n_frames and frame_count % args.inference_interval == 0:
                worker.infer_async(frame_buffer)

            # Draw HUD
            with worker.lock:
                result = dict(worker.result)

            display = draw_hud(frame.copy(), result, frame_count,
                               args.inference_interval, args.confidence_threshold)
            cv2.imshow('Intent Detection', display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current buffer as video clip
                ts       = int(time.time())
                out_path = os.path.join(args.save_dir, f'clip_{ts}.avi')
                h, w     = frame.shape[:2]
                writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), 15, (w, h))
                for f in frame_buffer:
                    writer.write(f)
                writer.release()
                print(f"Saved clip → {out_path}")

    finally:
        cam.release()
        cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == '__main__':
    main()
