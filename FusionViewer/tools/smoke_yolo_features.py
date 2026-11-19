"""Check local detector and feature extraction on one supplied video frame."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from yolo_features import extract_yolo_features, feature_schema


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--weights', default=str(ROOT / 'models/custom_yolo11.pt'))
    parser.add_argument('--output', default=str(ROOT / '.runtime/yolo-smoke.json'))
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default='0')
    args = parser.parse_args()
    cap = cv2.VideoCapture(args.video)
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise RuntimeError('Could not read source video')
    model = YOLO(args.weights)
    result = model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                           device=args.device, verbose=False)[0]
    features = extract_yolo_features(result)
    assert features.shape == (18,) and np.isfinite(features).all()
    report = {'status': 'passed', 'source_video': args.video, 'frame_index': 0,
              'source_shape': list(frame.shape), 'classes': model.names,
              'confidence_threshold': args.conf, 'image_size': args.imgsz,
              'device': str(model.predictor.device), 'detections': len(result.boxes),
              'features': features.tolist(), 'schema': feature_schema(),
              'note': 'Runtime/format check only; partner visual testing reported by user. Not an accuracy evaluation.'}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

