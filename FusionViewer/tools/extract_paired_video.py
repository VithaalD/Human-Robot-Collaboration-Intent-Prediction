"""Extract and verify the first complete paired-feature window from a video."""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from frame_sampling import recording_timestamps
from paired_features import PairedFeatureExtractor, PairedFeatureStream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--intent-weights', required=True)
    parser.add_argument('--yolo-weights', default=str(ROOT / 'models/custom_yolo11.pt'))
    parser.add_argument('--output', default=str(ROOT / '.runtime/paired-window.npz'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    extractor = PairedFeatureExtractor.from_checkpoints(args.intent_weights, args.yolo_weights)
    stream = PairedFeatureStream(extractor)
    cap = cv2.VideoCapture(args.video)
    source = str(Path(args.video).resolve())
    try:
        if not cap.isOpened():
            raise RuntimeError('Could not open video')
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count <= 0:
            raise ValueError('Video frame count unavailable')
        stamps, metadata = recording_timestamps(args.video, count, cap.get(cv2.CAP_PROP_FPS))
        first_frame = None
        first_record = None
        started = time.perf_counter()
        for frame_id, stamp in enumerate(stamps):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'Video ended before expected frame {frame_id}')
            before = frame.copy()
            row = stream.push(frame, stamp, frame_id, source)
            if row is not None:
                assert np.array_equal(before, frame), 'Input frame was modified'
                if first_record is None:
                    first_frame, first_record = before, row
            window = stream.window()
            if window is not None:
                break
        else:
            raise RuntimeError('No complete eight-frame window was available')
    finally:
        cap.release()
    elapsed = time.perf_counter() - started
    # Independent baseline branch call verifies the extracted CNN slice.
    rgb = Image.fromarray(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
    tensor = extractor.engine._transform(rgb).unsqueeze(0).to(extractor.device)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.inference_mode(), torch.backends.cudnn.flags(allow_tf32=False):
            expected = extractor.engine._model.resnet(tensor)[0].float().cpu().numpy()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    np.testing.assert_allclose(first_record.cnn, expected, rtol=1e-5, atol=1e-5)
    assert np.isfinite(window['features']).all()
    assert np.all(np.diff(window['timestamps']) > 0)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, features=window['features'],
                        cnn=window['features'][:, :300], objects=window['features'][:, 300:],
                        timestamps=window['timestamps'], frame_ids=window['frame_ids'])
    report = {'status': 'passed', 'source': source, 'intent_weights': args.intent_weights,
              'yolo_weights': args.yolo_weights, 'input_contract': extractor.metadata(),
              'timestamp_source': 'capture metadata' if metadata is not None else 'nominal video FPS',
              'frame_ids': window['frame_ids'], 'timestamps': window['timestamps'].tolist(),
              'shape': list(window['features'].shape), 'dtype': str(window['features'].dtype),
              'cnn_parity_max_error': float(np.abs(first_record.cnn - expected).max()),
              'input_frames_unmodified': True, 'cnn_device': str(extractor.device),
              'yolo_device': str(extractor.detector.predictor.device),
              'extraction_seconds_including_warmup': elapsed,
              'note': 'Feature extraction only. No fusion classifier training or live camera test. Warmup-inclusive runtime is not steady-state throughput.'}
    output.with_suffix('.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
