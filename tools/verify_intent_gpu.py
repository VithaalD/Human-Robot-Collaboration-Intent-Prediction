"""Check the actual intent network on the GPU without saving changed weights."""
import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Intent Monitor'))
from intent_engine_v3_1_windows import EngineConfig, IntentEngine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--sample-duration', type=int, default=16)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use the intent-gpu environment.')
    engine = IntentEngine(EngineConfig(
        resume_path=args.weights, class_labels=['INTERACTION', 'PASSTHRU', 'WAIT'],
        n_classes=3, sample_duration=args.sample_duration, sample_size=150))
    model = engine._model
    frames = torch.randn(1, args.sample_duration, 3, 150, 150, device=engine.device)
    with torch.no_grad():
        predictions = model(frames)
    assert predictions.shape == (1, 3) and torch.isfinite(predictions).all().item()
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
    loss = torch.nn.functional.cross_entropy(model(frames), torch.tensor([0], device=engine.device))
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    assert torch.isfinite(loss).item()
    print(json.dumps({
        'device': str(engine.device), 'sample_duration': args.sample_duration,
        'input_size': 150, 'batch_size': 1,
        'peak_allocated_GiB': round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        'checks': ['Checkpoint loading', 'ResNet101/LSTM GPU inference',
                   'ResNet101/LSTM GPU backward and optimizer step'],
        'note': 'Synthetic test; changed weights exist only in memory and are not saved.'
    }, indent=2))


if __name__ == '__main__':
    main()
