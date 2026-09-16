"""Verify GPU kernels, optimization, LSTM and detection ops without downloading weights."""
import json
import os
import tempfile
from pathlib import Path

import torch
import torchvision
from torchvision.ops import nms


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Run setup_gpu.ps1 using a supported NVIDIA driver.')
    device = torch.device('cuda:0')
    torch.manual_seed(42)
    # Run an optimizer step, not just a CUDA availability query.
    net = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3), torch.nn.ReLU(),
                              torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
                              torch.nn.Linear(8, 3)).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    before = net[0].weight.detach().clone()
    loss = torch.nn.functional.cross_entropy(net(torch.randn(2, 3, 64, 64, device=device)),
                                            torch.tensor([0, 2], device=device))
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss).item() and not torch.equal(before, net[0].weight)
    lstm = torch.nn.LSTM(300, 256, 3, batch_first=True).to(device)
    output, _ = lstm(torch.randn(1, 16, 300, device=device))
    output.square().mean().backward()
    assert torch.isfinite(output).all().item()
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device=device)
    kept = nms(boxes, torch.tensor([.9, .8], device=device), .5)
    assert kept.tolist() == [0]
    # Isolate Ultralytics settings; this check creates no downloaded checkpoint.
    with tempfile.TemporaryDirectory(prefix='intent-gpu-check-') as temporary:
        os.environ['YOLO_CONFIG_DIR'] = temporary
        import ultralytics
        detector = ultralytics.YOLO('yolo11n.yaml', task='detect')
        detector.model.to(device).eval()
        with torch.no_grad():
            detector.model(torch.zeros(1, 3, 320, 320, device=device))
        detector.model.train()
        tensors = detector.model(torch.randn(2, 3, 64, 64, device=device))
        # Exercise the detector's backward kernels with a synthetic loss.
        def leaves(value):
            if torch.is_tensor(value):
                yield value
            elif isinstance(value, dict):
                for item in value.values():
                    yield from leaves(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    yield from leaves(item)
        train_loss = sum(t.float().square().mean() for t in leaves(tensors) if t.requires_grad)
        train_loss.backward()
        assert torch.isfinite(train_loss).item()
        torch.cuda.synchronize()
        report = {
            'device': torch.cuda.get_device_name(0),
            'memory_GiB': round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
            'torch': torch.__version__, 'torchvision': torchvision.__version__,
            'cuda_build': torch.version.cuda, 'ultralytics': ultralytics.__version__,
            'checks': ['GPU optimizer step', '16-frame LSTM forward/backward',
                       'GPU detection NMS', 'YOLO11 nano forward/backward'],
            'note': 'Synthetic execution checks; no detector has been trained or evaluated for screwdriver accuracy.'
        }
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
