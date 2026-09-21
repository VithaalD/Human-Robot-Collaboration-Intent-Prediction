# DeepLabCut integration verification

Verified on this laptop on 2026-09-21 with Python 3.12, DeepLabCut 3.0.1,
PyTorch 2.10.0+cu128, and NVIDIA RTX 5070 Laptop GPU.

- Full `tests` suite: **62 passed**. Existing recording and temporal regression
  tests remain passing. Warnings came from deprecated APIs in the legacy loader.
- New tests: causal sampling, normalized motion, masks for missing landmarks,
  motion/fusion gradient flow, stale-pose refusal, bounded worker/frame identity,
  split leakage checks, strict CSV frame IDs, synthetic preparation/training/
  evaluation, and matching replay/evaluation predictions.
- CUDA optimizer steps passed for both motion and fusion classifiers.
- The actual DeepLabCut 3.0.1 bottom-up inference runner loaded a temporary
  random-weight ResNet50 pose snapshot and produced eight x/y/likelihood points
  on CUDA. No pretrained model was downloaded for this test.
- The existing external epoch-196 RGB checkpoint loaded strictly into the
  shared RGB encoder and produced finite 128-dimensional embeddings and
  three-class logits for a 16-frame synthetic clip on CUDA.

Tests use synthetic fixtures. No newly trained person/screwdriver model exists
yet. Real pose accuracy, live camera throughput, live graphics, and improved
intent accuracy have not been established by these tests. A labeled pilot and
held-out recording sessions are required. Offline event timing excludes compute
latency; live predictions enforce a freshness limit and report unavailable data.

Run in the activated `deeplabcut-gpu` environment from the repository root:

```powershell
python -m pytest tests -q
```

On this machine the default pytest temporary folder had a Windows ownership
conflict. Verification used a fresh directory under the Git-ignored
`.pose-verification` folder and disabled automatic third-party pytest plugins.
Neither synthetic videos nor random model snapshots are included in the branch.
