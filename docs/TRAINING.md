# Training

Training uses the fixed `train` and `val` splits described in [DATA.md](DATA.md). Supply trusted processed PKLs through `--data-root` before running the commands below.

## Released Full-VAE Checkpoints

The three model-only full-VAE checkpoints can be used for DiT training or VAE reconstruction.

| Variant | Dataset | State | Checkpoint |
| --- | --- | --- | --- |
| `4-30` | DeepCAD | online | `training/deepcad/hifi-brep-vae-4-30-online.pt` |
| `7-30` | DeepCAD | online | `training/deepcad/hifi-brep-vae-7-30-online.pt` |
| `abc-4-50` | ABC | EMA | `training/abc/hifi-brep-vae-4-50-ema.pt` |

HuggingFace:

```bash
hf download 1nnoh/HiFi-BRep \
  --include "training/*/*.pt" \
  --local-dir checkpoints
```

ModelScope:

```bash
python -m pip install modelscope-hub==0.1.8
modelscope download innohou/HiFi-BRep \
  --include "training/*/*.pt" \
  --local-dir checkpoints
```

## Train the VAE

ABC (`abc-4-50`):

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 -m train.vae \
  --config configs/training/vae_abc.yaml \
  --data-root /data/hifi-brep/abc \
  --output-dir /runs/hifi-brep/abc-vae
```

DeepCAD (`4-30`):

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 -m train.vae \
  --config configs/training/vae_deepcad.yaml \
  --data-root /data/hifi-brep/deepcad \
  --output-dir /runs/hifi-brep/deepcad-vae
```

When validation finds a new best checkpoint, the trainer also writes the selected full-VAE weight to:

- ABC: `/runs/hifi-brep/abc-vae/artifacts/training/abc/hifi-brep-vae-4-50-ema.pt`
- DeepCAD: `/runs/hifi-brep/deepcad-vae/artifacts/training/deepcad/hifi-brep-vae-4-30-online.pt`

## Train the DiT

ABC (`abc-4-50`):

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 -m train.diffusion \
  --config configs/training/diffusion_abc.yaml \
  --data-root /data/hifi-brep/abc \
  --vae-checkpoint /runs/hifi-brep/abc-vae/artifacts/training/abc/hifi-brep-vae-4-50-ema.pt \
  --per-device-batch-size 512 \
  --output-dir /runs/hifi-brep/abc-dit
```

The released ABC DiT recipe uses Adam with zero weight decay for at most 1,500
epochs, a 30-epoch warmup, and a global batch size of 512. The selected release
artifact is the epoch-500 EMA state.

DeepCAD (`4-30`):

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 -m train.diffusion \
  --config configs/training/diffusion_deepcad.yaml \
  --data-root /data/hifi-brep/deepcad \
  --vae-checkpoint /runs/hifi-brep/deepcad-vae/artifacts/training/deepcad/hifi-brep-vae-4-30-online.pt \
  --output-dir /runs/hifi-brep/deepcad-dit
```

The best DiT checkpoint writes a decoder/diffusion pair under `<output-dir>/artifacts/` with the same layout used by the released checkpoints. For example:

```bash
CUDA_VISIBLE_DEVICES=0 python demo.py \
  --variant abc-4-50 \
  --checkpoint-dir /runs/hifi-brep/abc-dit/artifacts \
  --output-dir outputs/demo/abc-retrained
```

## Resume Training

To continue an interrupted run, repeat the same command with `--resume <output-dir>/checkpoint-latest.pt`, using the same configuration and output directory.
Training resume accepts only checkpoints written by the current safe checkpoint
format; legacy training checkpoints are not supported.

## VAE Reconstruction Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python -m eval.vae_reconstruction \
  --config configs/training/vae_deepcad.yaml \
  --data-root /data/hifi-brep/deepcad \
  --variant 4-30 \
  --checkpoint checkpoints/training/deepcad/hifi-brep-vae-4-30-online.pt \
  --output-dir outputs/vae-reconstruction/4-30
```

This evaluator decodes posterior means with ground-truth face and edge counts. A reconstruction is Valid when Open CASCADE produces a closed solid with nonzero volume.
