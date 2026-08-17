# Unconditional Generation Evaluation

The generation evaluator measures Valid for the released `4-30`, `7-30`, and `abc-4-50` checkpoint pairs without loading a dataset.

## Evaluation Settings

| Setting | Value |
| --- | --- |
| Samples | 4,096 per variant |
| Sampling | 400-step DDIM, `eta=0.2` |
| Precision | FP32 |
| Batch size | 32 |
| Seed | 42; batch seed is `42 + batch_index` |
| STEP output | Disabled |

For exact reproduction, keep the batch size and seed unchanged.

## Run the Evaluation

After downloading the six inference checkpoints to `checkpoints/`, run:

```bash
CUDA_VISIBLE_DEVICES=0 python -m eval.generation \
  --variant 4-30 \
  --output-dir outputs/evaluation/4-30
```

Replace `4-30` with `7-30` or `abc-4-50` for the other released variants. Repeat the same command with `--resume` to continue an interrupted run.

The evaluator writes `run_config.json`, `samples.jsonl`, and `summary.json`. Add `--save-steps valid` to save Valid shapes or `--save-steps all` to test STEP serialization for every sample; only the latter measures Compilability.

## Valid Metric

A sample is Valid when its predicted face count is inside the selected variant's range and Open CASCADE reconstructs a closed solid with nonzero volume. The denominator is the full requested sample set, including failed reconstructions and out-of-range predictions. `summary.json` records this score as `metrics.qualified_rate`.

The DeepCAD result combines equal numbers of samples from `4-30` and `7-30`:

`Valid_DeepCAD = (Q_4-30 + Q_7-30) / (N_4-30 + N_7-30)`

For the reference runs, this is `5,916/8,192 = 72.2168%`.

## Reference Results

| Variant | Dataset | Valid |
| --- | --- | ---: |
| `4-30` | DeepCAD | 3,732/4,096 (91.1133%) |
| `7-30` | DeepCAD | 2,184/4,096 (53.3203%) |
| `abc-4-50` | ABC | 2,725/4,096 (66.5283%) |

The released ABC checkpoint was trained separately because the paper's original ABC diffusion checkpoint was unavailable. Its result should not be compared directly with the paper's ABC result.

Exact saved summaries are available for [`4-30`](../eval/reference_results/4-30.json), [`7-30`](../eval/reference_results/7-30.json), and [`abc-4-50`](../eval/reference_results/abc-4-50.json).

## VAE Reconstruction

VAE reconstruction uses processed validation data and ground-truth face and edge counts. Its command is documented in [TRAINING.md](TRAINING.md), and its Valid result is separate from unconditional generation.
