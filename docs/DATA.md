# Data Preparation

Processed ABC and DeepCAD training data are available from the [HiFi-BRep ModelScope Dataset](https://www.modelscope.cn/datasets/innohou/HiFi-BRep).

| Dataset | Train | Validation | Test | Total | Download | Extracted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ABC | 186,148 | 10,341 | 10,343 | 206,832 | 19.17 GB | 85.06 GB |
| DeepCAD | 83,612 | 6,709 | 0 | 90,321 | 5.87 GB | 29.12 GB |

## Download and Prepare

Set `dataset=abc` to use ABC instead of DeepCAD. Replace `/path/to/HiFi-BRep` with the cloned repository path.

```bash
cd /path/to/HiFi-BRep
set -euo pipefail

dataset=deepcad
dataset_repo=../HiFi-BRep-Dataset
data_root=/data/hifi-brep

python -m pip install modelscope-hub==0.2.0
ms-hub download innohou/HiFi-BRep \
  --repo-type dataset \
  --include \
    "data/${dataset}/*.tar.gz" \
    "manifests/${dataset}-v1.json" \
    "release_manifest.json" \
    "tools/extract_dataset.py" \
  --local-dir "$dataset_repo"

python "$dataset_repo/tools/extract_dataset.py" \
  --repo-root "$dataset_repo" \
  --subset "$dataset" \
  --output "$data_root"

python -m tools.validate_processed_dataset \
  --data-root "$data_root/$dataset" \
  --manifest "datasets/manifests/${dataset}-v1.json"
```

Only load PKL files downloaded from the canonical ModelScope repository. After validation, continue with [TRAINING.md](TRAINING.md) and use `$data_root/$dataset` as `--data-root`.

## Build ABC PKLs from STEP

```bash
python -m preprocess.steps \
  --input-root /data/abc-step \
  --output-root /data/hifi-brep/abc \
  --layout abc \
  --max-face 50 \
  --workers 16 \
  --resume
```

The preprocessor may produce additional PKLs; training selects only paths listed in `datasets/manifests/abc-v1.json`.
