# Data Preparation

Training reads processed B-rep PKL files selected by the manifests tracked in this code repository. The corresponding archive shards are distributed separately in the [HiFi-BRep ModelScope Dataset](https://www.modelscope.cn/datasets/innohou/HiFi-BRep).

## Processed Subsets

| Subset | Train | Validation | Test | Total | Download | Extracted | Same-volume peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ABC v1 | 186,148 | 10,341 | 10,343 | 206,832 | 19.17 GB | 85.06 GB | 108.48 GB |
| DeepCAD-aligned v1 | 83,612 | 6,709 | 0 | 90,321 | 5.87 GB | 29.12 GB | 36.45 GB |

ABC v1 contains processed geometry from the [ABC Dataset](https://deep-geometry.github.io/abc-dataset/). DeepCAD-aligned v1 is the fixed historical HiFi-BRep DeepCAD ID split applied to ABC-derived processed B-reps; it is not the official DeepCAD CAD-sequence archive. The upstream geometry remains subject to the rights and terms of its original sources.

The two subsets are independently downloadable. Their archive shards are not binary parts that need concatenation: extract every shard for one subset into the same parent directory.

## Download One Subset

Install the ModelScope Hub CLI:

```bash
python -m pip install modelscope-hub==0.2.0
```

Download DeepCAD-aligned v1:

```bash
ms-hub download innohou/HiFi-BRep \
  --repo-type dataset \
  --include "README.md" "data/deepcad/*.tar.gz" \
  --local-dir ../HiFi-BRep-Dataset
```

Download ABC v1 instead:

```bash
ms-hub download innohou/HiFi-BRep \
  --repo-type dataset \
  --include "README.md" "data/abc/*.tar.gz" \
  --local-dir ../HiFi-BRep-Dataset
```

To download the complete Dataset with Git LFS:

```bash
git lfs install
git clone https://www.modelscope.cn/datasets/innohou/HiFi-BRep.git \
  ../HiFi-BRep-Dataset
```

## Extract the Shards

The output parent must not already contain the selected subset directory. For DeepCAD-aligned v1:

```bash
mkdir -p /data/hifi-brep
for archive in ../HiFi-BRep-Dataset/data/deepcad/*.tar.gz; do
  tar -xzf "$archive" -C /data/hifi-brep --no-same-owner || exit 1
done
```

This creates `/data/hifi-brep/deepcad`. For ABC v1, replace `deepcad` with `abc`; the result is `/data/hifi-brep/abc`.

Python pickle can execute code while loading. Download the PKLs only from the canonical Dataset repository and do not load modified copies from an untrusted source.

## Validate Against the Canonical Manifest

Run the validator from the HiFi-BRep code repository:

```bash
python -m tools.validate_processed_dataset \
  --data-root /data/hifi-brep/deepcad \
  --manifest datasets/manifests/deepcad-v1.json
```

For ABC v1, use `/data/hifi-brep/abc` and `datasets/manifests/abc-v1.json`. The validator checks that every selected path exists. The lean archive distribution does not install a separate provenance file, so `provenance_verified` is expected to be `false`; a missing required PKL still fails validation.

After validation, follow [TRAINING.md](TRAINING.md) and pass the selected subset directory as `--data-root`.

## Build ABC PKLs from STEP

The released preprocessor remains available for researchers who prefer to process official ABC STEP files locally:

```bash
python -m preprocess.steps \
  --input-root /data/abc-step \
  --output-root /data/hifi-brep/abc \
  --layout abc \
  --max-face 50 \
  --workers 16 \
  --resume
```

The preprocessor may produce more PKLs than ABC v1. Training still selects samples exclusively through `datasets/manifests/abc-v1.json`.
