# Data Preparation

Training reads processed B-rep PKL files selected by the tracked dataset manifests. Processed PKLs are not distributed with this repository.

## Data Sources

- ABC: [ABC Dataset](https://deep-geometry.github.io/abc-dataset/). The released preprocessor converts its STEP files to the required PKL format.
- DeepCAD: [rundiwu/DeepCAD](https://github.com/rundiwu/DeepCAD). Supply processed PKLs whose relative paths match `datasets/manifests/deepcad-v1.json`; this repository does not include a raw DeepCAD-to-PKL converter.

The upstream datasets remain subject to their own licenses and access terms.

## Fixed Splits

| Dataset | Train | Validation | Test | Selected PKLs | Manifest |
| --- | ---: | ---: | ---: | ---: | --- |
| ABC | 186,148 | 10,341 | 10,343 | 206,832 | `datasets/manifests/abc-v1.json` |
| DeepCAD | 83,612 | 6,709 | 0 | 90,321 | `datasets/manifests/deepcad-v1.json` |

Training uses only the declared `train` and `val` splits. The paper reports 83,611 DeepCAD training models; repository runs use the 83,612 entries fixed by the tracked manifest.

## Validate Processed Data

Python pickle can execute code while loading. Use only PKLs that you created or obtained from a trusted source.

```bash
python -m tools.validate_processed_dataset \
  --data-root /data/hifi-brep/abc \
  --manifest datasets/manifests/abc-v1.json
```

The validator checks manifest and path completeness only: it verifies that every
selected path exists and fails when a required file is missing. It does not load
or validate PKL contents.

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

The preprocessor may produce more PKLs than the fixed subset. Training still selects samples exclusively through `datasets/manifests/abc-v1.json`.
