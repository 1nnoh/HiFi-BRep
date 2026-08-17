# Environment

Use the `hifi-brep` conda environment with the following versions:

| Component | Version |
| --- | --- |
| Python | 3.10.12 |
| PyTorch | 2.5.1+cu124 |
| CUDA runtime | 12.4 |
| pythonocc-core / OCCT | 7.5.1 / 7.5.1 |
| OCCWL | 3.0.0 |
| diffusers | 0.39.0 |

## Create the Environment

```bash
conda create -n hifi-brep python=3.10
conda activate hifi-brep

conda install -c lambouj -c conda-forge \
  occwl=3.0.0 pythonocc-core=7.5.1

python -m pip install \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124

export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
python -m pip install --no-build-isolation -r requirements.txt
```

Adjust `CUDA_HOME` if the CUDA 12.4 toolkit is installed elsewhere; it is used when compiling `chamferdist`.
