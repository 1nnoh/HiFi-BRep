#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: bash tools/test_deepcad_stability.sh [DATA_ROOT [OUTPUT_DIR]] [--dry-run]"
    echo "Defaults: auto-detect datasets/deepcad; create a new run under outputs/runs."
    echo "Environment: CUDA_VISIBLE_DEVICES=0, MAX_TRAIN_STEPS=200000, PYTHON=python"
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi
dry_run=false
arguments=()
for argument in "$@"; do
    if [[ $argument == --dry-run ]]; then
        dry_run=true
    else
        arguments+=("$argument")
    fi
done
if (( ${#arguments[@]} > 2 )); then
    usage >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "$script_dir/.." && pwd)
default_data_root="$repository_root/datasets/deepcad"
if [[ ! -d $default_data_root ]]; then
    # The local Fix checkout can reuse the reference checkout's data read-only.
    default_data_root="$repository_root/../../HiFi-BRep/datasets/deepcad"
fi
# Resolve explicit caller-relative paths before moving to the repository root.
data_root=$(realpath -e -- "${arguments[0]:-${HIFI_BREP_DATA_ROOT:-$default_data_root}}")
default_output="$repository_root/outputs/runs/deepcad-parallel-layerscale-$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)-$$"
run_output=$(realpath -m -- "${arguments[1]:-$default_output}")
if [[ ! -d $data_root ]]; then
    echo "Data root is not a directory: $data_root" >&2
    exit 1
fi
if [[ -e $run_output ]]; then
    echo "Use a new output directory: $run_output" >&2
    exit 1
fi

cd -- "$repository_root"
python_bin=${PYTHON:-python}
max_steps=${MAX_TRAIN_STEPS:-200000}
if [[ ! $max_steps =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_TRAIN_STEPS must be a positive integer." >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
process_count=${#gpu_ids[@]}

# Only the temporary recipe enables the new behavior; released defaults stay intact.
run_config=$(mktemp "${TMPDIR:-/tmp}/deepcad-stability.XXXXXX.yaml")
trap 'rm -f -- "$run_config"' EXIT
sed \
    -e 's/^  bidirectional_cross_update:.*/  bidirectional_cross_update: parallel/' \
    -e 's/^  encoder_layer_scale:.*/  encoder_layer_scale: true/' \
    -e 's/^  encoder_bimodal_layer_scale_init:.*/  encoder_bimodal_layer_scale_init: 0.1/' \
    -e 's/^  encoder_latent_layer_scale_init:.*/  encoder_latent_layer_scale_init: 0.01/' \
    configs/training/vae_deepcad.yaml > "$run_config"
"$python_bin" -c '
import sys, yaml
with open(sys.argv[1]) as stream:
    model = yaml.safe_load(stream)["model"]
assert model["bidirectional_cross_update"] == "parallel"
assert model["encoder_layer_scale"] is True
assert model["encoder_bimodal_layer_scale_init"] == 0.1
assert model["encoder_latent_layer_scale_init"] == 0.01
' "$run_config"

command=("$python_bin" -m accelerate.commands.launch --num_machines 1
         --num_processes "$process_count")
if (( process_count > 1 )); then
    command+=(--multi_gpu)
fi
command+=(-m train.vae --config "$run_config" --data-root "$data_root"
          --output-dir "$run_output" --max-train-steps "$max_steps")
echo "DeepCAD: parallel + Encoder LayerScale (0.1 / 0.01), GPUs=$CUDA_VISIBLE_DEVICES"
echo "Target: $max_steps updates; previous parallel-only failure: 125382 updates."
echo "The trainer saves the effective recipe in OUTPUT_DIR/run_config.json."
printf 'Launch: '
printf '%q ' "${command[@]}"
printf '\n'
if [[ $dry_run == false ]]; then
    "${command[@]}"
fi
