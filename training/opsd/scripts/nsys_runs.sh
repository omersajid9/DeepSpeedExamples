#!/bin/bash
# nsys capture for the A100 profiling study. GPU must be free.
# Usage: bash nsys_runs.sh   (runs all four captures, then stats)
set -uo pipefail
cd "$(dirname "$0")/.."
SP=/work/nvme/bcjw/msajid/conda/opsd/lib/python3.12/site-packages
RAW=runs/a100_2026-11-11/profile_raw
NSYS=/usr/local/bin/nsys
export CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/u/msajid/cuda-12.9-shim
export CC=/opt/rh/gcc-toolset-14/root/usr/bin/gcc CXX=/opt/rh/gcc-toolset-14/root/usr/bin/g++
export HF_HOME=/u/msajid/hf_cache PYTHONUNBUFFERED=1 TMPDIR=/u/msajid/tmp_bench
export TORCHINDUCTOR_CACHE_DIR=/u/msajid/torchinductor_cache VLLM_CACHE_ROOT=/u/msajid/vllm_cache
export CPATH=$SP/nvidia/cu13/include:$SP/nvidia/cu13/include/cccl:$SP/nvidia/curand/include
export FLASHINFER_EXTRA_CUDAFLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
export PATH=/work/nvme/bcjw/msajid/conda/opsd/bin:/u/msajid/cuda-12.9-shim/bin:$PATH

capture() { # name cmd...
  local name="$1"; shift
  echo "=== nsys $name start $(date +%T)"
  "$NSYS" profile --force-overwrite true -t cuda,cublas,cudnn -o "$RAW/$name" -- "$@" > "$RAW/nsys_$name.log" 2>&1
  echo "=== nsys $name done rc=$? $(date +%T)"
}

capture ds_6k /work/nvme/bcjw/msajid/conda/opsd/bin/python benchmarks/profile_torch_rollout.py \
  --data data/longbench_20k.jsonl --workload 6k --no-torch-profile --out "$RAW/nsys_ds_6k"
capture ds_20k /work/nvme/bcjw/msajid/conda/opsd/bin/python benchmarks/profile_torch_rollout.py \
  --data data/longbench_20k.jsonl --workload 20k --no-torch-profile --out "$RAW/nsys_ds_20k"

export VLLM_ENABLE_V1_MULTIPROCESSING=0
capture vllm_6k /u/msajid/vllm_env/bin/python benchmarks/profile_torch_vllm.py \
  --data data/longbench_20k.jsonl --workload 6k --no-torch-profile --out "$RAW/nsys_vllm_6k"
capture vllm_20k /u/msajid/vllm_env/bin/python benchmarks/profile_torch_vllm.py \
  --data data/longbench_20k.jsonl --workload 20k --no-torch-profile --out "$RAW/nsys_vllm_20k"

# stats
for n in ds_6k ds_20k vllm_6k vllm_20k; do
  echo "=== stats $n"
  "$NSYS" stats -r cuda_gpu_kern_sum,cuda_api_sum "$RAW/$n.nsys-rep" > "$RAW/$n.kernsum.txt" 2>&1 || true
done
echo ALL_NSYS_DONE
