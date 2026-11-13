#!/bin/bash
# vLLM 4B profiling runs (nsys + torch profiler via vLLM's built-in profiler),
# 3072 prompt -> 1024 new, C=4. Runs after the DS profiling releases the GPU.
#
# Build env: vLLM 0.33's EngineCore subprocess JIT-builds FlashInfer's sampling
# module at startup, needing ninja + nvcc. Same CPATH trap as the DS side: login
# modules set CPATH=<HPC-SDK>/cuda/13.2/include (searched before -I/-isystem), so
# cooperative_groups.h always resolves to the 13.2 tree, which lacks the cuda/std
# (CCCL) headers its CG includes -> add the vllm env's flashinfer-bundled CCCL to
# CPATH. 13.2 nvcc + CPATH-13.2 + bundled CCCL is the consistent combo.
#   ninja: /work/nvme/bcjw/msajid/conda/opsd/bin/ninja (only ninja on the node)
# First run pays a one-time flashinfer JIT build (minutes); later runs hit the
# cache under ~/.cache/flashinfer/0.7.2.post1/.
cd /u/msajid/projects/DeepSpeedExamples/training/opsd || exit 1
V=/u/msajid/vllm_env/bin/python
OUT=runs/profiling_2026-11-12
export HF_HOME=/work/nvme/bcjw/msajid/hf_cache
export VLLM_CACHE_ROOT=/u/msajid/vllm_cache
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2
export CPATH="$CUDA_HOME/include:/u/msajid/vllm_env/lib/python3.14/site-packages/flashinfer/data/cccl/libcudacxx/include"
export PATH=/work/nvme/bcjw/msajid/conda/opsd/bin:$PATH
COMMON4B="--model Qwen/Qwen3-4B --data data/longbench_20k.jsonl --batch-size 4 --max-prompt-length 3072 --max-new-tokens 1024 --temperature 0.7 --top-p 0.9 --top-k -1 --seed 0 --warmup 1 --iterations 1"

echo "=== waiting for GPU to free up (max 45 min)"
for i in $(seq 1 180); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1 | tr -dc 0-9)
  busy=$(ps -eo args | grep -c "[b]ench_continuous\|[n]sys profile")
  [ "${used:-99999}" -lt 2000 ] && [ "$busy" -eq 0 ] && break
  sleep 15
done
echo "GPU mem now: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

echo "=== [3/4] nsys: vLLM 4B $(date +%H:%M:%S)"
nsys profile --force-overwrite=true -t cuda \
  -o $OUT/vllm4b/nsys_vllm4b $V benchmarks/bench_vllm_offline.py $COMMON4B \
  --output $OUT/vllm4b/result.json
rc=$?
[ $rc -ne 0 ] && { echo "NSYS RUN FAILED rc=$rc"; exit $rc; }

echo "=== [4/4] torch profiler: vLLM 4B $(date +%H:%M:%S)"
$V benchmarks/bench_vllm_offline.py $COMMON4B \
  --output $OUT/vllm4b/result_torch.json --torch-profile $OUT/vllm4b/torch
rc=$?
[ $rc -ne 0 ] && { echo "TORCH RUN FAILED rc=$rc"; exit $rc; }

echo "=== stats $(date +%H:%M:%S)"
nsys stats --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum \
  -o $OUT/vllm4b/nsys_vllm4b $OUT/vllm4b/nsys_vllm4b.nsys-rep >/dev/null 2>&1
echo "=== VLLM PROFILING DONE $(date +%H:%M:%S)"
