#!/usr/bin/env bash
# One-time B300 environment setup for the paged/fused-kernel ablation.
#
# Prereqs (placed by b300/transfer_to_b300.sh, run from Delta):
#   ~/pfuse/projects/DeepSpeed                    (incl. perf/paged_fused_kernels)
#   ~/pfuse/projects/DeepSpeedExamples/training/opsd
#   ~/pfuse/models/Qwen3.5-9B
#   ~/pfuse/hf_cache                              (optional: 4B model cache)
#
# This script (idempotent; re-run after a failed step):
#   1. installs Miniforge under ~/pfuse/conda (if conda is missing)
#   2. builds the 'opsd' env   — mirror of the Delta opsd env:
#        torch 2.15.0 (cu13) + transformers 5.22.2 + flashinfer-python 0.7.4
#        + flash-linear-attention + causal-conv1d + deepspeed-kernels
#        + DeepSpeed editable from the perf branch
#   3. builds the 'vllm33' env — vLLM 0.33.0 reference
#   4. installs nvcc 13.2 from PyPI (no root needed) and assembles a
#      CUDA_HOME shim at ~/pfuse/cuda-shim
#   5. verifies: nvcc compiles sm_103 and runs on the B300, torch sees the
#      GPUs, deepspeed imports from the local branch, vllm imports
#
# The 4B model is NOT transferred by default: if ~/pfuse/hf_cache lacks it,
# the first run downloads Qwen/Qwen3.5-4B from HuggingFace (~10 GB, needs
# internet). Set PFUSE_B300_4B=1 in the transfer script to push the cache
# instead.
set -uo pipefail

BASE="${PFUSE_BASE:-$HOME/pfuse}"
ENV_ROOT="$BASE/conda"
OPS_ENV="$ENV_ROOT/envs/opsd"
VLLM_ENV="$ENV_ROOT/envs/vllm33"
OPS_PY="$OPS_ENV/bin/python"
VLLM_PY="$VLLM_ENV/bin/python"
DS_REPO="$BASE/projects/DeepSpeed"
SHIM="$BASE/cuda-shim"

die() { echo "FATAL: $*" >&2; exit 1; }

echo "=== B300 setup: BASE=$BASE $(date -Is) ==="

# --- 0. prereqs -------------------------------------------------------------
command -v gcc  >/dev/null || die "gcc not found (install build-essential)"
command -v g++ >/dev/null || die "g++ not found (install build-essential)"
command -v git >/dev/null || die "git not found"
[ -d "$DS_REPO/.git" ] || die "$DS_REPO missing — run b300/transfer_to_b300.sh first"
git -C "$DS_REPO" show-ref --quiet refs/heads/perf/paged_fused_kernels \
  || die "perf/paged_fused_kernels not in $DS_REPO — re-run the transfer"
[ -d "$BASE/models/Qwen3.5-9B" ] || echo "WARN: 9B model missing (9B cells will fail)"
nvidia-smi -L || die "nvidia-smi failed — are you on the B300 box?"

# --- 1. conda ---------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
  [ -x "$ENV_ROOT/bin/conda" ] || {
    echo "installing Miniforge -> $ENV_ROOT"
    curl -fsSL -o /tmp/Miniforge3.sh \
      https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
      || die "Miniforge download failed (no internet?)"
    bash /tmp/Miniforge3.sh -b -p "$ENV_ROOT"
    rm -f /tmp/Miniforge3.sh
  }
  export PATH="$ENV_ROOT/bin:$PATH"
  # shellcheck disable=SC1091
  source "$ENV_ROOT/etc/profile.d/conda.sh"
fi
command -v conda >/dev/null || die "conda still not available after install"

# --- 2. opsd env ------------------------------------------------------------
if [ ! -x "$OPS_PY" ]; then
  echo "creating opsd env -> $OPS_ENV"
  conda create -y -p "$OPS_ENV" python=3.12
fi
"$OPS_PY" -m pip install -q -U pip wheel
SP="$("$OPS_PY" -c 'import site; print(site.getsitepackages()[0])')"
echo "opsd site-packages: $SP"

echo "installing core packages (torch 2.15 cu13, transformers, flashinfer, ...)"
"$OPS_PY" -m pip install \
  "torch==2.15.0" "transformers==5.22.2" "flashinfer-python==0.7.4" \
  "flash-linear-attention==0.6.0" "datasets" "einops" "safetensors" \
  || die "core pip install failed"

# --- 3. nvcc 13.2 from PyPI + CUDA_HOME shim (no root needed) ---------------
echo "installing nvcc 13.2 (nvidia-cuda-nvcc from PyPI)"
"$OPS_PY" -m pip install "nvidia-cuda-nvcc==13.2.86" \
  || "$OPS_PY" -m pip install nvidia-cuda-nvcc \
  || die "pip install nvidia-cuda-nvcc failed"

echo "assembling CUDA_HOME shim -> $SHIM"
mkdir -p "$SHIM/bin"
NVCC_BIN="$("$OPS_PY" - <<'PY'
import site
from glob import glob
hits = []
for sp in site.getsitepackages():
    hits += glob(sp + "/nvidia/**/nvcc", recursive=True)
print(hits[0] if hits else "")
PY
)"
[ -n "$NVCC_BIN" ] && [ -e "$NVCC_BIN" ] || die "nvcc not found under site-packages/nvidia"
NVCC_DIR=$(dirname "$NVCC_BIN")
for t in "$NVCC_DIR"/*; do ln -sf "$t" "$SHIM/bin/$(basename "$t")"; done

CU_INC="$("$OPS_PY" - <<'PY'
import site
from glob import glob
hits = []
for sp in site.getsitepackages():
    hits += glob(sp + "/nvidia/**/include/cuda.h", recursive=True)
print(hits[0] if hits else "")
PY
)"
[ -n "$CU_INC" ] || die "cuda.h not found under site-packages/nvidia (CCCL headers missing?)"
ln -sfn "$(dirname "$CU_INC")" "$SHIM/include"

CU_LIB="$("$OPS_PY" - <<'PY'
import site
from glob import glob
hits = []
for sp in site.getsitepackages():
    hits += glob(sp + "/nvidia/**/libcudart.so*", recursive=True)
print(hits[0] if hits else "")
PY
)"
[ -n "$CU_LIB" ] && ln -sfn "$(dirname "$CU_LIB")" "$SHIM/lib64"

# --- 4. CUDA-extension packages (need nvcc) + DeepSpeed editable -----------
echo "building causal-conv1d against the shim"
CUDA_HOME="$SHIM" PATH="$SHIM/bin:$PATH" \
  "$OPS_PY" -m pip install "causal-conv1d==1.7.0" \
  || echo "WARN: causal-conv1d build failed (GDN layers may misbehave)"

echo "installing deepspeed-kernels (prebuilt wheel)"
"$OPS_PY" -m pip install "deepspeed-kernels==0.0.1.dev1698255861" \
  || "$OPS_PY" -m pip install deepspeed-kernels

echo "installing DeepSpeed editable from $DS_REPO (perf branch)"
git -C "$DS_REPO" checkout -q perf/paged_fused_kernels
(cd "$DS_REPO" && DS_BUILD_OPS=0 "$OPS_PY" -m pip install -q -e . \
  ) || die "deepspeed editable install failed"

# --- 5. vllm33 env ----------------------------------------------------------
if [ ! -x "$VLLM_PY" ]; then
  echo "creating vllm33 env -> $VLLM_ENV"
  conda create -y -p "$VLLM_ENV" python=3.12
fi
"$VLLM_PY" -m pip install -q -U pip wheel
echo "installing vllm==0.33.0 (pulls torch 2.14 + deps; ~2 GB)"
"$VLLM_PY" -m pip install "vllm==0.33.0" || die "vllm install failed"

# --- 6. verification --------------------------------------------------------
echo "=== verification ==="
"$SHIM/bin/nvcc" --version | tail -2
SMOKE_CU=$(mktemp -d)/smoke.cu
cat > "$SMOKE_CU" <<'EOF'
__global__ void k() {}
int main() { k<<<1, 1>>>(); cudaDeviceSynchronize(); return 0; }
EOF
if CUDA_HOME="$SHIM" "$SHIM/bin/nvcc" -arch=sm_103 -o "$SMOKE_CU.bin" "$SMOKE_CU" \
   && "$SMOKE_CU.bin"; then
  echo "OK: nvcc compiles sm_103 and the binary runs on the B300"
else
  echo "WARN: sm_103 compile/run test failed — try:"
  echo "  $OPS_PY -m pip install -U nvidia-cuda-nvcc && re-run this script"
fi
"$OPS_PY" -c "import torch; print('OK torch', torch.__version__,
        '| GPUs:', torch.cuda.device_count(),
        '|', torch.cuda.get_device_name(0))" \
  || die "torch CUDA smoke failed"
CUDA_HOME="$SHIM" PATH="$SHIM/bin:$PATH" \
  "$OPS_PY" -c "import flashinfer; print('OK flashinfer', flashinfer.__version__)" \
  || echo "WARN: flashinfer import failed (JIT may still work at cell time)"
PYTHONPATH="$DS_REPO" CUDA_HOME="$SHIM" PATH="$SHIM/bin:$PATH" \
  "$OPS_PY" -c "import deepspeed; print('OK deepspeed', deepspeed.__version__,
         '->', deepspeed.__file__)" \
  || die "deepspeed import failed (editable install?)"
"$VLLM_PY" -c "import vllm; print('OK vllm', vllm.__version__)" \
  || die "vllm import failed"

cat <<EOF

=== setup complete ===
  next:  bash b300/run_smoke.sh      (6-cell smoke, ~30-60 min incl. JIT)
  then:  bash b300/run_matrix.sh     (full 60 cells, resumable)
  env:   source b300/b300.env        (or the run scripts source it for you)
EOF
