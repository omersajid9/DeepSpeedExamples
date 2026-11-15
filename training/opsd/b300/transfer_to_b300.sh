#!/usr/bin/env bash
# Push everything the B300 box needs, from a machine that can both read the
# Delta paths below AND ssh into the B300 box (usually the Delta login node).
#
# usage:
#   PFUSE_B300_DEST=ubuntu@<b300-host> bash b300/transfer_to_b300.sh
#
# Remote layout (under <dest>/~/pfuse):
#   pfuse/projects/DeepSpeed                  (full clone, incl. perf branch + .git)
#   pfuse/projects/DeepSpeedExamples/training/opsd
#   pfuse/models/Qwen3.5-9B
#   pfuse/hf_cache/...                        (only with PFUSE_B300_4B=1)
#
# ~20 GB total (9B model + repos). If the Delta login node cannot reach the
# B300 box directly, do this in two hops via your laptop (rsync to laptop,
# then laptop -> box) — the remote layout is what matters.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # opsd root

DEST="${PFUSE_B300_DEST:?set PFUSE_B300_DEST=user@host (ssh-reachable from here)}"
echo "=== transferring to $DEST:~/pfuse/ ==="

echo "-- DeepSpeed repo (with perf/paged_fused_kernels branch) --"
rsync -az --info=progress2 \
  /u/msajid/projects/DeepSpeed/ "$DEST:pfuse/projects/DeepSpeed/"

echo "-- opsd project (repos/scripts/data; runs/ excluded) --"
rsync -az --info=progress2 \
  --exclude 'runs/' --exclude '__pycache__/' --exclude '*.pyc' \
  /u/msajid/projects/DeepSpeedExamples/training/opsd/ \
  "$DEST:pfuse/projects/DeepSpeedExamples/training/opsd/"

echo "-- 9B model (~19 GB) --"
rsync -az --info=progress2 \
  /work/nvme/bcjw/runguol2/models/Qwen3.5-9B/ \
  "$DEST:pfuse/models/Qwen3.5-9B/"

# 4B model: by default the box downloads it from HuggingFace on first run
# (needs internet, ~10 GB). If the box is offline, push the cache instead:
#   PFUSE_B300_4B=1 PFUSE_B300_DEST=... bash b300/transfer_to_b300.sh
# The HF cache uses a SHARED blob store (hub/blobs/**) that the model dir
# symlinks into, so we resolve the symlinks and push only the blobs this
# model references.
if [ "${PFUSE_B300_4B:-0}" = "1" ]; then
  echo "-- 4B model HF cache (shared-blob aware) --"
  python3 - "$DEST" <<'PY'
import os, shlex, subprocess, sys
from pathlib import Path

dest = sys.argv[1]
base = Path("/work/nvme/bcjw/msajid/hf_cache/hub")
model = base / "models--Qwen--Qwen3.5-4B"
blobs = set()
for p in (model / "snapshots").rglob("*"):
    if p.is_symlink():
        tgt = os.path.realpath(p)
        if str(tgt).startswith(str(base / "blobs")):
            blobs.add(tgt)
rsync = subprocess.run(["rsync", "-az", str(model), f"{dest}:pfuse/hf_cache/hub/"])
assert rsync.returncode == 0, "model dir rsync failed"
for b in sorted(blobs):
    r = subprocess.run(["rsync", "-az", str(b),
                        f"{dest}:pfuse/hf_cache/{os.path.relpath(b, base)}"])
    assert r.returncode == 0, f"blob rsync failed: {b}"
    print(f"  pushed {os.path.relpath(b, base)}")
print(f"4B cache: model dir + {len(blobs)} shared blob(s)")
PY
fi

cat <<EOF

=== transfer complete ===
On the B300 box:
  cd ~/pfuse/projects/DeepSpeedExamples/training/opsd
  bash b300/setup_b300.sh       # installs both conda envs + nvcc; verifies
  bash b300/run_smoke.sh        # 6-cell smoke (JIT cold start + all paths)
  bash b300/run_matrix.sh       # full 60-cell matrix (resumable)
EOF
