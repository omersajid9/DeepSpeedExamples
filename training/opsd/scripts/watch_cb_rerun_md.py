# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Watchdog for the lazy-init-fix CB rerun (2026-11-12).

Regenerates cb_ablation.md/csv every time one of the 12 rerun cell JSONs
changes, and exits when the 8 still-pending cells are all fresh (the other
4 were already redone before this watchdog started). Hard-capped at 3 h.

Run detached:  nohup <env>/bin/python scripts/watch_cb_rerun_md.py \
    > runs/cb_ablation_2026-11-12/watchdog_rerun2.log 2>&1 &
"""

import subprocess
import time
from pathlib import Path

OPSD = Path("/u/msajid/projects/DeepSpeedExamples/training/opsd")
OUT = OPSD / "runs" / "cb_ablation_2026-11-12"
RAW = OUT / "raw"
PY = "/work/nvme/bcjw/msajid/conda/opsd/bin/python"
DRIVER = OPSD / "scripts" / "run_cb_ablation.py"
INTERVAL_S = 20
MAX_RUNTIME_S = 3 * 3600

# The 8 cells the current rerun still has to write (4 already redone:
# qwen3_4b_6k D/E/F + qwen3_4b_20k D).
PENDING = [
    "qwen3_4b_20k_E_fused",
    "qwen3_4b_20k_F_flashinfer_sampler",
    "qwen3_8b_6k_D_paged",
    "qwen3_8b_6k_E_fused",
    "qwen3_8b_6k_F_flashinfer_sampler",
    "qwen3_8b_20k_D_paged",
    "qwen3_8b_20k_E_fused",
    "qwen3_8b_20k_F_flashinfer_sampler",
]
ALL = [
    "qwen3_4b_6k_D_paged", "qwen3_4b_6k_E_fused", "qwen3_4b_6k_F_flashinfer_sampler",
    "qwen3_4b_20k_D_paged", "qwen3_4b_20k_E_fused", "qwen3_4b_20k_F_flashinfer_sampler",
    "qwen3_8b_6k_D_paged", "qwen3_8b_6k_E_fused", "qwen3_8b_6k_F_flashinfer_sampler",
    "qwen3_8b_20k_D_paged", "qwen3_8b_20k_E_fused", "qwen3_8b_20k_F_flashinfer_sampler",
]


def mtime(name):
    p = RAW / f"{name}.json"
    return p.stat().st_mtime if p.exists() else 0.0


def regen():
    subprocess.run([str(PY), str(DRIVER), "--summary-only"], cwd=OPSD)


def main():
    start = time.time()
    seen = {t: mtime(t) for t in ALL}
    print(f"rerun watchdog started {time.ctime()}; watching {len(ALL)} cells, "
          f"{len(PENDING)} pending", flush=True)
    while True:
        time.sleep(INTERVAL_S)
        now = {t: mtime(t) for t in ALL}
        if now != seen:
            seen = now
            regen()
            fresh = sum(1 for t in PENDING if now[t] >= start)
            print(f"{time.ctime()} regenerated md/csv; {fresh}/{len(PENDING)} pending cells fresh",
                  flush=True)
        if all(now[t] >= start for t in PENDING):
            regen()
            print(f"{time.ctime()} all {len(PENDING)}/{len(PENDING)} pending cells fresh; "
                  f"final md/csv written; exiting", flush=True)
            break
        if time.time() - start > MAX_RUNTIME_S:
            print(f"{time.ctime()} max runtime reached; exiting with md/csv as-is", flush=True)
            break


if __name__ == "__main__":
    main()
