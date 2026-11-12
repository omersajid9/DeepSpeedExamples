# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Watchdog for the long-generation CB run (runs/longgen_2026-11-12).

Regenerates longgen md/csv every time one of the 12 raw cell JSONs lands;
exits when all 12 are fresh. Hard-capped at 5 h.

Run detached:  nohup <env>/bin/python scripts/watch_longgen_md.py \
    > runs/longgen_2026-11-12/watchdog.log 2>&1 &
"""

import subprocess
import sys
import time
from pathlib import Path

OPSD = Path("/u/msajid/projects/DeepSpeedExamples/training/opsd")
OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else OPSD / "runs" / "longgen_2026-11-12"
TASK = sys.argv[2] if len(sys.argv) > 2 else "6k"
RAW = OUT / "raw"
PY = "/work/nvme/bcjw/msajid/conda/opsd/bin/python"
DRIVER = OPSD / "scripts" / "run_cb_ablation.py"
INTERVAL_S = 30
MAX_RUNTIME_S = 5 * 3600

CELLS = [f"{m}_{TASK}_{lv}" for m in ("qwen3_4b", "qwen3_8b") for lv in
         ("A_eager", "B_single_graph", "C_multi_graph",
          "D_paged", "E_fused", "F_flashinfer_sampler")]


def mtime(name):
    p = RAW / f"{name}.json"
    return p.stat().st_mtime if p.exists() else 0.0


def regen():
    subprocess.run([str(PY), str(DRIVER), "--summary-only", "--out", str(OUT)])


def main():
    start = time.time()
    seen = {c: mtime(c) for c in CELLS}
    # On resume, cells already on disk before this watchdog started are
    # "done"; only wait for the ones still missing at start.
    pending = [c for c in CELLS if mtime(c) < start]
    print(f"longgen watchdog started {time.ctime()}; watching {len(CELLS)} cells, "
          f"{len(pending)} pending", flush=True)
    while True:
        time.sleep(INTERVAL_S)
        now = {c: mtime(c) for c in CELLS}
        if now != seen:
            seen = now
            regen()
            fresh = sum(1 for c in pending if now[c] >= start)
            print(f"{time.ctime()} regenerated md/csv; {fresh}/{len(pending)} "
                  f"pending cells fresh", flush=True)
        if all(now[c] >= start for c in pending):
            regen()
            print(f"{time.ctime()} all {len(pending)}/{len(pending)} pending cells "
                  f"fresh; final md/csv written; exiting", flush=True)
            break
        if time.time() - start > MAX_RUNTIME_S:
            print(f"{time.ctime()} max runtime reached; exiting with md/csv as-is", flush=True)
            break


if __name__ == "__main__":
    main()
