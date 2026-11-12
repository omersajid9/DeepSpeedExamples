# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Watchdog: regenerate the cb_ablation md/csv every time a raw cell JSON lands.

Polls raw/*.json; when the file set (name -> mtime) changes, re-runs
`run_cb_ablation.py --summary-only`, which rewrites cb_ablation.md and
cb_ablation.csv from every raw JSON present. Broken/half-written JSONs are
skipped by the summarizer, so an interrupted write never stalls the loop.

Run detached:  nohup <env>/bin/python scripts/watch_cb_ablation_md.py \
    > runs/cb_ablation_2026-11-12/watchdog.log 2>&1 &
"""

import subprocess
import time
from pathlib import Path

OPSD = Path(__file__).resolve().parent.parent
PY = "/work/nvme/bcjw/msajid/conda/opsd/bin/python"
OUT = OPSD / "runs" / "cb_ablation_2026-11-12"
RAW = OUT / "raw"
DRIVER = OPSD / "scripts" / "run_cb_ablation.py"
INTERVAL_S = 15
TOTAL = 24


def state():
    st = {}
    for p in RAW.glob("*.json"):
        try:
            st[p.name] = p.stat().st_mtime_ns
        except OSError:
            pass
    return st


def log(msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}\n"
    with open(OUT / "watchdog.log", "a") as fh:
        fh.write(line)
    print(line, end="", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prev = state()
    last_change = time.time()
    log(f"watchdog started; {len(prev)}/{TOTAL} cells present")
    while True:
        cur = state()
        if cur != prev:
            last_change = time.time()
            added = sorted(set(cur) - set(prev))
            try:
                proc = subprocess.run([PY, str(DRIVER), "--summary-only"], cwd=OPSD,
                                      capture_output=True, text=True)
                tail = (proc.stdout or "").strip().splitlines()
                log(f"state change {added} (rc={proc.returncode}) {tail[-1] if tail else ''}")
            except Exception as exc:  # keep watching no matter what
                log(f"summary regeneration failed: {exc}")
            prev = state()
        done = sum(1 for m in ("qwen3_4b", "qwen3_8b") for t in ("6k", "20k")
                   for lv in ("A_eager", "B_single_graph", "C_multi_graph", "D_paged",
                              "E_fused", "F_flashinfer_sampler")
                   if (RAW / f"{m}_{t}_{lv}.json").exists())
        if done >= TOTAL:
            log(f"all {TOTAL}/{TOTAL} cells present; watchdog exiting")
            return
        if time.time() - last_change > 45 * 60:
            log(f"no new cells for 45 min ({done}/{TOTAL}); watchdog exiting "
                f"(driver likely finished or died)")
            return
        time.sleep(INTERVAL_S)


if __name__ == "__main__":
    main()
