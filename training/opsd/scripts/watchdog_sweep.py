# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Post-sweep watchdog: re-run failed cells at a lower concurrency with more warmup."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_optimization_sweep as sweep


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="/workspace/DeepSpeedExamples/training/opsd/runs/sweep_2026-11-10")
    parser.add_argument("--opsd-dir", default="/workspace/DeepSpeedExamples/training/opsd")
    parser.add_argument("--fallback-c", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    root = Path(args.dir)
    manifest = load(root / "sweep_manifest.json")
    if not manifest:
        print("no manifest", flush=True)
        return
    env = dict(os.environ)
    env.setdefault("HF_HOME", "/dev/shm/hf_cache")
    env["PYTHONUNBUFFERED"] = "1"

    for key, cell in sorted(manifest.get("cells", {}).items()):
        if cell.get("status") == "ok" and cell.get("token_match"):
            continue
        model, workload = cell["model"], cell["workload"]
        c = min(int(cell.get("continuous_batch_size", args.fallback_c)), args.fallback_c)
        print(f"[watchdog] {key} status={cell.get('status')} -> retry C={c} warmup={args.warmup}", flush=True)
        results = {}
        for arm in ("baseline", "optimized"):
            stem = f"{model}__{workload}__{arm}"
            raw = root / "raw" / f"{stem}.json"
            log = root / "logs" / f"{stem}.log"
            cmd = sweep.build_command(SimpleNamespace(opsd_dir=args.opsd_dir), model, workload, arm, c, raw,
                                      args.warmup, sweep.WORKLOADS[workload]["iterations"])
            with open(log, "w") as fh:
                proc = subprocess.run(cmd, cwd=args.opsd_dir, env=env, stdout=fh, stderr=subprocess.STDOUT)
            data = sweep.load_json(raw)
            status = sweep.validate(data, arm)
            print(f"[watchdog] {stem} exit={proc.returncode} status={status}", flush=True)
            results[arm] = data
        b, o = results.get("baseline"), results.get("optimized")
        cell["continuous_batch_size"] = c
        cell["status"] = "ok"
        cell["token_match"] = None
        cell["peak_mb"] = {}
        for arm, data in results.items():
            cell[f"{arm}_sha"] = None if data is None else data.get("response_sha256")
            cell["peak_mb"][arm] = None if data is None else max(
                data.get("peak_memory_mb", 0.0), data.get("warmup_peak_memory_mb", 0.0))
        if b is not None and o is not None:
            cell["token_match"] = b.get("response_sha256") == o.get("response_sha256")
            if not cell["token_match"]:
                cell["status"] = "token_mismatch"
        for arm, data in results.items():
            if data is None:
                cell["status"] = "missing"
            elif arm == "optimized" and not (data.get("warmup_stats") or {}).get("graph_captures", 0):
                cell["status"] = "no_graph_captures"
        manifest["cells"][key] = cell
        (root / "sweep_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("watchdog complete", flush=True)


if __name__ == "__main__":
    main()
