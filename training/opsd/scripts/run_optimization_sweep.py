# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Run the OPSD baseline-vs-optimized rollout sweep with a retrying watchdog."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORKLOADS = {
    "20k": dict(data="data/longbench_20k.jsonl", max_prompt=20480, max_new=32, warmup=1, iterations=2),
    "10k": dict(data="data/longbench_20k.jsonl", max_prompt=10240, max_new=32, warmup=1, iterations=2),
    "longbench": dict(data="data/longbench_mfqa.jsonl", max_prompt=8192, max_new=256, warmup=1, iterations=3),
    "retool": dict(data="data/retool_prompts.jsonl", max_prompt=2048, max_new=512, warmup=1, iterations=3),
}

MODELS = {
    "qwen3-4b": "/dev/shm/hf_cache/models/Qwen3-4B",
    "qwen3-8b": "/dev/shm/hf_cache/models/Qwen3-8B",
    "qwen3.5-4b": "/dev/shm/hf_cache/models/Qwen3.5-4B",
    "qwen3.5-9b": "/dev/shm/hf_cache/models/Qwen3.5-9B",
}

PEAK_LIMIT_MB = 43000


def build_command(args, model, workload, arm, c, output, warmup, iterations):
    spec = WORKLOADS[workload]
    cmd = [
        sys.executable, "benchmarks/bench_continuous_graph.py",
        "--model", MODELS[model],
        "--data", spec["data"],
        "--batch-size", "16",
        "--continuous-batch-size", str(c),
        "--max-prompt-length", str(spec["max_prompt"]),
        "--max-new-tokens", str(spec["max_new"]),
        "--warmup", str(warmup),
        "--iterations", str(iterations),
        "--save-response-ids",
        "--output", str(output),
    ]
    if arm == "baseline":
        cmd.append("--no-graph-capture")
    else:
        cmd.extend(["--graph-capture", "--graph-capture-batch-sizes",
                    ",".join(str(w) for w in range(1, c + 1)),
                    "--enable-chunked-prefill", "--prefill-chunk-size", "512"])
    return cmd


def load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def run_one(args, model, workload, arm, c, out_dir, warmup, iterations, attempt):
    stem = f"{model}__{workload}__{arm}"
    raw = out_dir / "raw" / f"{stem}.json"
    log = out_dir / "logs" / f"{stem}.log"
    if args.resume and raw.exists():
        data = load_json(raw)
        if data is not None and data.get("iterations"):
            print(f"[skip] {stem} (existing)", flush=True)
            return data, raw, log
    cmd = build_command(args, model, workload, arm, c, raw, warmup, iterations)
    env = dict(os.environ)
    env.setdefault("HF_HOME", "/dev/shm/hf_cache")
    env["PYTHONUNBUFFERED"] = "1"
    print(f"[run ] {stem} C={c} warmup={warmup} iter={iterations} attempt={attempt}", flush=True)
    start = time.time()
    with open(log, "w") as fh:
        proc = subprocess.run(cmd, cwd=args.opsd_dir, env=env,
                              stdout=fh, stderr=subprocess.STDOUT)
    elapsed = time.time() - start
    data = load_json(raw)
    print(f"[done] {stem} exit={proc.returncode} {elapsed:.1f}s", flush=True)
    return data, raw, log


def validate(data, arm):
    if data is None:
        return "missing"
    if not data.get("iterations"):
        return "no_iterations"
    peak = max(data.get("peak_memory_mb", 0.0), data.get("warmup_peak_memory_mb", 0.0))
    if peak > PEAK_LIMIT_MB:
        return f"peak={peak:.0f}MB"
    if arm == "optimized" and not (data.get("warmup_stats") or {}).get("graph_captures", 0):
        return "no_graph_captures"
    return "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opsd-dir", default="/workspace/DeepSpeedExamples/training/opsd")
    parser.add_argument("--out-dir", default="/workspace/DeepSpeedExamples/training/opsd/runs/sweep_2026-11-10")
    parser.add_argument("--continuous-batch-size", type=int, default=3)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--workloads", default="20k,10k,longbench,retool")
    parser.add_argument("--arms", default="baseline,optimized")
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-retries", type=int, default=1)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "sweep_manifest.json"
    manifest = load_json(manifest_path) or {"cells": {}, "config": vars(args)}
    manifest["config"] = vars(args)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    for model in models:
        for workload in workloads:
            c = args.continuous_batch_size
            spec = WORKLOADS[workload]
            warmup = spec["warmup"] if args.warmup is None else args.warmup
            iterations = spec["iterations"] if args.iterations is None else args.iterations
            results = {}
            logs = {}
            for arm in arms:
                attempts = 0
                while True:
                    attempts += 1
                    data, raw, log = run_one(args, model, workload, arm, c, out_dir,
                                             warmup, iterations, attempts)
                    status = validate(data, arm)
                    if status == "ok":
                        results[arm] = data
                        logs[arm] = log
                        break
                    print(f"[warn] {model}/{workload}/{arm}: {status}", flush=True)
                    if attempts > args.max_retries:
                        results[arm] = data
                        logs[arm] = log
                        break

            cell_key = f"{model}__{workload}"
            cell = {
                "model": model,
                "workload": workload,
                "continuous_batch_size": c,
                "status": "ok",
                "baseline_sha": None,
                "optimized_sha": None,
                "token_match": None,
                "peak_mb": {},
            }
            for arm in arms:
                data = results.get(arm)
                cell["peak_mb"][arm] = None if data is None else max(
                    data.get("peak_memory_mb", 0.0), data.get("warmup_peak_memory_mb", 0.0))
                cell[f"{arm}_sha"] = None if data is None else data.get("response_sha256")
            if "baseline" in results and "optimized" in results:
                b, o = results["baseline"], results["optimized"]
                if b is not None and o is not None:
                    cell["token_match"] = b.get("response_sha256") == o.get("response_sha256")
                    if not cell["token_match"]:
                        cell["status"] = "token_mismatch"
            for arm in arms:
                data = results.get(arm)
                if data is None:
                    cell["status"] = "missing"
                elif arm == "optimized" and not (data.get("warmup_stats") or {}).get("graph_captures", 0):
                    cell["status"] = "no_graph_captures"
            manifest["cells"][cell_key] = cell
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"[cell] {cell_key} status={cell['status']} match={cell['token_match']} "
                  f"peak={cell['peak_mb']}", flush=True)

    print("sweep complete", flush=True)


if __name__ == "__main__":
    main()
