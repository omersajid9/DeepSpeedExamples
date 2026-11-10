# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Focused chunked-prefill sweep on Qwen3-4B at 5k and 10k prompt lengths.

Arms per prompt length:
  baseline   continuous eager, no graph, no chunked prefill
  graph_only continuous batching + decode graphs, no chunked prefill
  chunk_X    continuous batching + decode graphs + chunked prefill at X tokens

The graph_only arm separates the graph effect from the chunking effect.
"""

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

PROMPT_LENGTHS = (5120, 10240)
CHUNK_SIZES = (256, 512, 1024, 2048, 4096, 8192)
MODEL = "/dev/shm/hf_cache/models/Qwen3-4B"
DATA = "data/longbench_20k.jsonl"
PEAK_LIMIT_MB = 43000


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def fmt(value, digits=2):
    return "" if value is None else f"{value:.{digits}f}"


def build(opsd_dir, prompt_len, arm, chunk_size, output):
    cmd = [
        sys.executable, "benchmarks/bench_continuous_graph.py",
        "--model", MODEL,
        "--data", DATA,
        "--batch-size", "16",
        "--continuous-batch-size", "3",
        "--max-prompt-length", str(prompt_len),
        "--max-new-tokens", "32",
        "--warmup", "1",
        "--iterations", "3",
        "--save-response-ids",
        "--output", output,
    ]
    if arm == "baseline":
        cmd.append("--no-graph-capture")
    if arm in ("graph_only", "chunk"):
        cmd.extend(["--graph-capture", "--graph-capture-batch-sizes", "1,2,3"])
    if arm == "chunk":
        cmd.extend(["--enable-chunked-prefill", "--prefill-chunk-size", str(chunk_size)])
    return cmd


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def run_cell(opsd_dir, out_dir, prompt_len, arm, chunk_size):
    if arm == "baseline":
        label = f"qwen3-4b_p{prompt_len}_baseline"
    elif arm == "graph_only":
        label = f"qwen3-4b_p{prompt_len}_graph_only"
    else:
        label = f"qwen3-4b_p{prompt_len}_chunk{chunk_size}"
    raw = out_dir / "raw" / f"{label}.json"
    log = out_dir / "logs" / f"{label}.log"
    cmd = build(opsd_dir, prompt_len, arm, chunk_size, str(raw))
    env = dict(os.environ)
    env.setdefault("HF_HOME", "/dev/shm/hf_cache")
    env["PYTHONUNBUFFERED"] = "1"
    print(f"[run] {label}", flush=True)
    with open(log, "w") as fh:
        proc = subprocess.run(cmd, cwd=opsd_dir, env=env, stdout=fh, stderr=subprocess.STDOUT)
    data = load(raw)
    if proc.returncode != 0 or data is None or not data.get("iterations"):
        print(f"[FAIL] {label} exit={proc.returncode}", flush=True)
        return None
    peak = max(data.get("peak_memory_mb", 0.0), data.get("warmup_peak_memory_mb", 0.0))
    if peak > PEAK_LIMIT_MB:
        print(f"[FAIL] {label} peak={peak:.0f} MiB", flush=True)
        return None
    if arm in ("graph_only", "chunk"):
        caps = (data.get("warmup_stats") or {}).get("graph_captures", 0)
        if not caps:
            print(f"[FAIL] {label} no graph captures", flush=True)
            return None
    print(f"[done] {label}", flush=True)
    return data


def summarize(rows, out_dir):
    md = ["# Chunked-prefill sweep — Qwen3-4B, 5k/10k prompts", ""]
    md.append("Model: Qwen3-4B. batch 16, continuous 3, max_new_tokens 32. "
              "Baseline = eager, no graph, no chunk. graph_only = decode graphs, no chunk. "
              "chunk_X = decode graphs + chunked prefill at X tokens.")
    md.append("")
    md.append("| Prompt | Arm | Chunk | Wall (s) | Prefill (s) | Decode (s) | Chunks | Staging (GB) | Peak (MiB) | Wall Δ vs baseline | Prefill Δ vs graph_only |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for prompt_len in PROMPT_LENGTHS:
        group = [r for r in rows if r["prompt_len"] == prompt_len]
        base = next((r for r in group if r["arm"] == "baseline"), None)
        graph = next((r for r in group if r["arm"] == "graph_only"), None)
        for r in group:
            wall_delta = None
            if base and base["wall_s"]:
                wall_delta = (r["wall_s"] - base["wall_s"]) / base["wall_s"] * 100.0
            prefill_delta = None
            if graph and graph["prefill_s"]:
                prefill_delta = (r["prefill_s"] - graph["prefill_s"]) / graph["prefill_s"] * 100.0
            md.append("| {p} | {arm} | {chunk} | {wall} | {prefill} | {decode} | {chunks} | "
                      "{staging} | {peak} | {wd} | {pd} |".format(
                          p=prompt_len, arm=r["arm"], chunk=r["chunk"],
                          wall=fmt(r["wall_s"]), prefill=fmt(r["prefill_s"]),
                          decode=fmt(r["decode_s"]), chunks=r["chunks"],
                          staging=fmt(r["staging_gb"]), peak=fmt(r["peak_mb"], 0),
                          wd=fmt(wall_delta, 1), pd=fmt(prefill_delta, 1)))
    md.append("")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")
    csv_path = out_dir / "summary.csv"
    fields = ["prompt_len", "arm", "chunk", "wall_s", "prefill_s", "decode_s", "chunks",
              "staging_gb", "peak_mb"]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fields})
    print(f"wrote {out_dir/'summary.md'} and {csv_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opsd-dir", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent.parent / "runs" / "chunk_sweep_2026-11-10"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    rows = []
    for prompt_len in PROMPT_LENGTHS:
        for arm, chunk in [("baseline", 0), ("graph_only", 0)] + [("chunk", c) for c in CHUNK_SIZES]:
            label = (f"qwen3-4b_p{prompt_len}_baseline" if arm == "baseline" else
                     f"qwen3-4b_p{prompt_len}_graph_only" if arm == "graph_only" else
                     f"qwen3-4b_p{prompt_len}_chunk{chunk}")
            raw = out_dir / "raw" / f"{label}.json"
            data = load(raw) if args.resume and raw.exists() else None
            if data is None or not data.get("iterations"):
                data = run_cell(args.opsd_dir, out_dir, prompt_len, arm, chunk)
            if data is None:
                continue
            wall = median([it["wall_ms"] for it in data["iterations"]]) / 1000.0
            prefill = median([it["profile"]["prefill_forward_ms"] for it in data["iterations"]]) / 1000.0
            decode = median([it["profile"]["decode_forward_ms"] for it in data["iterations"]]) / 1000.0
            chunks = median([it["continuous_stats"].get("chunked_prefill_chunks", 0) for it in data["iterations"]])
            staging = (data["iterations"][0]["continuous_stats"].get("chunked_prefill_staging_bytes", 0)
                       or 0) / 1e9
            peak = max(data.get("peak_memory_mb", 0.0), data.get("warmup_peak_memory_mb", 0.0))
            rows.append({"prompt_len": prompt_len, "arm": arm, "chunk": chunk or 0,
                         "wall_s": wall, "prefill_s": prefill, "decode_s": decode,
                         "chunks": int(chunks or 0), "staging_gb": staging, "peak_mb": peak})
    summarize(rows, out_dir)


if __name__ == "__main__":
    main()
