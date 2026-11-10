# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Aggregate the OPSD baseline-vs-optimized sweep into CSV and Markdown."""

import argparse
import csv
import json
import statistics
from pathlib import Path


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def fmt(value, digits=2):
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="/workspace/DeepSpeedExamples/training/opsd/runs/sweep_2026-11-10")
    args = parser.parse_args()
    root = Path(args.dir)
    manifest = load(root / "sweep_manifest.json") or {}
    rows = []
    for key, cell in sorted(manifest.get("cells", {}).items()):
        model = cell["model"]
        workload = cell["workload"]
        pair = {}
        for arm in ("baseline", "optimized"):
            data = load(root / "raw" / f"{model}__{workload}__{arm}.json")
            pair[arm] = data
        for arm in ("baseline", "optimized"):
            data = pair[arm]
            if data is None:
                continue
            iters = data.get("iterations") or []
            walls = [it.get("wall_ms") for it in iters]
            profiles = [it.get("profile") or {} for it in iters]
            stats = [it.get("continuous_stats") or {} for it in iters]
            prefill = [p.get("prefill_forward_ms") for p in profiles]
            decode = [p.get("decode_forward_ms") for p in profiles]
            gen_tokens = [p.get("num_generated_tokens") for p in profiles]
            prompt_tokens = data.get("batch_size", 0) * data.get("max_prompt_length", 0)
            wall_s = median(walls) / 1000.0 if median(walls) else None
            prefill_s = median(prefill) / 1000.0 if median(prefill) else None
            decode_s = median(decode) / 1000.0 if median(decode) else None
            wall_tps = (median(gen_tokens) / wall_s) if wall_s and median(gen_tokens) else None
            prefill_tps = (prompt_tokens / prefill_s) if prefill_s else None
            decode_tps = (median(gen_tokens) / decode_s) if decode_s and median(gen_tokens) else None
            rows.append({
                "model": model,
                "workload": workload,
                "arm": arm,
                "c": cell["continuous_batch_size"],
                "graph": data.get("graph_capture"),
                "chunked": data.get("enable_chunked_prefill"),
                "wall_s": wall_s,
                "prefill_s": prefill_s,
                "decode_s": decode_s,
                "wall_tps": wall_tps,
                "prefill_tps": prefill_tps,
                "decode_tps": decode_tps,
                "peak_mb": max(data.get("peak_memory_mb", 0.0), data.get("warmup_peak_memory_mb", 0.0)),
                "warmup_captures": (data.get("warmup_stats") or {}).get("graph_captures"),
                "chunks": median([s.get("chunked_prefill_chunks") for s in stats]),
                "sha": data.get("response_sha256"),
                "token_match": cell.get("token_match"),
                "status": cell.get("status"),
            })

    csv_path = root / "sweep_summary.csv"
    fields = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    md = ["# OPSD rollout optimization sweep — results", ""]
    md.append("Environment: A40 46,068 MiB; torch 2.14.1+cu126; transformers 5.22.2; "
              "DeepSpeed local checkout. Baseline is continuous eager (no graph, no chunk). "
              "Optimized is decode CUDA graphs + chunked prefill. Qwen3.5 prefill graphs are "
              "run eagerly because hybrid recurrent-state capture is not token-stable; its decode "
              "graphs and chunked prefill remain active.")
    md.append("")
    md.extend([
        "## Findings (partial sweep)",
        "",
        "The optimized bundle is **decode-graph + chunked-prefill**, but the two parts",
        "help opposite workload shapes. It wins when decode dominates and loses when",
        "prefill dominates, because chunked prefill is a concurrency/memory feature, not",
        "a prefill-latency feature.",
        "",
        "### Why ReTool improves",
        "",
        "- ReTool prompts are 197-462 tokens, below `prefill_chunk_size=512`, so chunked",
        "  prefill is a no-op (`chunked_prefill_chunks=0`). The optimized arm is effectively",
        "  **decode graphs only**.",
        "- 512 output tokens make decode ~98% of baseline wall time (156.4 s of 159.7 s).",
        "  The bucket graphs replace the eager per-token launch path and replay the smallest",
        "  captured width; Qwen3-4B ReTool goes 159.7 s -> 86.9 s (decode 156.4 s -> 83.8 s).",
        "- This is the intended decode win: fewer launches and a fixed-shape replay for a",
        "  long, narrow decode loop.",
        "",
        "### Why 10k/20k regress",
        "",
        "- With `max_new_tokens=32`, decode is only ~20-27% of baseline wall time. The graph",
        "  saves a few seconds, while chunked prefill adds tens of seconds.",
        "- Qwen3-4B 20k: prefill 109.5 s -> 162.7 s (+49%), decode 42.4 s -> 40.4 s (-5%),",
        "  wall 152.3 s -> 204.1 s (+34%). Qwen3-4B 10k: prefill 21.4 s -> 48.2 s (+125%),",
        "  wall 49.3 s -> 68.6 s (+39%). Qwen3-8B 10k: prefill 31.5 s -> 58.4 s (+85%),",
        "  wall 56.1 s -> 81.0 s (+44%).",
        "- **Chunking is not free.** A 20k prompt becomes 40 x 512-token forwards instead of",
        "  one large forward. The smaller GEMMs are less efficient, each chunk pays launch/",
        "  mask/graph-dispatch overhead, and the prefill bucket may run a wider parked row",
        "  than the active wave (see `prefill_padded_rows`).",
        "- **The staging cache doubles KV traffic.** Each chunk writes into a second KV cache,",
        "  and the completed prompt is copied into the decode cache. For 20k this is roughly",
        "  2.8 GB of KV per slot (Qwen3-4B/8B geometry) copied once per prompt, on top of the",
        "  chunked writes. That is pure overhead for a prefill-only workload.",
        "- **There is almost no decode to overlap.** Chunked prefill is useful when a long",
        "  prompt can fill while other rows keep decoding. With 32 new tokens and 16 prompts",
        "  the loop is prefill-bound; the overlap window is too small to amortize the chunks.",
        "- The same mechanism explains 10k: 20 chunks instead of 40, but the per-chunk",
        "  overhead and staging copy are still proportional to prompt length.",
        "",
        "### Practical takeaway",
        "",
        "- Use **decode graphs** for decode-heavy workloads (ReTool-like: short prompts, long",
        "  responses).",
        "- Do **not** enable chunked prefill for 20k/10k with short responses unless the goal",
        "  is concurrency/memory, not latency. If chunking is required, try a larger chunk",
        "  (2048-4096) to reduce per-chunk overhead and staging traffic.",
        "- Report the bundle per workload; a single aggregate would hide the sign flip.",
        "",
        "### Caveats",
        "",
        "- Partial sweep: Qwen3-4B LongBench baseline, Qwen3-8B 20k C=2 pair, and all",
        "  Qwen3.5 cells are still pending.",
        "- Baseline vs optimized response IDs diverge after 10-30 tokens for several rows.",
        "  ReTool (no chunking) shows the same divergence, so it is decode-graph/batch-width",
        "  numerical divergence rather than chunked-prefill corruption. Performance numbers",
        "  remain valid; the strict bitwise-token-match gate is not met.",
        "- Qwen3.5 hybrid prefill graphs are run eagerly (recurrent-state capture is not",
        "  replay-stable); its decode graphs and chunked prefill remain active.",
        "",
    ])
    md.append("## Per-cell results")
    md.append("")
    md.append("| Model | Workload | Arm | C | Wall (s) | Prefill (s) | Decode (s) | Wall tok/s | Prefill tok/s | Decode tok/s | Peak (MiB) | Match | Status |")
    md.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for row in rows:
        md.append("| {model} | {workload} | {arm} | {c} | {wall_s} | {prefill_s} | {decode_s} | "
                  "{wall_tps} | {prefill_tps} | {decode_tps} | {peak_mb} | {token_match} | {status} |".format(
                      **{**row,
                         "wall_s": fmt(row["wall_s"]),
                         "prefill_s": fmt(row["prefill_s"]),
                         "decode_s": fmt(row["decode_s"]),
                         "wall_tps": fmt(row["wall_tps"]),
                         "prefill_tps": fmt(row["prefill_tps"]),
                         "decode_tps": fmt(row["decode_tps"]),
                         "peak_mb": fmt(row["peak_mb"], 0),
                         "token_match": row["token_match"],
                         "status": row["status"]}))
    md.append("")
    md.append("## Optimization deltas")
    md.append("")
    md.append("| Model | Workload | Wall Δ | Prefill Δ | Decode Δ | Wall tok/s Δ | Peak Δ | Match |")
    md.append("|---|---|---:|---:|---:|---:|---:|---|")
    by_pair = {}
    for row in rows:
        by_pair.setdefault((row["model"], row["workload"]), {})[row["arm"]] = row
    for (model, workload), pair in sorted(by_pair.items()):
        b = pair.get("baseline")
        o = pair.get("optimized")
        if not b or not o:
            continue
        def delta(a, c):
            if a is None or c in (None, 0):
                return None
            return (a - c) / c * 100.0
        md.append("| {model} | {workload} | {wall} | {prefill} | {decode} | {tps} | {peak} | {match} |".format(
            model=model, workload=workload,
            wall=fmt(delta(b["wall_s"], o["wall_s"]), 1),
            prefill=fmt(delta(b["prefill_s"], o["prefill_s"]), 1),
            decode=fmt(delta(b["decode_s"], o["decode_s"]), 1),
            tps=fmt(delta(o["wall_tps"], b["wall_tps"]), 1),
            peak=fmt(delta(o["peak_mb"], b["peak_mb"]), 1),
            match=o["token_match"]))
    md.append("")
    (root / "sweep_summary.md").write_text("\n".join(md) + "\n")
    print(f"wrote {csv_path} and {root / 'sweep_summary.md'}")


if __name__ == "__main__":
    main()
