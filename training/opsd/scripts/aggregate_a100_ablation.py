# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Aggregate the A100 re-run of the clean rollout optimization ablation.

Reads runs/a100_2026-11-11/*.json (A100) and runs/*.json (A40 reference) and
writes runs/a100_2026-11-11/optimization_ablation_a100.md, mirroring the
tables of runs/optimization_ablation.md with A40 reference columns.
"""

import json
from pathlib import Path

OPSD = Path(__file__).resolve().parent.parent
A40 = OPSD / "runs"
A100 = OPSD / "runs" / "a100_2026-11-11"
OUT = A100 / "optimization_ablation_a100.md"


def cell(directory, name):
    path = directory / f"{name}.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    it = d["iterations"][0]
    prof = it.get("profile", {})
    cs = it.get("continuous_stats", {})
    peak = max(d.get("peak_memory_mb") or 0.0, d.get("warmup_peak_memory_mb") or 0.0)
    sha = it.get("response_sha256") or d.get("response_sha256") or ""
    return {
        "wall_ms": it.get("wall_ms"),
        "prefill_ms": prof.get("prefill_forward_ms"),
        "decode_ms": prof.get("decode_forward_ms"),
        "tok_s": prof.get("tokens_per_second"),
        "peak_mb": peak if peak > 0 else None,
        "sha": sha[:8],
        "prefix_blocks": cs.get("prefix_reuse_blocks"),
        "prefix_tokens": cs.get("prefix_reuse_tokens"),
    }


def f(value, digits=1, na="-"):
    return na if value is None else f"{value:.{digits}f}"


def ratio(a, b, digits=2):
    if a in (None, 0) or b in (None, 0):
        return None
    return a / b


def pct(a, b):
    """Relative change of b vs a (+ means b is larger/slower/higher)."""
    if a in (None, 0) or b is None:
        return None
    return (b - a) / a * 100.0


def safe(c):
    if c:
        return c
    return {"wall_ms": None, "prefill_ms": None, "decode_ms": None,
            "tok_s": None, "peak_mb": None, "sha": None,
            "prefix_blocks": None, "prefix_tokens": None}


def speedup_row(label, a40_pair, a100_pair, na="—"):
    a40 = ratio(a40_pair[0], a40_pair[1])
    a100 = ratio(a100_pair[0], a100_pair[1])
    return f"| {label} | {f(ratio(a40_pair[1], a40_pair[0])) if a40 else na} | {f(a100 if a100 else (None and na)) if a100 else na} |"


def table(cells, order, header_cols=("Prefill ms", "Decode ms", "Total ms", "Tok/s", "Peak MiB"),
          show_sha=True):
    """One table; cells: list of (label, a40_cell, a100_cell)."""
    cols = ["Step"] + [f"A40 {c}" for c in header_cols] + [f"A100 {c}" for c in header_cols]
    if show_sha:
        cols += ["SHA A40", "SHA A100"]
    lines = ["| " + " | ".join(cols) + " |",
             "|---|" + "---:|" * (len(cols) - 1)]
    for label, a40, a100 in cells:
        row = [label]
        for c in (a40, a100):
            if c is None:
                row += ["—"] * len(header_cols)
            else:
                row += [f(c["prefill_ms"]), f(c["decode_ms"]), f(c["wall_ms"]),
                        f(c["tok_s"]), f(c["peak_mb"], 0)]
        if show_sha:
            row += [f"`{(a40 or {}).get('sha') or '…'}`", f"`{(a100 or {}).get('sha') or '…'}`"]
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def incr_table(rows):
    """rows: (label, a40 (num,den), a100 (num,den)) speedup pairs."""
    lines = ["| Increment | Total speedup A40 | Total speedup A100 | Decode speedup A40 | Decode speedup A100 |",
             "|---|---:|---:|---:|---:|"]
    for label, (n40, d40), (n100, d100) in rows:
        lines.append(f"| {label} | {f(ratio(d40, n40))} | {f(ratio(d100, n100))} | "
                     f"{f(ratio(d40[2] if isinstance(d40, tuple) else d40, n40[2] if isinstance(n40, tuple) else n40))} | "
                     f"{f(ratio(d100, n100))} |")
    return "\n".join(lines)


def decode_ratio(c_from, c_to):
    return (c_from["decode_ms"], c_to["decode_ms"]) if c_from and c_to else (None, None)


def main():
    # A40 reference: the official figures from runs/optimization_ablation.md.
    # (Three 6k dense cells on disk were re-run later with a ~355 ms prefill
    # instead of the md's ~716 ms; the md values are used as the A40 column.)
    A40_MD = {
        "ablation_6k_dense_eager_nochunk": dict(prefill=716.6, decode=775.0, wall=1515.6, tok=10.56, peak=9931.1),
        "ablation_6k_dense_graph_nochunk": dict(prefill=715.6, decode=590.1, wall=1325.8, tok=12.07, peak=9947.6),
        "ablation_6k_paged_decode_graph_nochunk": dict(prefill=718.5, decode=389.8, wall=1147.8, tok=13.94, peak=9955.6),
        "ablation_6k_paged_prefill_graph_nochunk": dict(prefill=719.4, decode=389.9, wall=1128.6, tok=14.18, peak=9091.5),
        "compile_6k_graph": dict(prefill=649.9, decode=348.1, wall=1017.8, tok=15.72, peak=10436.1),
        "compiled_sampler_6k": dict(prefill=719.4, decode=389.9, wall=1126.0, tok=None, peak=None),
        "flashinfer_6k_baseline": dict(prefill=None, decode=None, wall=1129.9, tok=None, peak=None),
        "flashinfer_6k": dict(prefill=None, decode=None, wall=1134.7, tok=None, peak=9202.0),
        "ablation_20k_dense_eager_nochunk": dict(prefill=3058.6, decode=767.6, wall=3852.9, tok=2.08, peak=14697.0),
        "ablation_20k_dense_graph_nochunk": dict(prefill=3067.5, decode=688.4, wall=3777.5, tok=2.12, peak=14713.6),
        "ablation_20k_paged_decode_graph_nochunk": dict(prefill=3071.1, decode=271.5, wall=3397.9, tok=2.35, peak=14717.3),
        "ablation_20k_paged_prefill_graph_nochunk": dict(prefill=3129.2, decode=273.9, wall=3421.5, tok=2.34, peak=11975.2),
        "compile_20k_graph": dict(prefill=2933.0, decode=254.9, wall=3208.0, tok=2.49, peak=16244.2),
        "flashinfer_20k": dict(prefill=None, decode=None, wall=3217.9, tok=None, peak=12086.0),
        "chunk_paged_prefill_6k": dict(prefill=None, decode=None, wall=1233.8, tok=None, peak=None),
        "chunk_dense_staging_6k": dict(prefill=None, decode=None, wall=1809.2, tok=None, peak=None),
        "chunk_paged_prefill_20k_greedy": dict(prefill=None, decode=None, wall=3871.2, tok=None, peak=None),
        "prefix_no_reuse": dict(prefill=2081.3, decode=None, wall=2704.6, tok=5.92, peak=9521.3),
        "prefix_reuse": dict(prefill=1269.7, decode=None, wall=1875.8, tok=8.15, peak=9521.3),
        "vllm_6k_nochunk": dict(prefill=None, decode=None, wall=823.6, tok=19.43, peak=None),
        "vllm_20k_nochunk_greedy": dict(prefill=None, decode=None, wall=2794.4, tok=2.86, peak=None),
    }
    A40_SHAS = {
        "d4a324e9": ["ablation_6k_dense_eager_nochunk", "ablation_6k_dense_graph_nochunk",
                     "ablation_6k_paged_decode_graph_nochunk", "ablation_6k_paged_prefill_graph_nochunk",
                     "compile_6k_graph", "compiled_sampler_6k", "flashinfer_6k_baseline", "flashinfer_6k",
                     "chunk_paged_prefill_6k", "chunk_dense_staging_6k", "vllm_6k_nochunk"],
        "c4291b80": ["ablation_20k_dense_eager_nochunk", "ablation_20k_dense_graph_nochunk",
                     "ablation_20k_paged_decode_eager_nochunk", "ablation_20k_paged_decode_graph_nochunk",
                     "ablation_20k_paged_prefill_graph_nochunk", "compile_20k_graph", "flashinfer_20k",
                     "chunk_paged_prefill_20k_greedy", "vllm_20k_nochunk_greedy"],
        "5530534d": ["prefix_no_reuse", "prefix_reuse"],
    }

    def md_cell(name):
        if name not in A40_MD:
            return None
        v = A40_MD[name]
        sha = next((s for s, names in A40_SHAS.items() if name in names), None)
        return {"wall_ms": v["wall"], "prefill_ms": v["prefill"], "decode_ms": v["decode"],
                "tok_s": v["tok"], "peak_mb": v["peak"], "sha": sha or "",
                "prefix_blocks": 95 if name == "prefix_reuse" else None,
                "prefix_tokens": 6080 if name == "prefix_reuse" else None}

    C40 = {n: md_cell(n) or cell(A40, n) for n in [
        "ablation_6k_dense_eager_nochunk", "ablation_6k_dense_graph_nochunk",
        "ablation_6k_paged_decode_graph_nochunk", "ablation_6k_paged_prefill_graph_nochunk",
        "compile_6k_graph", "compiled_sampler_6k", "flashinfer_6k_baseline", "flashinfer_6k",
        "chunk_paged_prefill_6k", "chunk_dense_staging_6k", "vllm_6k_nochunk",
        "ablation_20k_dense_eager_nochunk", "ablation_20k_dense_graph_nochunk",
        "ablation_20k_paged_decode_eager_nochunk", "ablation_20k_paged_decode_graph_nochunk",
        "ablation_20k_paged_prefill_graph_nochunk", "compile_20k_graph", "flashinfer_20k",
        "chunk_paged_prefill_20k_greedy", "vllm_20k_nochunk_greedy",
        "prefix_no_reuse", "prefix_reuse"]}
    # A40 chunk cells lived under different names; fall back.
    if C40["chunk_paged_prefill_6k"] is None:
        C40["chunk_paged_prefill_6k"] = cell(A40, "fourway_paged_prefill_6k")
    if C40["chunk_dense_staging_6k"] is None:
        C40["chunk_dense_staging_6k"] = cell(A40, "fourway_current_best_6k")
    if C40["chunk_paged_prefill_20k_greedy"] is None:
        C40["chunk_paged_prefill_20k_greedy"] = cell(A40, "fourway_paged_prefill_20k_greedy")

    C100 = {n: cell(A100, n) for n in C40}
    have_vllm_100 = C100.get("vllm_6k_nochunk") is not None or C100.get("vllm_20k_nochunk_greedy") is not None

    md = ["# Rollout optimization ablation — A100 re-run", ""]
    md.append(f"Qwen3-4B, A100-SXM4-40GB, bf16. Same data, batch 1, warmup 1, one measured "
              f"iteration, seed 0 as the A40 run (`runs/optimization_ablation.md`). "
              f"6k = LongBench truncated to 6144, 16 new tokens, Gumbel T=0.7/top_p=0.9; "
              f"20k = LongBench 20480, 8 new tokens, greedy. A40 column: official md values. "
              f"JSON outputs: `runs/a100_2026-11-11/`.")
    md.append("")
    md.append("Note: three on-disk A40 JSONs (6k dense-eager, dense-graph, paged-prefill) were "
              "re-run on 2026-11-11 and show a ~355 ms prefill versus the md's ~716 ms; cause not "
              "established (md values remain the official A40 reference). All A100 6k prefills "
              "measured ~355 ms on every path, so prefill-vs-A40 ratios in the tables use the "
              "md's 716 ms basis and should be read as upper bounds on the prefill speedup.")
    md.append("")

    # ---- 6k ----
    md.append("## Clean 4B 6k ablation (no chunked prefill)")
    md.append("")
    rows6 = [
        ("Dense eager (no graph)", C40["ablation_6k_dense_eager_nochunk"], C100["ablation_6k_dense_eager_nochunk"]),
        ("+ CUDA graph capture", C40["ablation_6k_dense_graph_nochunk"], C100["ablation_6k_dense_graph_nochunk"]),
        ("+ paged decode", C40["ablation_6k_paged_decode_graph_nochunk"], C100["ablation_6k_paged_decode_graph_nochunk"]),
        ("+ paged prefill", C40["ablation_6k_paged_prefill_graph_nochunk"], C100["ablation_6k_paged_prefill_graph_nochunk"]),
        ("+ torch.compile", C40["compile_6k_graph"], C100["compile_6k_graph"]),
    ]
    vllm6_100 = C100.get("vllm_6k_nochunk")
    rows6.append(("vLLM 0.33 no-chunk reference", C40.get("vllm_6k_nochunk"), vllm6_100))
    md.append(table(rows6, None))
    md.append("")

    e40, g40, pd40, pp40, tc40 = (C40[k] for k in [
        "ablation_6k_dense_eager_nochunk", "ablation_6k_dense_graph_nochunk",
        "ablation_6k_paged_decode_graph_nochunk", "ablation_6k_paged_prefill_graph_nochunk",
        "compile_6k_graph"])
    e100, g100, pd100, pp100, tc100 = (C100[k] for k in [
        "ablation_6k_dense_eager_nochunk", "ablation_6k_dense_graph_nochunk",
        "ablation_6k_paged_decode_graph_nochunk", "ablation_6k_paged_prefill_graph_nochunk",
        "compile_6k_graph"])

    def inc(label, a40_from, a100_from, a40_to, a100_to):
        return (label,
                (a40_to["wall_ms"] if a40_to else None, a40_from["wall_ms"] if a40_from else None,
                 a40_to["decode_ms"] if a40_to else None, a40_from["decode_ms"] if a40_from else None),
                (a100_to["wall_ms"] if a100_to else None, a100_from["wall_ms"] if a100_from else None,
                 a100_to["decode_ms"] if a100_to else None, a100_from["decode_ms"] if a100_from else None))

    md.append("| Increment | Total A40 | Total A100 | Decode A40 | Decode A100 |")
    md.append("|---|---:|---:|---:|---:|")
    for label, t40, t100 in [
            inc("CUDA graph capture", e40, e100, g40, g100),
            inc("Paged decode (on graph)", g40, g100, pd40, pd100),
            inc("Paged prefill (on paged decode)", pd40, pd100, pp40, pp100),
            inc("torch.compile on top", pp40, pp100, tc40, tc100)]:
        md.append(f"| {label} | {f(ratio(t40[1], t40[0]))} | {f(ratio(t100[1], t100[0]))} | "
                  f"{f(ratio(t40[3], t40[2]))} | {f(ratio(t100[3], t100[2]))} |")
    vllm40_6k, vllm100_6k = C40.get("vllm_6k_nochunk"), C100.get("vllm_6k_nochunk")
    md.append(f"| Best vs vLLM (no chunk) | "
              f"{f(ratio(tc40['wall_ms'], vllm40_6k['wall_ms'])) if tc40 and vllm40_6k else '—'} | "
              f"{f(ratio(tc100['wall_ms'], vllm100_6k['wall_ms'])) if tc100 and vllm100_6k else '—'} | — | — |")
    md.append("")

    # ---- 20k ----
    md.append("## Clean 4B 20k ablation (no chunked prefill)")
    md.append("")
    rows20 = [
        ("Dense eager (no graph)", C40["ablation_20k_dense_eager_nochunk"], C100["ablation_20k_dense_eager_nochunk"]),
        ("+ CUDA graph capture", C40["ablation_20k_dense_graph_nochunk"], C100["ablation_20k_dense_graph_nochunk"]),
        ("+ paged decode", C40["ablation_20k_paged_decode_graph_nochunk"], C100["ablation_20k_paged_decode_graph_nochunk"]),
        ("+ paged prefill", C40["ablation_20k_paged_prefill_graph_nochunk"], C100["ablation_20k_paged_prefill_graph_nochunk"]),
        ("+ torch.compile", C40["compile_20k_graph"], C100["compile_20k_graph"]),
    ]
    vllm20_100 = C100.get("vllm_20k_nochunk_greedy")
    rows20.append(("vLLM 0.33 no-chunk reference", C40.get("vllm_20k_nochunk_greedy"), vllm20_100))
    md.append(table(rows20, None))
    md.append("")

    e40b, g40b, pd40b, pp40b, tc40b = (C40[k] for k in [
        "ablation_20k_dense_eager_nochunk", "ablation_20k_dense_graph_nochunk",
        "ablation_20k_paged_decode_graph_nochunk", "ablation_20k_paged_prefill_graph_nochunk",
        "compile_20k_graph"])
    e100b, g100b, pd100b, pp100b, tc100b = (C100[k] for k in [
        "ablation_20k_dense_eager_nochunk", "ablation_20k_dense_graph_nochunk",
        "ablation_20k_paged_decode_graph_nochunk", "ablation_20k_paged_prefill_graph_nochunk",
        "compile_20k_graph"])
    md.append("| Increment | Total A40 | Total A100 | Decode A40 | Decode A100 |")
    md.append("|---|---:|---:|---:|---:|")
    for label, t40, t100 in [
            inc("CUDA graph capture", e40b, e100b, g40b, g100b),
            inc("Paged decode (on graph)", g40b, g100b, pd40b, pd100b),
            inc("Paged prefill (on paged decode)", pd40b, pd100b, pp40b, pp100b),
            inc("torch.compile on top", pp40b, pp100b, tc40b, tc100b)]:
        md.append(f"| {label} | {f(ratio(t40[1], t40[0]))} | {f(ratio(t100[1], t100[0]))} | "
                  f"{f(ratio(t40[3], t40[2]))} | {f(ratio(t100[3], t100[2]))} |")
    vllm40_20k, vllm100_20k = C40.get("vllm_20k_nochunk_greedy"), C100.get("vllm_20k_nochunk_greedy")
    md.append(f"| Best vs vLLM (no chunk) | "
              f"{f(ratio(tc40b['wall_ms'], vllm40_20k['wall_ms'])) if tc40b and vllm40_20k else '—'} | "
              f"{f(ratio(tc100b['wall_ms'], vllm100_20k['wall_ms'])) if tc100b and vllm100_20k else '—'} | — | — |")
    md.append("")

    # ---- vLLM distance ----
    md.append("## vLLM reference and distance")
    md.append("")
    vllm100 = "vLLM cells re-run on A100 2026-11-12" if have_vllm_100 else "A40 reference only (vLLM re-run pending/skipped)"
    md.append(f"vLLM 0.33.0, same prompt/sampler/counts; KV capped to 1.1 GiB (6k) / 3.3 GiB (20k). {vllm100}.")
    md.append("")
    md.append("| Workload | Before clean (dense eager) | Best DeepSpeed A100 | vLLM | Best vs vLLM (A100) |")
    md.append("|---|---:|---:|---:|---:|")
    v6 = C100.get("vllm_6k_nochunk") or C40.get("vllm_6k_nochunk")
    v20 = C100.get("vllm_20k_nochunk_greedy") or C40.get("vllm_20k_nochunk_greedy")
    if e100 and tc100 and v6:
        r6 = ratio(tc100["wall_ms"], v6["wall_ms"]) if C100.get("vllm_6k_nochunk") else None
        md.append(f"| 4B 6k, 16 new, no chunk | {f(e100['wall_ms'])} ms | {f(tc100['wall_ms'])} ms (compiled) | "
                  f"{f(v6['wall_ms'])} ms | {f(r6) if r6 else '—'}x slower |")
    if e100b and tc100b and v20:
        r20 = ratio(tc100b["wall_ms"], v20["wall_ms"]) if C100.get("vllm_20k_nochunk_greedy") else None
        md.append(f"| 4B 20k, 8 new greedy | {f(e100b['wall_ms'])} ms | {f(tc100b['wall_ms'])} ms (compiled) | "
                  f"{f(v20['wall_ms'])} ms | {f(r20) if r20 else '—'}x slower |")
    md.append("")

    # ---- chunk penalty ----
    md.append("## Chunk penalty: DeepSpeed (A100 re-run vs A40)")
    md.append("")
    md.append("| Engine / path | 6k no-chunk A100 | 6k chunk 512 A100 | Penalty A100 | 20k no-chunk A100 | 20k chunk 512 A100 | Penalty A100 |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    c6p, c6d, c20p = (safe(C100.get("chunk_paged_prefill_6k")),
                      safe(C100.get("chunk_dense_staging_6k")),
                      safe(C100.get("chunk_paged_prefill_20k_greedy")))
    pp100s, pp100bs, g100s = safe(pp100), safe(pp100b), safe(g100)
    md.append(f"| DeepSpeed paged prefill/decode | {f(pp100s['wall_ms'])} | {f(c6p['wall_ms'])} | "
              f"{f(pct(pp100s['wall_ms'], c6p['wall_ms']))}% | {f(pp100bs['wall_ms'])} | {f(c20p['wall_ms'])} | "
              f"{f(pct(pp100bs['wall_ms'], c20p['wall_ms']))}% |")
    if c6d["wall_ms"] is not None:
        md.append(f"| DeepSpeed dense staging + paged decode | {f(g100s['wall_ms'])} | {f(c6d['wall_ms'])} | "
                  f"{f(pct(g100s['wall_ms'], c6d['wall_ms']))}% | — | — | — |")
    md.append("A40 reference: vLLM +15.9%/+20.4%, DeepSpeed paged +9.3%/+13.1%, dense staging +36.5% (see `runs/optimization_ablation.md`).")
    md.append("")

    # ---- prefix reuse ----
    md.append("## Prefix reuse (6k, batch 2, identical prompts, chunk 512, no graph)")
    md.append("")
    pr40, ppr40 = C40["prefix_no_reuse"], C40["prefix_reuse"]
    pr100, ppr100 = C100["prefix_no_reuse"], C100["prefix_reuse"]
    md.append("| Path | Prefill ms A40 | Prefill ms A100 | Total ms A100 | Tok/s A100 | Peak MiB A100 | Reuse (A100) |")
    md.append("|---|---:|---:|---:|---:|---:|---|")
    for label, a40, a100 in [("Paged, no reuse", pr40, pr100), ("Paged, prefix reuse", ppr40, ppr100)]:
        a40, a100 = safe(a40), safe(a100)
        reuse = "—"
        if a100["prefix_blocks"]:
            reuse = f"{a100['prefix_blocks']} blocks / {a100['prefix_tokens']} tokens"
        md.append(f"| {label} | {f(a40['prefill_ms'])} | {f(a100['prefill_ms'])} | {f(a100['wall_ms'])} | "
                  f"{f(a100['tok_s'])} | {f(a100['peak_mb'], 0)} | {reuse} |")
    if pr100 and ppr100:
        md.append(f"\n- Prefill speedup on A100: **{f(ratio(pr100['prefill_ms'], ppr100['prefill_ms']))}x**; "
                  f"total: **{f(ratio(pr100['wall_ms'], ppr100['wall_ms']))}x**.")
    md.append("- A100 prefix cells re-run 2026-11-12 with `--enable-paged-prefill`: the original cells ran on "
              "code where paged prefill was still implicit under `--use-paged-cache` (the A40 effective config), "
              "so they took the dense-staging chunk path instead (0 blocks reused, 24 prefill forwards) — the "
              "\"prefix anomaly\" in the profile notes, a flag/code drift, not hardware.")
    md.append("")

    # ---- flashinfer ----
    md.append("## FlashInfer prefill backend")
    md.append("")
    md.append("| Workload | Blocked prefill A100 | FlashInfer prefill A100 | Total A100 | Peak MiB A100 |")
    md.append("|---|---:|---:|---:|---:|")
    fb40, fi40 = C100.get("flashinfer_6k_baseline") or C100.get("ablation_6k_paged_prefill_graph_nochunk"), C100.get("flashinfer_6k")
    fb20, fi20 = C100.get("ablation_20k_paged_prefill_graph_nochunk"), C100.get("flashinfer_20k")
    if fb40:
        md.append(f"| 4B 6k | {f(fb40['wall_ms'])} ms | {f(fi40['wall_ms']) if fi40 else '—'} ms | "
                  f"{f(ratio(fb40['wall_ms'], fi40['wall_ms']) if fi40 else None)} | "
                  f"{f(fi40['peak_mb'], 0) if fi40 else '—'} vs {f(fb40['peak_mb'], 0)} |")
    if fb20:
        md.append(f"| 4B 20k | {f(fb20['wall_ms'])} ms | {f(fi20['wall_ms']) if fi20 else '—'} ms | "
                  f"{f(ratio(fb20['wall_ms'], fi20['wall_ms']) if fi20 else None)} | "
                  f"{f(fi20['peak_mb'], 0) if fi20 else '—'} vs {f(fb20['peak_mb'], 0)} |")
    md.append("")

    # ---- token check ----
    md.append("## Response-token check")
    md.append("")
    md.append("| Cell | A40 SHA | A100 SHA | Match |")
    md.append("|---|---|---|---|")
    for name in list(C40):
        a40, a100 = C40[name], C100[name]
        if not a40 or not a100:
            continue
        match = "yes" if a40["sha"] == a100["sha"] else f"no ({a40['sha']} vs {a100['sha']})"
        md.append(f"| {name} | `{a40['sha']}` | `{a100['sha']}` | {match} |")
    md.append("")

    OUT.write_text("\n".join(md) + "\n")
    n_done = sum(1 for c in C100.values() if c)
    print(f"wrote {OUT} ({n_done}/{len(C100)} A100 cells present)")
    missing = [n for n, c in C100.items() if c is None]
    if missing:
        print("missing:", ", ".join(missing))


if __name__ == "__main__":
    main()
