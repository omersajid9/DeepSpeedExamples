# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Meeting figures for the rollout-optimization study (2026-11-12).

Figures (all in runs/meeting_figs_2026-11-12/, PDF + PNG):
  CB ablation (A_eager = 1.0, 4 panels per metric):
    cb_levels_vs_eager_latency / _prefill / _decode / _tok_s / _peak
  A100 C=1 ladder, vLLM = 1.0 (total latency + throughput — the only
  metrics the vLLM oracle records; throughput = new tokens / total wall):
    ablation_vs_vllm_{6k,20k}_{total,tok_s}
  A100 C=1 ladder, dense eager = 1.0 (phase/memory metrics — the vLLM
  oracle has no phase-level or memory data):
    ablation_vs_dense_eager_{6k,20k}_{prefill,decode,peak}
  Chunk sweep:
    chunk_sweep_latency_memory — chunk size vs wall latency + peak memory

Requires the lazy-init-fix CB rerun to be complete (asserts all 12 D/E/F raw
JSONs are newer than the pre-fix backup before plotting).
"""

import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OPSD = Path(__file__).resolve().parent.parent
CB = OPSD / "runs" / "cb_ablation_2026-11-12"
CB_RAW = CB / "raw"
CB_BAK = CB / "pre_lazy_init_fix"
A100 = OPSD / "runs" / "a100_2026-11-11"
OUT = OPSD / "runs" / "meeting_figs_2026-11-12"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "axes.edgecolor": "0.35",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
C_TOTAL, C_PREF, C_DEC = "#4e79a7", "#f28e2b", "#59a14f"
C_TOK, C_MEM = "#7b3294", "#a3a3a3"


def cell(directory, name):
    p = directory / f"{name}.json"
    d = json.loads(p.read_text())
    it = d["iterations"][0]
    prof = it.get("profile", {})
    peak = max(d.get("peak_memory_mb") or 0.0, d.get("warmup_peak_memory_mb") or 0.0)
    tok_s = prof.get("tokens_per_second")
    if tok_s is None and it.get("wall_ms") and d.get("max_new_tokens"):
        # vLLM oracle records wall only; throughput := new tokens / total wall
        # (same definition as the bench's tokens_per_second = tokens / total_ms)
        tok_s = d["max_new_tokens"] / (it["wall_ms"] / 1000.0)
    return {
        "wall": it.get("wall_ms"),
        "prefill": prof.get("prefill_forward_ms"),
        "decode": prof.get("decode_forward_ms"),
        "tok_s": tok_s,
        "peak": peak if peak > 0 else None,
    }


def assert_rerun_fresh():
    stale = []
    for m in ("qwen3_4b", "qwen3_8b"):
        for t in ("6k", "20k"):
            for lv in ("D_paged", "E_fused", "F_flashinfer_sampler"):
                new, bak = CB_RAW / f"{m}_{t}_{lv}.json", CB_BAK / f"{m}_{t}_{lv}.json"
                if not (new.exists() and bak.exists() and new.stat().st_mtime > bak.stat().st_mtime):
                    stale.append(f"{m}_{t}_{lv}")
    if stale:
        raise SystemExit(f"CB rerun not complete for: {', '.join(stale)} — refusing to plot stale cells")


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"wrote {OUT}/{name}.pdf/.png")


def annotate_ratios(ax, bars, values, digits=2):
    for r in bars:
        v = r.get_height()
        if v is None:
            continue
        ax.annotate(f"{v:.{digits}f}", (r.get_x() + r.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 2), ha="center", fontsize=7.5)


# ---------------------------------------------------------------- CB vs eager
def cb_figure(metric, fname, ylabel):
    cells = []
    for m, t, label in [("qwen3_4b", "6k", "Qwen3-4B, 6k (C=4)"),
                        ("qwen3_4b", "20k", "Qwen3-4B, 20k (C=2)"),
                        ("qwen3_8b", "6k", "Qwen3-8B, 6k (C=4)"),
                        ("qwen3_8b", "20k", "Qwen3-8B, 20k (C=2)")]:
        row = {}
        for lv in ("A_eager", "B_single_graph", "C_multi_graph", "D_paged", "E_fused", "F_flashinfer_sampler"):
            c = cell(CB_RAW, f"{m}_{t}_{lv}")
            row[lv] = c[metric]
        cells.append((label, row))
    labels = ["A\neager", "B\ngraph", "C\nladder", "D\npaged", "E\nfused", "F\nFI dec"]
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.4))
    for ax, (label, row) in zip(axes, cells):
        base = row["A_eager"]
        vals = [row[lv] / base for lv in row]
        x = range(len(row))
        bars = ax.bar(x, vals, width=0.62, color=[C_TOTAL] * len(vals),
                      edgecolor="0.3", linewidth=0.5, zorder=3)
        ax.axhline(1.0, color="0.3", linestyle="--", linewidth=0.8, zorder=2)
        annotate_ratios(ax, bars, vals)
        ax.set_yscale("log")
        ax.set_ylim(bottom=max(0.10, min(vals) * 0.75))
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_title(label, fontsize=10.5)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(f"Continuous-batching ablation, A100 — ratio to level A (eager + continuous batching)   [{metric}]",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, fname)


# ------------------------------------------------- A100 C=1 ladder vs vLLM
LADDER = [
    ("ablation_6k_dense_eager_nochunk", "Dense\neager"),
    ("ablation_6k_dense_graph_nochunk", "+CUDA\ngraph"),
    ("ablation_6k_paged_decode_graph_nochunk", "+paged\ndecode"),
    ("ablation_6k_paged_prefill_graph_nochunk", "+paged\nprefill"),
    ("compile_6k_graph", "+torch.\ncompile"),
    ("newbest_6k", "new best\n(FI decode,\ncompiled sampler)"),
]
LADDER_20 = [
    ("ablation_20k_dense_eager_nochunk", "Dense\neager"),
    ("ablation_20k_dense_graph_nochunk", "+CUDA\ngraph"),
    ("ablation_20k_paged_decode_graph_nochunk", "+paged\ndecode"),
    ("ablation_20k_paged_prefill_graph_nochunk", "+paged\nprefill"),
    ("compile_20k_graph", "+torch.\ncompile"),
    ("newbest_20k", "new best\n(FI decode,\ncompiled sampler)"),
]


def a100_figure(metric, fname, ylabel, ladder_key):
    ladder = LADDER if ladder_key == 6 else LADDER_20
    vllm_name = "vllm_6k_nochunk" if ladder_key == 6 else "vllm_20k_nochunk_greedy"
    vllm_val = cell(A100, vllm_name)[metric]
    rows = []
    if vllm_val:
        for name, label in ladder:
            rows.append((label, cell(A100, name)[metric] / vllm_val))
        rows.append(("vLLM\n0.33", 1.0))
        ref = "ratio to vLLM 0.33"
    else:
        # vLLM oracle has no phase-level data for this metric:
        # normalize against the dense-eager rung instead
        base = cell(A100, ladder[0][0])[metric]
        for name, label in ladder:
            rows.append((label, cell(A100, name)[metric] / base))
        ref = "ratio to dense eager"
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    x = range(len(rows))
    has_vllm = bool(vllm_val)
    colors = [C_TOTAL if i < len(rows) - (1 if has_vllm else 0) else "#8c8c8c" for i in x]
    bars = ax.bar(x, [v for _, v in rows], width=0.62, color=colors,
                  edgecolor="0.3", linewidth=0.5, zorder=3)
    ax.axhline(1.0, color="0.3", linestyle="--", linewidth=0.8, zorder=2)
    annotate_ratios(ax, bars, [v for _, v in rows])
    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels([l for l, _ in rows], fontsize=8)
    ax.set_title(f"A100 clean ablation, batch 1 / C=1, 4B, {ladder_key}k — {ref}   [{metric}]",
                 fontsize=11)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    save(fig, fname)


# ------------------------------------------------------- chunk sweep
CHUNKS = [128, 256, 512, 1024, 2048, 4096, 8192]


def chunk_figure():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6), sharex=True)
    mem_axes = []
    for ax, (key, title) in zip(axes, [(6, "6k, 16 new, Gumbel"), (20, "20k, 8 new, greedy")]):
        nb = cell(A100, f"newbest_{key}k")
        walls, peaks, cks = [], [], []
        for c in CHUNKS:
            d = cell(A100, f"chunk_sweep_{key}k_c{c}")
            cks.append(c)
            walls.append(d["wall"])
            peaks.append(d["peak"])
        cks.append("no\nchunk")
        walls.append(nb["wall"])
        peaks.append(nb["peak"])
        ax.plot(range(len(cks) - 1), walls[:-1], marker="o", markersize=4.5, linewidth=1.4,
                color=C_TOTAL, label="Wall latency", zorder=3)
        ax.plot(len(cks) - 1, nb["wall"], marker="s", markersize=5, color=C_TOTAL,
                linestyle="", zorder=3)
        ax.annotate(f"no chunk\n{nb['wall']:.0f} ms", (len(cks) - 1, nb["wall"]),
                    textcoords="offset points", xytext=(-34, 10), fontsize=8, color=C_TOTAL)
        ax2 = ax.twinx()
        ax2.plot(range(len(cks) - 1), peaks[:-1], marker="^", markersize=4.5, linewidth=1.4,
                 linestyle=":", color=C_MEM, label="Peak memory", zorder=2)
        ax2.plot(len(cks) - 1, nb["peak"], marker="D", markersize=4.5, color=C_MEM,
                 linestyle="", zorder=2)
        ax2.set_ylabel("Peak memory (MiB)", color=C_MEM)
        ax2.tick_params(axis="y", colors=C_MEM)
        ax2.spines["right"].set_visible(True)
        mem_axes.append(ax2)
        ax.set_xlabel("Prefill chunk size (tokens, log scale)")
        ax.set_title(title, fontsize=10.5)
        ax.set_ylabel("Wall latency (ms)")
        ticks = list(range(len(cks)))
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(c) for c in CHUNKS] + ["no\nchunk"], fontsize=8)
        ax.set_xlim(-0.4, len(cks) - 0.6)
    wall_h, _ = axes[0].get_legend_handles_labels()
    mem_h, _ = mem_axes[0].get_legend_handles_labels()
    fig.legend(wall_h[:1] + mem_h[:1], ["Wall latency", "Peak memory"],
               loc="upper center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Chunked prefill sweep on the new-best config — latency vs memory "
                 "(new-best config, batch 1 / C=1)", fontsize=12, y=1.10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "chunk_sweep_latency_memory")


if __name__ == "__main__":
    assert_rerun_fresh()
    cb_figure("wall", "cb_levels_vs_eager_latency", "Total latency, ratio to A (log scale)")
    cb_figure("prefill", "cb_levels_vs_eager_prefill", "Prefill forward, ratio to A (log scale)")
    cb_figure("decode", "cb_levels_vs_eager_decode", "Decode forward, ratio to A (log scale)")
    # tok/s and peak memory: ratios are fine for tok/s; for memory ratio = mem/ A
    cb_figure("tok_s", "cb_levels_vs_eager_tok_s", "Throughput (tok/s), ratio to A")
    cb_figure("peak", "cb_levels_vs_eager_peak", "Peak memory (MiB), ratio to A")
    for key in (6, 20):
        # vLLM oracle carries total wall + (derivable) total throughput only
        a100_figure("wall", f"ablation_vs_vllm_{key}k_total", "Total latency, ratio to vLLM (log scale)", key)
        a100_figure("tok_s", f"ablation_vs_vllm_{key}k_tok_s", "Throughput (tok/s), ratio to vLLM", key)
        # phase/memory metrics: dense-eager normalized (no vLLM phase data)
        a100_figure("prefill", f"ablation_vs_dense_eager_{key}k_prefill", "Prefill forward, ratio to dense eager (log scale)", key)
        a100_figure("decode", f"ablation_vs_dense_eager_{key}k_decode", "Decode forward, ratio to dense eager (log scale)", key)
        a100_figure("peak", f"ablation_vs_dense_eager_{key}k_peak", "Peak memory (MiB), ratio to dense eager", key)
    chunk_figure()
    print("all meeting figures written to", OUT)
