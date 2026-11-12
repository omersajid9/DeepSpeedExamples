# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Meeting-style figures for a long-generation CB study.

Usage: plot_longgen_figures.py [OUT-DIR] [TASK] [PREFIX] [LABEL] [TITLE]
Defaults reproduce the longgen study (runs/longgen_2026-11-12, task 6k,
prefix "longgen"). The 3k study runs it as:
    python scripts/plot_longgen_figures.py runs/3k3k_2026-11-12 3k 3k3k \
        "3k prompt / 3k new (C=4)" "CB ablation at 3k prompt / 3k decode, A100"

Same 6-step ladder (A_eager ... F_flashinfer_sampler) plus vLLM oracles at the
same concurrency (vllm_oracles/vllm_{4b,8b}_{TASK}_c4.json).

Figures (<OUT>/figs/, PDF + PNG):
  {PREFIX}_levels_vs_eager_{latency,prefill,decode,tok_s,peak}
  {PREFIX}_vs_vllm_{total,tok_s}            (vLLM = 1.0, gray bar)
  {PREFIX}_vs_vllm_{prefill,decode}_vs_A    (vLLM has no phase data)
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OPSD = Path(__file__).resolve().parent.parent
OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else OPSD / "runs" / "longgen_2026-11-12"
TASK = sys.argv[2] if len(sys.argv) > 2 else "6k"
PREFIX = sys.argv[3] if len(sys.argv) > 3 else "longgen"
LABEL = sys.argv[4] if len(sys.argv) > 4 else "6k prompt / 6k new (C=4)"
TITLE = sys.argv[5] if len(sys.argv) > 5 else "Long-generation CB ablation, A100"
RAW = OUT / "raw"
VLLM_DIR = OUT / "vllm_oracles"
FIGS = OUT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

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
C_TOTAL = "#4e79a7"

LEVELS = ("A_eager", "B_single_graph", "C_multi_graph",
          "D_paged", "E_fused", "F_flashinfer_sampler")
STEP_LABELS = ["A\neager", "B\ngraph", "C\nladder", "D\npaged", "E\nfused", "F\nFI dec"]
WORKLOADS = [("qwen3_4b", f"Qwen3-4B, {LABEL}", f"vllm_4b_{TASK}_c4"),
             ("qwen3_8b", f"Qwen3-8B, {LABEL}", f"vllm_8b_{TASK}_c4")]


def cell(directory, name):
    p = directory / f"{name}.json"
    d = json.loads(p.read_text())
    it = d["iterations"][0]
    prof = it.get("profile", {})
    peak = max(d.get("peak_memory_mb") or 0.0, d.get("warmup_peak_memory_mb") or 0.0)
    tok_s = prof.get("tokens_per_second")
    if tok_s is None and it.get("wall_ms") and d.get("max_new_tokens"):
        tok_s = d.get("batch_size", 1) * d["max_new_tokens"] / (it["wall_ms"] / 1000.0)
    return {"wall": it.get("wall_ms"), "prefill": prof.get("prefill_forward_ms"),
            "decode": prof.get("decode_forward_ms"), "tok_s": tok_s,
            "peak": peak if peak > 0 else None}


def assert_fresh():
    missing = [f"{m}_{TASK}_{lv}" for m, _l, _v in WORKLOADS for lv in LEVELS
               if not (RAW / f"{m}_{TASK}_{lv}.json").exists()]
    if missing:
        raise SystemExit(f"cells missing: {', '.join(missing)}")


def assert_vllm():
    missing = [v for _m, _l, v in WORKLOADS if not (VLLM_DIR / f"{v}.json").exists()]
    if missing:
        raise SystemExit("vLLM oracles missing: " + ", ".join(missing)
                         + " — run the matching r_vllm_*_oracles.sh first")


def save(fig, name):
    fig.savefig(FIGS / f"{name}.pdf")
    fig.savefig(FIGS / f"{name}.png")
    plt.close(fig)
    print(f"wrote {FIGS}/{name}.pdf/.png")


def annotate(ax, bars):
    for r in bars:
        v = r.get_height()
        if v is not None:
            ax.annotate(f"{v:.2f}", (r.get_x() + r.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 2), ha="center",
                        fontsize=8)


def base_figure(title_suffix, fname, ylabel):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    return fig, axes


def finish(fig, axes, fname, ylabel, title_suffix):
    axes[0].set_ylabel(ylabel)
    fig.suptitle(f"{TITLE} — {title_suffix}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, fname)


# ------------------------------------------------------- vs level A
def vs_eager(metric, fname, ylabel):
    fig, axes = base_figure("", fname, ylabel)
    for ax, (m, label, _v) in zip(axes, WORKLOADS):
        base = cell(RAW, f"{m}_{TASK}_A_eager")[metric]
        vals = [cell(RAW, f"{m}_{TASK}_{lv}")[metric] / base for lv in LEVELS]
        bars = ax.bar(range(6), vals, width=0.62, color=[C_TOTAL] * 6,
                      edgecolor="0.3", linewidth=0.5, zorder=3)
        ax.axhline(1.0, color="0.3", linestyle="--", linewidth=0.8, zorder=2)
        annotate(ax, bars)
        ax.set_yscale("log")
        ax.set_ylim(bottom=max(0.10, min(vals) * 0.75))
        ax.set_xticks(range(6))
        ax.set_xticklabels(STEP_LABELS, fontsize=8)
        ax.set_title(label, fontsize=10.5)
    finish(fig, axes, f"{PREFIX}_levels_vs_eager_{fname}", ylabel,
           "ratio to level A (eager + continuous batching)")


# ------------------------------------------------------- vs vLLM
def vs_vllm(metric, fname, ylabel, use_vllm):
    fig, axes = base_figure("", fname, ylabel)
    for ax, (m, label, vname) in zip(axes, WORKLOADS):
        if use_vllm:
            base = cell(VLLM_DIR, vname)[metric]
            order = list(LEVELS) + ["vllm"]
            labels = STEP_LABELS + ["vLLM\nC=4"]
        else:
            base = cell(RAW, f"{m}_{TASK}_A_eager")[metric]
            order = list(LEVELS)
            labels = STEP_LABELS
        vals = []
        for lv in order:
            c = cell(VLLM_DIR, vname) if lv == "vllm" else cell(RAW, f"{m}_{TASK}_{lv}")
            vals.append(c[metric] / base)
        bars = ax.bar(range(len(order)), vals, width=0.62,
                      color=[C_TOTAL] * (len(order) - 1) + (["#8c8c8c"] if use_vllm else []),
                      edgecolor="0.3", linewidth=0.5, zorder=3)
        ax.axhline(1.0, color="0.3", linestyle="--", linewidth=0.8, zorder=2)
        annotate(ax, bars)
        ax.set_yscale("log")
        ax.set_ylim(bottom=max(0.10, min(vals) * 0.75))
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(label, fontsize=10.5)
    note = ("ratio to vLLM at same concurrency" if use_vllm
            else "vLLM has no phase data — ratio to level A (eager + CB)")
    finish(fig, axes, f"{PREFIX}_vs_vllm_{fname}", ylabel, note)


if __name__ == "__main__":
    assert_fresh()
    vs_eager("wall", "latency", "Total latency, ratio to A (log scale)")
    vs_eager("prefill", "prefill", "Prefill forward, ratio to A (log scale)")
    vs_eager("decode", "decode", "Decode forward, ratio to A (log scale)")
    vs_eager("tok_s", "tok_s", "Throughput (tok/s), ratio to A")
    vs_eager("peak", "peak", "Peak memory (MiB), ratio to A")
    assert_vllm()
    vs_vllm("wall", "total", "Total latency, ratio to vLLM (log scale)", True)
    vs_vllm("tok_s", "tok_s", "Throughput (tok/s), ratio to vLLM", True)
    vs_vllm("prefill", "prefill_vs_A", "Prefill forward, ratio to A (log scale)", False)
    vs_vllm("decode", "decode_vs_A", "Decode forward, ratio to A (log scale)", False)
    print(f"all {PREFIX} figures written to", FIGS)
