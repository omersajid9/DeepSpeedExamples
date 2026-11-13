# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Relabeled meeting figures: prefill / decode / total e2e across models and tasks.

Usage: python scripts/plot_meeting_v2.py [OUT-DIR]
Writes three figures (PDF + PNG) to OUT-DIR (default
runs/meeting_figs_2026-11-12_v2):
  prefill_ms   prefill forward latency, ms (log scale)
  decode_ms    decode forward latency, ms (log scale)
  total_e2e_ms end-to-end latency, ms (log scale), includes the vLLM oracle bar
Each figure: rows = Qwen3-4B / Qwen3-8B, columns = 6k / 20k / 3k task shapes.
Levels use the short names: Dense Eager, CUDA Graph, Ladder, Paged, Fused,
FlashInfer, vLLM. Missing cells (still running) are omitted with a footnote.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OPSD = Path(__file__).resolve().parent.parent
OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else OPSD / "runs" / "meeting_figs_2026-11-12_v2"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "axes.edgecolor": "0.35",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# (task key, column label, raw dir, vllm oracle suffix)
TASKS = [
    ("6k", "6k: 6144 → 16 new, C=4", "cb_ablation_2026-11-12", "6k_c4"),
    ("20k", "20k: 20480 → 8 new, C=2", "cb_ablation_2026-11-12", "20k_c2"),
    ("3k", "3k: 3072 → 3072 new, C=4", "3k3k_2026-11-12", "3k_c4"),
]
MODELS = [("qwen3_4b", "4b", "Qwen3-4B"), ("qwen3_8b", "8b", "Qwen3-8B")]
LEVELS = [
    ("A_eager", "Dense Eager", "#4e79a7"),
    ("B_single_graph", "CUDA Graph", "#f28e2b"),
    ("C_multi_graph", "Ladder", "#e15759"),
    ("D_paged", "Paged", "#76b7b2"),
    ("E_fused", "Fused", "#59a14f"),
    ("F_flashinfer_sampler", "FlashInfer", "#edc949"),
]
VLLM_NAME, VLLM_COLOR = "vLLM", "#999999"


def load_cell(dir_name, model, task, level):
    p = OPSD / "runs" / dir_name / "raw" / f"{model}_{task}_{level}.json"
    if not p.exists():
        return None
    it = json.loads(p.read_text())["iterations"][0]
    prof = it.get("profile", {})
    return {"wall": it.get("wall_ms"), "prefill": prof.get("prefill_forward_ms"),
            "decode": prof.get("decode_forward_ms")}


def load_vllm(dir_name, short_model, suffix):
    p = OPSD / "runs" / dir_name / "vllm_oracles" / f"vllm_{short_model}_{suffix}.json"
    if not p.exists():
        return None
    return {"wall": json.loads(p.read_text())["iterations"][0]["wall_ms"]}


def fmt_ms(v):
    return f"{v / 1000.0:.2f}s" if v >= 1000 else f"{v:.0f}ms"


def plot(metric, fname, suptitle, include_vllm):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.8))
    missing = []
    for r, (mkey, mshort, mname) in enumerate(MODELS):
        for c, (task, tlabel, tdir, vkey) in enumerate(TASKS):
            ax = axes[r, c]
            names, vals, colors = [], [], []
            for lv, name, color in LEVELS:
                cell = load_cell(tdir, mkey, task, lv)
                v = cell[metric] if cell else None
                if v is None:
                    missing.append(f"{mname} {task} {name}")
                    continue
                names.append(name)
                vals.append(v)
                colors.append(color)
            if include_vllm:
                vcell = load_vllm(tdir, mshort, vkey)
                v = vcell["wall"] if vcell else None
                if v is None:
                    missing.append(f"{mname} {task} vLLM")
                else:
                    names.append(VLLM_NAME)
                    vals.append(v)
                    colors.append(VLLM_COLOR)
            if not vals:
                ax.set_visible(False)
                ax.set_title(f"{tlabel}\n(none yet)")
                continue
            bars = ax.bar(range(len(vals)), vals, width=0.72, color=colors,
                          edgecolor="0.3", linewidth=0.5, zorder=3)
            for b in bars:
                ax.annotate(fmt_ms(b.get_height()),
                            (b.get_x() + b.get_width() / 2, b.get_height()),
                            textcoords="offset points", xytext=(0, 2),
                            ha="center", fontsize=7.5)
            ax.set_yscale("log")
            ax.set_ylim(bottom=min(vals) * 0.35)
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels(names, rotation=18, ha="right", fontsize=7.5)
            if c == 0:
                ax.set_ylabel(mname, fontsize=10.5)
            if r == 0:
                ax.set_title(tlabel, fontsize=10)
    fig.suptitle(suptitle, fontsize=13)
    note = f"Omitted (still running): {', '.join(missing)}" if missing else \
        "A100-SXM4-40GB, bf16, seed 0, one measured iteration; vLLM oracle at matching concurrency"
    fig.text(0.99, 0.005, note, ha="right", fontsize=7, color="0.4")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{fname}.{ext}")
    plt.close(fig)
    print(f"wrote {fname}.pdf/.png")


def plot_ratio_vllm(fname="total_e2e_vs_vllm"):
    """Total e2e latency as a ratio to the vLLM oracle (vLLM = 1.0)."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.8))
    for r, (mkey, mshort, mname) in enumerate(MODELS):
        for c, (task, tlabel, tdir, vkey) in enumerate(TASKS):
            ax = axes[r, c]
            base = load_vllm(tdir, mshort, vkey)
            if base is None:
                ax.set_visible(False)
                ax.set_title(f"{tlabel}\n(no vLLM oracle)")
                continue
            names, vals, colors = ["vLLM"], [1.0], [VLLM_COLOR]
            for lv, name, color in LEVELS:
                cell = load_cell(tdir, mkey, task, lv)
                v = cell["wall"] if cell else None
                if v is not None:
                    names.insert(-1, name)
                    vals.insert(-1, v / base["wall"])
                    colors.insert(-1, color)
            bars = ax.bar(range(len(vals)), vals, width=0.72, color=colors,
                          edgecolor="0.3", linewidth=0.5, zorder=3)
            for b in bars:
                ax.annotate(f"{b.get_height():.2f}x",
                            (b.get_x() + b.get_width() / 2, b.get_height()),
                            textcoords="offset points", xytext=(0, 2),
                            ha="center", fontsize=7.5)
            ax.axhline(1.0, color="0.3", linestyle="--", linewidth=0.8, zorder=2)
            ax.set_yscale("log")
            ax.set_ylim(bottom=max(0.7, min(vals) * 0.55))
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels(names, rotation=18, ha="right", fontsize=7.5)
            if c == 0:
                ax.set_ylabel(mname, fontsize=10.5)
            if r == 0:
                ax.set_title(tlabel, fontsize=10)
    fig.suptitle("Total end-to-end latency, ratio to vLLM at matching concurrency (vLLM = 1.0)",
                 fontsize=13)
    fig.text(0.99, 0.005,
             "A100-SXM4-40GB, bf16, seed 0, one measured iteration; vLLM oracle at matching "
             "concurrency",
             ha="right", fontsize=7, color="0.4")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{fname}.{ext}")
    plt.close(fig)
    print(f"wrote {fname}.pdf/.png")


if __name__ == "__main__":
    plot("prefill", "prefill_ms", "Prefill forward latency by level and task", include_vllm=False)
    plot("decode", "decode_ms", "Decode forward latency by level and task", include_vllm=False)
    plot("wall", "total_e2e_ms", "Total end-to-end latency by level and task (incl. vLLM oracle)",
         include_vllm=True)
    plot_ratio_vllm()
    print("done:", OUT)
