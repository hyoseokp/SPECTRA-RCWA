"""Render the README figures and tables from the benchmark JSON files.

Reads only benchmarks/results/rtx5090.json, rtx5090_stages.json, rtx5090_torcwa.json and, for S4, summary.json; writes docs/figures/*.png and *.svg.
No number is typed in here.

usage: python scripts/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "benchmarks" / "results"
OUT = ROOT / "docs" / "figures"

INK, INK2, GRID = "#1a1a1a", "#555555", "#d9d9d9"
BLUE, AQUA, GREY_C, RED = "#2a78d6", "#1baf7a", "#9a9a96", "#c0392b"
STAGE_COLORS = ["#9a9a96", "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#eda100"]

plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"], "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "mathtext.fontset": "dejavusans", "svg.fonttype": "path", "savefig.dpi": 200,
    "legend.frameon": False,
})


def save(fig, stem, **kw):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}", **{"bbox_inches": "tight", "facecolor": "white", **kw})
    plt.close(fig)


def fmt_s(t):
    return f"{t:.3g} s"


# ---------------------------------------------------------------------------------------------- data
S = json.loads((RES / "summary.json").read_text())
R = json.loads((RES / "rtx5090.json").read_text())
T = json.loads((RES / "rtx5090_torcwa.json").read_text())
STAGES = json.loads((RES / "rtx5090_stages.json").read_text())
BASE_S = S["s4_cpu"]["value_s_12_processes"]
TORCWA_S = min(v["value_grad_s"] for v in T["value_grad"].values() if isinstance(v, dict))
THIS_S = R["one_layer"]["grey"]["default"]["value_grad_s"]
TML = R["stack_2um"]


# ----------------------------------------------------------------------------------- stage breakdown
GROUPS = ["assembly (E, A)", "cos, sin", "thin layer", "S-matrix doubling", "boundary, field below", "wells, objective", "adjoint on the planes", "integral over the layer"]
STAGE_GROUP = {"fwd: assemble E, E^-1": 0, "fwd: assemble A = P Q": 0, "fwd: cos, sin": 1, "fwd: thin layer in the reference basis": 2, "fwd: S-matrix doubling": 3, "fwd: boundary": 4, "fwd: solve for the two polarisations": 4,
               "fwd: detector field, wells": 5, "fwd: objective, autograd of the rest": 5, "bwd: fields on the ends, sweeps": 6, "bwd: fields on the planes": 6, "bwd: integral over the layer": 7}


def fig_stages():
    fwd, bwd = [0.0] * len(GROUPS), [0.0] * len(GROUPS)
    for name, ms in STAGES["stages_ms_median"].items():
        (bwd if name.startswith("bwd") else fwd)[STAGE_GROUP[name]] += ms
    colors = STAGE_COLORS + ["#c0392b", "#7a5c3e"]
    bars = [("forward", fwd), ("gradient", bwd)]
    fig, ax = plt.subplots(figsize=(7.2, 1.9))
    y = [1.0, 0.2]
    for yi, (_, vals) in zip(y, bars):
        left = 0.0
        for gi, v in enumerate(vals):
            ax.barh(yi, v, left=left, color=colors[gi], height=0.62, edgecolor="white", linewidth=1.0, label=GROUPS[gi] if v > 0 else None)
            left += v
        ax.text(left + 2, yi, f"{left:.0f} ms", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y, [b[0] for b in bars])
    ax.set_xlabel("time per evaluation on an RTX 5090 (ms)")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(ncol=4, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "stage_breakdown")


# -------------------------------------------------------------------------------------- single solve
def fig_single():
    """Value + gradient per polarisation: 288 of them make an evaluation."""
    sg = T["single"]
    rows = [("torcwa, one solve", sg["torcwa_one_polarisation"]["value_grad_s"] * 1e3, GREY_C),
            ("this work, one solve", sg["default_two_polarisations"]["value_grad_s"] / 2 * 1e3, BLUE),
            ("this work, 144 solves in one batch", THIS_S / 288 * 1e3, BLUE)]
    fig, ax = plt.subplots(figsize=(6.4, 1.9))
    y = list(range(len(rows)))[::-1]
    top = rows[0][1]
    ax.barh(y, [r[1] for r in rows], height=0.62, color=[r[2] for r in rows], edgecolor="white")
    for q, (_, t, _c) in zip(y, rows):
        ax.text(t + 0.012 * max(r[1] for r in rows), q, f"{t:.3g} ms" + ("" if t == top else f"  ({top / t:.0f}x)"), va="center", fontsize=8)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(r[1] for r in rows) * 1.25)
    ax.set_xlabel("value + gradient per polarisation, complex64, RTX 5090 (ms)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "single_solve")


if __name__ == "__main__":
    fig_stages()
    fig_single()
    print("wrote", sorted(p.name for p in OUT.iterdir()))
