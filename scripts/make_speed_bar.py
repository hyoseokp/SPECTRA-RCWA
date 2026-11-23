"""README speed figure: 144 solves with 578 x 578 layer matrices, 0.25 um layer, read from benchmarks/results.
(a) value, every code; (b) value and gradient, the differentiable codes. Ratios against SPECTRA-RCWA on the same problem.
Writes docs/figures/speed_comparison.png and .svg.   usage: python scripts/make_speed_bar.py"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R, OUT = ROOT / "benchmarks" / "results", ROOT / "docs" / "figures"
rows = [r for r in json.loads((R / "other_rcwa_rtx5090.json").read_text(encoding="utf8")) if r["thickness_um"] == 0.25]
summary = json.loads((R / "summary.json").read_text(encoding="utf8"))["s4_cpu"]
cr = json.loads((R / "rtx5090_torcwa.json").read_text(encoding="utf8"))["this_work"]["default"]

LABEL = {"spectra": "SPECTRA-RCWA", "torchrdit": "TorchRDIT", "fmmax": "fmmax", "torcwa": "torcwa", "meent_torch": "Meent (torch)",
         "meent_jax": "Meent (JAX)", "grcwa": "grcwa (CPU)", "inkstone": "inkstone (CPU)", "rcwa": "rcwa (CPU)"}
by = {r["code"]: r for r in rows if r["code"] in LABEL}
ours = by["spectra"]
BLUE, GREY, LIGHT = "#0072B2", "#9a9a9a", "#c8c8c8"


def panel(ax, key, extra=()):
    items = [(LABEL[c], r[key], ours[key], BLUE if c == "spectra" else (LIGHT if "CPU" in r["device"] else GREY))
             for c, r in by.items() if r.get(key)]
    items += list(extra)
    items.sort(key=lambda t: t[1])
    names = [t[0] for t in items][::-1]
    for y, (name, t, base, col) in enumerate(items[::-1]):
        ax.barh(y, t, color=col, height=0.62)
        lab = f"{t:.2f} s" if t < 1 else f"{t:.1f} s" if t < 100 else f"{t:.0f} s"
        if name != "SPECTRA-RCWA":
            lab += f"  ({float(f'{t / base:.2g}'):,.0f}x)"
        ax.text(t * 1.15, y, lab, va="center", fontsize=9, fontweight="bold" if name == "SPECTRA-RCWA" else "normal")
    ax.set_yticks(range(len(names)), names)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 6e4)
    ax.tick_params(axis="y", length=0)
    for s in ax.spines.values():
        s.set_visible(True)


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.0), dpi=200)
panel(a, "value_s", extra=[("S4 (CPU, 12 cores)", summary["value_s_12_processes"], cr["value_s"], LIGHT)])
a.set_xlabel("Value only (s)")
panel(b, "value_grad_s")
b.set_xlabel("Value and gradient (s)")
for ax, t in ((a, "a"), (b, "b")):
    ax.text(-0.02, 1.03, t, transform=ax.transAxes, fontsize=12, fontweight="bold", ha="right")
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "speed_comparison.png")
fig.savefig(OUT / "speed_comparison.svg")
print("ok")
