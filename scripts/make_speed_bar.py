"""README speed figure: time for value + gradient of 144 solves (578 x 578 layer matrices) on one RTX 5090, read from benchmarks/results.
Writes docs/figures/speed_comparison.png and .svg.   usage: python scripts/make_speed_bar.py"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1] / "benchmarks" / "results"
OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"


def j(name):
    return json.loads((R / name).read_text(encoding="utf8"))


ours = j("rtx5090_torcwa.json")["this_work"]["default"]["value_grad_s"]
ours_rdit = next(r["value_grad_s"] for b in j("torchrdit_rtx5090_025um.json")["blocks"] for r in b["rows"] if r["code"] == "SPECTRA")
rows = [
    ("SPECTRA-RCWA", ours, "#0072B2"),
    ("TorchRDIT 0.2.0", next(r["value_grad_s"] for b in j("torchrdit_rtx5090_025um.json")["blocks"] for r in b["rows"]
                             if r["code"] == "TorchRDIT" and "order 15" in r["setting"]), "#9a9a9a"),
    ("torcwa 0.1.4.2", j("rtx5090_torcwa.json")["value_grad"]["checkpoint_per_wavelength_default_threads"]["value_grad_s"], "#9a9a9a"),
    ("S4 (value only,\n12 CPU cores)", j("summary.json")["s4_cpu"]["value_s_12_processes"], "#c8c8c8"),
]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(figsize=(7.2, 3.3), dpi=200)
names = [r[0] for r in rows][::-1]
times = [r[1] for r in rows][::-1]
cols = [r[2] for r in rows][::-1]
ax.barh(names, times, color=cols, height=0.62)
ax.set_xscale("log")
ax.set_xlim(0.1, 6000)
for y, t in enumerate(times):
    label = f"{t:.2f} s" if t < 1 else f"{t:.1f} s" if t < 100 else f"{t:.0f} s"
    if t != ours and "S4" not in names[y]:               # S4 gives the value alone, on a CPU: no ratio
        base = ours_rdit if "TorchRDIT" in names[y] else ours                # each code against SPECTRA-RCWA on the same problem
        label += f"   ({t / base:.0f}x slower)"
    ax.text(t * 1.12, y, label, va="center", fontsize=10, fontweight="bold" if t == ours else "normal")
ax.set_xlabel("Time for value and gradient, 144 solves, one RTX 5090 (s)")
for s in ax.spines.values():
    s.set_visible(True)
ax.tick_params(axis="y", length=0)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "speed_comparison.png")
fig.savefig(OUT / "speed_comparison.svg")
print([(r[0].replace("\n", " "), round(r[1], 3)) for r in rows])
