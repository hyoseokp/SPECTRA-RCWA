"""Speed and accuracy of one evaluation (value and gradient) on a synthetic freeform layer, with nothing outside this package.

The problem has the size of the benchmark in the README: one 2 um x 2 um cell, a 0.8 um layer of index 2.0 in air on glass
(n = 1.46), a 128 x 128 design, (8, 8) Fourier orders (578 x 578 layer matrices), a 16-ray pupil (NA 0.30, chief ray 2.5 deg at
22.5 deg) and 9 wavelengths from 420 to 660 nm: 144 solves, each for both polarisations.  The design is a smoothed random
pattern (fixed seed) and the objective a fixed linear weighting of the four quadrant powers Tabs.  Errors are against complex128
with the converged reference configuration, for the first ray.  The README's tables were measured with another objective and design;
this script reproduces the size of the problem, not those exact numbers.

    python bench_synthetic.py results/synthetic.json            # on the GPU if there is one
    python bench_synthetic.py out.json --small                  # a small case that runs on a CPU in seconds
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

import spectra_rcwa
from spectra_rcwa import DEFAULT, REFERENCE_C128, ColorRouterForward, Stack, shifted_disk_pupil, value_and_grad


def freeform(n: int, seed: int = 0) -> torch.Tensor:
    """Smoothed random pattern in [0.1, 0.9] on an n x n periodic grid."""
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(n) * n
    kernel = np.exp(-(k[:, None] ** 2 + k[None, :] ** 2) / (2 * (n / 16) ** 2))
    field = np.fft.ifft2(np.fft.fft2(rng.standard_normal((n, n))) * kernel).real
    return torch.as_tensor(0.1 + 0.8 * (field > 0))


def problem(small: bool):
    n, orders, wl, radial, azimuth = (32, (3, 3), np.linspace(450, 630, 3), 1, 2) if small else (128, (8, 8), np.linspace(420, 660, 9), 2, 8)
    stack = Stack(period_um=(2.0, 2.0), thickness_um=0.8, density_shape=(n, n), fourier_orders=orders, eps_design=4.0, eps_fill=1.0,
                  eps_in=1.0, eps_out=1.46 ** 2, detector_z_um=1.25)
    rays = shifted_disk_pupil(numerical_aperture=0.30, chief_ray_angle_deg=2.5, chief_ray_azimuth_deg=22.5, radial_order=radial,
                              azimuth_count=azimuth)
    return stack, rays, [float(w) for w in wl], freeform(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stack, rays, wavelengths, rho = problem(args.small)
    weights = torch.linspace(-1.0, 1.0, 4 * len(wavelengths), dtype=torch.float64).reshape(4, -1)

    def objective(tabs):
        return (tabs * weights).sum()

    fwd = ColorRouterForward(stack=stack, rays=rays, wavelengths_nm=wavelengths, device=device, dtype=torch.complex64, config=DEFAULT)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    value_and_grad(fwd, rho, objective)                                  # warm-up: kernel compilation, caches
    times, value_times = [], []
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for _ in range(args.repeats):
        sync()
        t0 = time.perf_counter()
        tabs, value, grad = value_and_grad(fwd, rho, objective)
        sync()
        times.append(time.perf_counter() - t0)
        with torch.no_grad():
            sync()
            t0 = time.perf_counter()
            fwd.response(rho.to(device=device, dtype=torch.float32))
            sync()
            value_times.append(time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if device == "cuda" else None

    one = dict(stack=stack, rays=rays[:1], wavelengths_nm=wavelengths)
    fast = ColorRouterForward(**one, device=device, dtype=torch.complex64, config=DEFAULT)
    ref = ColorRouterForward(**one, device="cpu", dtype=torch.complex128, config=REFERENCE_C128)
    t_fast, _, g_fast = value_and_grad(fast, rho, objective)
    t_ref, _, g_ref = value_and_grad(ref, rho, objective)
    t_fast, g_fast = t_fast.double().cpu(), g_fast.double().cpu()
    out = dict(
        package=spectra_rcwa.__version__, torch=torch.__version__, device=torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
        problem=dict(solves=len(rays) * len(wavelengths), orders=list(stack.fourier_orders), layer_matrix=2 * fwd.n_orders, design=list(stack.density_shape)),
        value_s_median=float(np.median(value_times)), value_grad_s_median=float(np.median(times)), value_grad_s_min=min(times), peak_gib=peak,
        first_ray=dict(tabs_max_rel=float(((t_fast - t_ref).abs() / t_ref.abs().clamp_min(1e-12)).max()),
                       grad_rel_l2=float((g_fast - g_ref).norm() / g_ref.norm())),
    )
    Path(args.output).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
