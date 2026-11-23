"""Speed/accuracy comparison of open-source RCWA / FMM codes on the problem of gpu_compare/bench_gpu.py.

Problem: freeform grey pattern (64 x 64), period 1.2 um, eps 4.0 layer (eps = 1 + 3 rho) on an eps 2.125 substrate, light from air,
(8, 8) orders (17 x 17 = 289 harmonics, 578 x 578 layer matrices), 9 wavelengths 0.5-0.7 um x 16 directions = 144 solves,
two orthogonal incident polarisations, thickness 0.25 and 1.0 um.
Output: zeroth-order transmission for unpolarised light, T00 = (T00_a + T00_b) / 2 for two power-orthogonal inputs (basis free),
per solve.  Scalar J = sum over solves and both polarisations of T00 (as in bench_gpu.py); gradient dJ/drho (64 x 64).
Reference: SPECTRA's classical eigendecomposition RCWA in complex128 (value) and SPECTRA's complex128 gradient (ref mode).

One process per (code, thickness):  python bench_rcwa.py CODE THICK OUT.json [--quick] [--dev cuda|cpu]
                                   python bench_rcwa.py ref THICK refdir [--quick]
Every adapter calls the documented public API of its code; no third-party solver is modified.
"""
import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EPS_LAYER, EPS_SUB, PERIOD, GRID = 4.0, 2.125, 1.2, 64
# Fixed orientation map of the 64 x 64 grid for each code, found once on the CPU with tiny sizes (see orient.json).
ORIENT = {}
if (HERE / "orient.json").exists():
    ORIENT = json.loads((HERE / "orient.json").read_text())


def problem(quick):
    orders = 2 if quick else 8
    wls = [float(w) for w in np.linspace(0.50, 0.70, 2 if quick else 9)]
    th = [5.0, 12.0] if quick else [3.0, 7.0, 11.0, 15.0]
    ph = [20.0, 110.0] if quick else [20.0, 110.0, 200.0, 290.0]
    incs = [(math.radians(t), math.radians(p)) for t in th for p in ph]
    return orders, wls, incs                                   # solve index = wl * len(incs) + inc


def grey_pattern():
    yy, xx = np.meshgrid(np.linspace(-1, 1, GRID), np.linspace(-1, 1, GRID), indexing="ij")
    rho = ((xx - 0.2) ** 2 + (yy + 0.1) ** 2 < 0.45).astype(np.float64)
    rho[: GRID // 3, GRID // 2:] = 1.0
    return 0.1 + 0.8 * rho


def orient(x, key, lib):
    """Dihedral map of the grid: 't' transpose, 'f0' flip axis 0, 'f1' flip axis 1 (applied in this order)."""
    for op in ORIENT.get(key, "").split(","):
        if op == "t":
            x = x.T
        elif op == "f0":
            x = x[::-1, :] if lib is None else lib.flip(x, (0,))
        elif op == "f1":
            x = x[:, ::-1] if lib is None else lib.flip(x, (1,))
    return x


# ------------------------------------------------------------------ SPECTRA (reference and timed code)
def spectra(wls, incs, orders, thick, dev, dtype_name="complex64"):
    import torch
    sys.path.insert(0, str(HERE.parent / "gpu_compare"))
    import bench_gpu as bg
    bg.DEV = dev
    dt = getattr(torch, dtype_name)
    fwd = bg.ours_forward(wls, incs, (GRID, GRID), (orders, orders), dt)
    z = ((2 * orders + 1) ** 2) // 2
    rdt = torch.float64 if dt == torch.complex128 else torch.float32

    def val(rho):
        with torch.no_grad():
            return bg.ours_t(fwd, torch.as_tensor(rho, dtype=rdt), thick)[:, z].mean(-1).double().cpu().numpy()

    def grad(rho):
        r = torch.as_tensor(rho, dtype=rdt, device=dev).clone().requires_grad_(True)
        return torch.autograd.grad(bg.ours_t(fwd, r, thick)[:, z].sum(), r)[0].double().cpu().numpy()
    import spectra_rcwa
    meta = dict(version=getattr(spectra_rcwa, "__version__", "1.1.0"), precision=dtype_name, gradient="adjoint field (closed form)",
                formulation="Laurent (E, E^-1), FFT of samples, rectangular truncation")
    return val, grad, meta


def reference(wls, incs, orders, thick, dev):
    import torch
    sys.path.insert(0, str(HERE.parent / "gpu_compare"))
    import bench_gpu as bg
    bg.DEV = dev
    f128 = bg.ours_forward(wls, incs, (GRID, GRID), (orders, orders), torch.complex128)
    z = ((2 * orders + 1) ** 2) // 2
    rho = torch.as_tensor(grey_pattern())
    with torch.no_grad():
        t_ref = bg.ours_eig_t(f128, rho, thick)[:, z].double().cpu().numpy()          # [solves, 2]
    x = rho.clone().to(dev).requires_grad_(True)
    g_ref = torch.autograd.grad(bg.ours_t(f128, x, thick)[:, z].sum(), x)[0].double().cpu().numpy()
    return t_ref.mean(-1), g_ref, t_ref


# ------------------------------------------------------------------ fmmax (JAX)
def fmmax_code(wls, incs, orders, thick, dev, formulation="FFT"):
    import jax
    import jax.numpy as jnp
    import fmmax
    plv = fmmax.LatticeVectors(u=PERIOD * fmmax.X, v=PERIOD * fmmax.Y)
    exp = fmmax.generate_expansion(plv, approximate_num_terms=(2 * orders + 1) ** 2, truncation=fmmax.Truncation.PARALLELOGRAMIC)
    assert exp.num_terms == (2 * orders + 1) ** 2, exp.num_terms
    n = exp.num_terms
    form = getattr(fmmax.Formulation, formulation)
    wl = jnp.asarray(wls)[:, None]                                                  # [W, 1]
    th = jnp.asarray([t for t, _ in incs])[None, :]
    ph = jnp.asarray([p for _, p in incs])[None, :]
    kp = fmmax.plane_wave_in_plane_wavevector(wl, th, ph, jnp.asarray(1.0))         # [W, D, 2]
    # four input columns e1, e2, e1 + e2, e1 + i e2 on the zeroth order -> 2 x 2 flux forms (basis free)
    cols = np.zeros((2 * n, 4), complex)
    cols[0, 0] = 1; cols[n, 1] = 1; cols[0, 2] = 1; cols[n, 2] = 1; cols[0, 3] = 1; cols[n, 3] = 1j
    cols = jnp.asarray(cols)

    def forms(f):
        tot = f[..., 0, :] + f[..., n, :]                                           # zeroth order, both field components
        m11, m22 = tot[..., 0], tot[..., 1]
        re = (tot[..., 2] - m11 - m22) / 2
        im = -(tot[..., 3] - m11 - m22) / 2
        return m11, m22, re + 1j * im

    def t00(rho):
        eps = 1.0 + (EPS_LAYER - 1.0) * orient(rho, "fmmax", jnp)
        amb = fmmax.eigensolve_isotropic_media(wl, kp, plv, jnp.ones((1, 1)), exp, formulation=form)
        lay = fmmax.eigensolve_isotropic_media(wl, kp, plv, eps, exp, formulation=form)
        sub = fmmax.eigensolve_isotropic_media(wl, kp, plv, jnp.full((1, 1), EPS_SUB), exp, formulation=form)
        s = fmmax.stack_s_matrix([amb, lay, sub], [0.0, thick, 0.0])
        a_out = s.s11 @ cols
        fin, _ = fmmax.amplitude_poynting_flux(jnp.broadcast_to(cols, a_out.shape), jnp.zeros_like(a_out), amb)
        fout, _ = fmmax.amplitude_poynting_flux(a_out, jnp.zeros_like(a_out), sub)
        m11, m22, m12 = forms(fout)
        p11, p22, p12 = forms(fin)
        det = p11 * p22 - jnp.abs(p12) ** 2
        return ((p22 * m11 + p11 * m22 - 2 * jnp.real(jnp.conj(p12) * m12)) / det / 2).reshape(-1)

    jt = jax.jit(t00)
    jg = jax.jit(jax.grad(lambda r: 2 * jnp.sum(t00(r))))
    rdt = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

    def val(rho):
        return np.asarray(jax.block_until_ready(jt(jnp.asarray(rho, rdt))), np.float64)

    def grad(rho):
        return np.asarray(jax.block_until_ready(jg(jnp.asarray(rho, rdt))), np.float64)
    meta = dict(version=fmmax.__version__.lstrip("v"), precision="complex128" if jax.config.jax_enable_x64 else "complex64",
                gradient="jax autodiff, custom eig VJP", formulation=f"Formulation.{formulation}, parallelogramic truncation (289 terms)",
                jax=jax.__version__, eig_backend="jeig" if fmmax.eig._JEIG_AVAILABLE else "jax.lax.linalg.eig")
    return val, grad, meta


# ------------------------------------------------------------------ Meent (torch or JAX backend)
def meent_code(wls, incs, orders, thick, dev, backend="torch"):
    import meent
    from importlib.metadata import version
    tc = 1 if os.environ.get("MEENT_C64") else 0                                    # Meent default: complex128
    common = dict(n_top=1.0, n_bot=math.sqrt(EPS_SUB), fto=[orders, orders], period=[PERIOD, PERIOD], thickness=[thick],
                  type_complex=tc)
    if backend == "torch":
        import torch
        mee = meent.call_mee(backend=2, device=1 if dev == "cuda" else 0, **common)
        fdt = torch.float64 if tc == 0 else torch.float32

        def t00_all(rho_t, wl_list=wls):
            n_idx = torch.sqrt(1.0 + (EPS_LAYER - 1.0) * orient(rho_t, "meent", torch))[None]
            out = []
            for w in wl_list:
                for t, p in incs:
                    res = mee.conv_solve(ucell=n_idx, wavelength=w, theta=t, phi=p, psi=None)
                    a, b = res.res_te_inc.de_ti, res.res_tm_inc.de_ti
                    c0, c1 = a.shape[0] // 2, a.shape[1] // 2
                    out.append((a[c0, c1] + b[c0, c1]) / 2)
            return torch.stack(out)

        def val(rho):
            with torch.no_grad():
                return t00_all(torch.as_tensor(rho, dtype=fdt, device=mee.device)).double().cpu().numpy()

        def grad(rho):
            r = torch.as_tensor(rho, dtype=fdt, device=mee.device).clone().requires_grad_(True)
            return sum(torch.autograd.grad(2 * t00_all(r, [w]).sum(), r)[0] for w in wls).double().cpu().numpy()   # one graph per wavelength
        gm = "torch autograd, custom eig backward"
    else:
        import jax
        import jax.numpy as jnp
        mee = meent.call_mee(backend=1, **common)
        fdt = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

        def t00_one(rho_j, w, t, p):
            n_idx = jnp.sqrt(1.0 + (EPS_LAYER - 1.0) * orient(rho_j, "meent", jnp))[None]
            res = mee.conv_solve(ucell=n_idx, wavelength=w, theta=t, phi=p, psi=None)
            a, b = res.res_te_inc.de_ti, res.res_tm_inc.de_ti
            return (a[a.shape[0] // 2, a.shape[1] // 2] + b[b.shape[0] // 2, b.shape[1] // 2]) / 2

        def val(rho):
            r = jnp.asarray(rho, fdt)
            return np.asarray([float(t00_one(r, w, t, p)) for w in wls for t, p in incs])

        def grad(rho):
            r = jnp.asarray(rho, fdt)
            g = None
            for w in wls:
                for t, p in incs:
                    gi = jax.grad(lambda x: 2 * jnp.real(t00_one(x, w, t, p)))(r)
                    g = gi if g is None else g + gi
            return np.asarray(g, np.float64)
        gm = "jax autodiff"
    meta = dict(version=version("meent"), precision="complex128" if tc == 0 else "complex64", gradient=gm,
                formulation="Li inverse rule (fourier_type 0, enhanced_dfs), rectangular truncation, ETM (TMM)",
                backend=backend, loop="one call per wavelength and direction (no batching API)")
    return val, grad, meta


# ------------------------------------------------------------------ grcwa (autograd, CPU)
def grcwa_code(wls, incs, orders, thick, dev):
    import grcwa
    from importlib.metadata import version
    grcwa.set_backend("autograd")
    import autograd
    import autograd.numpy as anp
    nG = (2 * orders + 1) ** 2

    def t00_all(rho, wl_list=wls):
        eps = (1.0 + (EPS_LAYER - 1.0) * orient(rho, "grcwa", anp)).flatten()
        out = []
        for w in wl_list:
            for t, p in incs:
                obj = grcwa.obj(nG, [PERIOD, 0], [0, PERIOD], 1.0 / w, t, p, verbose=0)
                obj.Add_LayerUniform(0.0, 1.0)
                obj.Add_LayerGrid(thick, GRID, GRID)
                obj.Add_LayerUniform(0.0, EPS_SUB)
                obj.Init_Setup(Gmethod=1)
                obj.GridLayer_geteps(eps)
                z = int(np.where((obj.G[:, 0] == 0) & (obj.G[:, 1] == 0))[0][0])
                tt = 0.0
                for pa, sa in ((1.0, 0.0), (0.0, 1.0)):
                    obj.MakeExcitationPlanewave(pa, 0, sa, 0, order=0)
                    _, ti = obj.RT_Solve(normalize=1, byorder=1)
                    tt = tt + ti[z]
                out.append(tt / 2)
        return anp.stack(out) if hasattr(anp, "stack") else anp.array(out)

    def val(rho):
        return np.asarray(t00_all(np.asarray(rho, np.float64)), np.float64)

    def grad(rho):
        r0 = np.asarray(rho, np.float64)
        return sum(autograd.grad(lambda r: 2 * anp.sum(t00_all(r, [w])))(r0) for w in wls)        # one graph per wavelength
    meta = dict(version=version("grcwa"), precision="complex128", gradient="autograd (HIPS), reverse mode",
                formulation="Laurent (FFT of grid, E^-1 = inv(E)), parallelogramic truncation (Gmethod=1)",
                loop="one object per wavelength and direction")
    return val, grad, meta


# ------------------------------------------------------------------ inkstone (numpy, CPU, shapes only)
def runs_of(mask):
    """Row runs of a boolean grid -> list of (i, j0, j1) with mask[i, j0:j1] True."""
    out = []
    for i in range(mask.shape[0]):
        j = 0
        while j < mask.shape[1]:
            if mask[i, j]:
                j0 = j
                while j < mask.shape[1] and mask[i, j]:
                    j += 1
                out.append((i, j0, j))
            else:
                j += 1
    return out


def inkstone_code(wls, incs, orders, thick, dev):
    from inkstone import Inkstone
    from importlib.metadata import version
    nG = (2 * orders + 1) ** 2
    rho0 = grey_pattern()

    def build(rho):
        r = orient(rho, "inkstone", None)
        lo, hi = float(r.min()), float(r.max())
        assert np.all((np.abs(r - lo) < 1e-12) | (np.abs(r - hi) < 1e-12)), "inkstone adapter needs a two-level pattern"
        s = Inkstone()
        s.lattice = ((PERIOD, 0), (0, PERIOD))
        s.num_g = nG
        s.AddMaterial("bg", 1.0 + (EPS_LAYER - 1.0) * lo)
        s.AddMaterial("hi", 1.0 + (EPS_LAYER - 1.0) * hi)
        s.AddMaterial("sub", EPS_SUB)
        s.AddLayer("in", 0, "vacuum")
        s.AddLayer("slab", thick, "bg")
        s.AddLayer("out", 0, "sub")
        d = PERIOD / GRID
        for k, (i, j0, j1) in enumerate(runs_of(np.abs(r - hi) < 1e-12)):     # pixel rows as exact rectangles
            s.AddPattern("slab", "hi", "rectangle", pattern_name=f"r{k}", side_lengths=((j1 - j0) * d, d),
                         center=((j0 + j1) / 2 * d, (i + 0.5) * d))
        return s

    def val(rho):
        s = build(rho)
        out = []
        for w in wls:
            s.frequency = 1.0 / w
            for t, p in incs:
                tt = 0.0
                for sa, pa in ((1.0, 0.0), (0.0, 1.0)):
                    s.SetExcitation(theta=math.degrees(t), phi=math.degrees(p), s_amplitude=sa, p_amplitude=pa)
                    fi, _ = s.GetPowerFlux("in")
                    sf, _ = s.GetPowerFluxByOrder("out", (0, 0), 0)
                    tt += float(np.ravel(sf)[0]) / float(np.ravel(fi)[0])
                out.append(tt / 2)
        return np.asarray(out)
    meta = dict(version=version("inkstone"), precision="complex128", gradient="none",
                formulation="exact Fourier series of pixel rectangles (no grid input), circular truncation (num_g 289)",
                loop="one solve per wavelength and direction")
    return val, None, meta


# ------------------------------------------------------------------ rcwa (Rumpf-style, numpy, CPU)
def rcwa_code(wls, incs, orders, thick, dev):
    import rcwa
    from importlib.metadata import version

    def val(rho):
        er = 1.0 + (EPS_LAYER - 1.0) * orient(rho, "rcwa", None)
        out = []
        for w in wls:
            for t, p in incs:
                tt = 0.0
                for pol in ([1, 0], [0, 1]):
                    crystal = rcwa.Crystal([PERIOD, 0], [0, PERIOD], er=np.ascontiguousarray(er), ur=np.ones_like(er))
                    slab = rcwa.Layer(thickness=thick, crystal=crystal)
                    stack = rcwa.LayerStack(slab, incident_layer=rcwa.Layer(er=1.0), transmission_layer=rcwa.Layer(er=EPS_SUB))
                    src = rcwa.Source(wavelength=w, theta=t, phi=p, pTEM=pol, layer=stack.incident_layer)
                    res = rcwa.Solver(stack, src, n_harmonics=(2 * orders + 1, 2 * orders + 1)).solve()
                    tm = np.asarray(res["T"])
                    tt += float(tm.reshape(2 * orders + 1, 2 * orders + 1)[orders, orders])
                out.append(tt / 2)
        return np.asarray(out)
    meta = dict(version=version("rcwa"), precision="complex128", gradient="none",
                formulation="Rumpf (Laurent, FFT convolution of grid), rectangular truncation", loop="one Solver per wavelength, direction, polarisation")
    return val, None, meta


# ------------------------------------------------------------------ torcwa (torch)
def torcwa_code(wls, incs, orders, thick, dev):
    import torch
    import torcwa
    from importlib.metadata import version
    cdt, rdt = torch.complex64, torch.float32
    torcwa.rcwa_geo.dtype, torcwa.rcwa_geo.device = rdt, torch.device(dev)

    def t00_all(r, wl_list=wls):
        eps = 1.0 + (EPS_LAYER - 1.0) * orient(r, "torcwa", torch)
        out = []
        for w in wl_list:
            for t, p in incs:
                sim = torcwa.rcwa(freq=1.0 / w, order=[orders, orders], L=[PERIOD, PERIOD], dtype=cdt, device=torch.device(dev))
                sim.add_input_layer(eps=1.0)
                sim.add_output_layer(eps=EPS_SUB)
                sim.set_incident_angle(inc_ang=t, azi_ang=p)
                sim.add_layer(thickness=thick, eps=eps)
                sim.solve_global_smatrix()
                tt = 0.0
                for po in ("s", "p"):
                    for pi in ("s", "p"):
                        tt = tt + torch.abs(sim.S_parameters(orders=[0, 0], direction="forward", port="transmission",
                                                              polarization=po + pi, ref_order=[0, 0], power_norm=True)) ** 2
                out.append(tt / 2)
        return torch.stack(out)

    def val(rho):
        with torch.no_grad():
            return t00_all(torch.as_tensor(rho, dtype=rdt, device=dev)).double().cpu().numpy().ravel()

    def grad(rho):
        r = torch.as_tensor(rho, dtype=rdt, device=dev).clone().requires_grad_(True)
        return sum(torch.autograd.grad(2 * t00_all(r, [w]).sum(), r)[0] for w in wls).double().cpu().numpy()   # one graph per wavelength
    meta = dict(version=version("torcwa"), precision="complex64", gradient="torch autograd (eig backward with broadening)",
                formulation="Laurent (E, E^-1), FFT of grid, rectangular truncation", loop="one rcwa object per wavelength and direction")
    return val, grad, meta


CODES = dict(spectra=spectra, fmmax=fmmax_code, fmmax_jones=lambda *a: fmmax_code(*a, formulation="JONES_DIRECT_FOURIER"),
             meent_torch=lambda *a: meent_code(*a, backend="torch"), meent_jax=lambda *a: meent_code(*a, backend="jax"),
             grcwa=grcwa_code, inkstone=inkstone_code, rcwa=rcwa_code, torcwa=torcwa_code)


# ------------------------------------------------------------------ timing and memory
def peak_host_gib():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    except ImportError:
        import psutil
        return psutil.Process().memory_info().peak_wset / 2**30


def gpu_peak(code):
    if code.startswith("fmmax") or code == "meent_jax":
        import jax
        st = jax.devices()[0].memory_stats() or {}
        return st.get("peak_bytes_in_use", float("nan")) / 2**30
    import torch
    return torch.cuda.max_memory_allocated() / 2**30


def timed(fn, x, dev, max_total_s, once=False):
    """Warm-up call (compilation for JAX), then repeats chosen from its duration; returns median and all."""
    import torch
    sync = (lambda: torch.cuda.synchronize()) if dev == "cuda" and torch.cuda.is_available() else (lambda: None)
    sync(); t0 = time.perf_counter(); out = fn(x); sync(); first = time.perf_counter() - t0
    reps = 5 if first < 3 else 3 if first < 30 else 1
    if first * (reps + 1) > max_total_s:
        reps = 1
    if once:
        return out, dict(first_call_s=first, median_s=first, all_s=[first], untimed_accuracy_run=True)
    ts = []
    for _ in range(reps):
        sync(); t0 = time.perf_counter(); out = fn(x); sync(); ts.append(time.perf_counter() - t0)
    return out, dict(first_call_s=first, median_s=statistics.median(ts), all_s=ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("code")
    ap.add_argument("thick", type=float)
    ap.add_argument("out")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dev", default="cpu")
    ap.add_argument("--no-grad", action="store_true")
    ap.add_argument("--max-s", type=float, default=600.0, help="time budget per mode for the repeats")
    ap.add_argument("--once", action="store_true", help="one call per mode, for accuracy only")
    ap.add_argument("--nsolve", type=int, default=0, help="time only the first N directions at the first wavelength")
    a = ap.parse_args()
    orders, wls, incs = problem(a.quick)
    if a.nsolve:
        wls, incs = wls[:1], incs[:a.nsolve]
    refdir = Path(a.out).parent
    tag = f"{'q' if a.quick else 'f'}_{a.thick:g}"
    if a.code == "ref":
        t, g, tp = reference(wls, incs, orders, a.thick, a.dev)
        np.savez(refdir / f"ref_{tag}.npz", t=t, g=g, t_pol=tp)
        print("ref saved", t[:4], float(np.linalg.norm(g)), flush=True)
        return
    ref = dict(np.load(refdir / f"ref_{tag}.npz"))
    if a.nsolve:
        ref["t"] = ref["t"][:a.nsolve]
    if a.dev == "cuda":
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.8)
    rho = grey_pattern()
    row = dict(code=a.code, thickness_um=a.thick, device=a.dev, solves=len(wls) * len(incs), orders=orders, quick=a.quick,
               subset=bool(a.nsolve))

    def save():
        row["peak_host_gib"] = peak_host_gib()
        Path(a.out).write_text(json.dumps(row, indent=1))
    t_start = time.perf_counter()
    val, grad, meta = CODES[a.code](wls, incs, orders, a.thick, a.dev)
    row.update(meta, setup_s=time.perf_counter() - t_start)
    save()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v, tv = timed(val, rho, a.dev, a.max_s, a.once)
        row.update(value=tv, value_error=float(np.max(np.abs(v - ref["t"]))), finite=bool(np.isfinite(v).all()),
                   t00_first=[float(x) for x in v[:4]])
        if a.dev == "cuda":
            row["peak_gpu_gib_value"] = gpu_peak(a.code)
        save()
        print(f"{a.code} {a.thick} value {tv['median_s']:.3f} s err {row['value_error']:.2e}", flush=True)
        if grad is not None and not a.no_grad:
            g, tg = timed(grad, rho, a.dev, a.max_s, a.once)
            row.update(value_grad=tg, gradient_error=None if a.nsolve else float(np.linalg.norm(g - ref["g"]) / np.linalg.norm(ref["g"])))
            if a.dev == "cuda":
                row["peak_gpu_gib_value_grad"] = gpu_peak(a.code)
            save()
            print(f"{a.code} {a.thick} value+grad {tg['median_s']:.3f} s grad err {row['gradient_error']}", flush=True)
    row["environment"] = dict(python=sys.version.split()[0], platform=platform.platform(), cpu_count=os.cpu_count(),
                              omp=os.environ.get("OMP_NUM_THREADS"))
    save()


if __name__ == "__main__":
    main()
