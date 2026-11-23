"""Grating problems described as plain dictionaries (JSON), for the command line and the MCP server.

A problem names the lattice, the Fourier orders, the wavelengths, the incidence directions, the media above and below,
and the layers, top first, each a pattern of two materials.  :func:`solve` returns the power in every diffraction order
for p and s incidence, :func:`gradient` the derivative of one of them with respect to the density of every layer.
Lengths are in um, wavelengths in nm, angles in degrees; the time dependence is exp(-i w t), so absorption is Im(eps) > 0.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import torch

from .config import DEFAULT
from .forward import ColorRouterForward, Ray, Stack

EXAMPLES = {
    "lamellar_grating": dict(
        description="1D lamellar grating: index 2.0 ridges, period 1 um, depth 0.5 um, fill factor 0.5, in air, 633 nm at 10 degrees",
        problem=dict(period_um=[1.0, 1.0], fourier_orders=[20, 0], wavelengths_nm=[633.0],
                     incidence=[dict(theta_deg=10.0, phi_deg=0.0)], eps_in=1.0, eps_out=1.0, eps_design=4.0, eps_fill=1.0,
                     layers=[dict(thickness_um=0.5, pattern=dict(type="lamellar", fill_factor=0.5, axis="x"))])),
    "pillar_array": dict(
        description="Square array of index 2.0 disks (radius 0.15 um, height 0.6 um, period 0.5 um) on glass, 450 to 650 nm at normal incidence",
        problem=dict(period_um=[0.5, 0.5], fourier_orders=[5, 5], wavelengths_nm=[450.0, 550.0, 650.0],
                     incidence=[dict(theta_deg=0.0, phi_deg=0.0)], eps_in=1.0, eps_out=1.46 ** 2, eps_design=4.0, eps_fill=1.0,
                     layers=[dict(thickness_um=0.6, pattern=dict(type="disk", radius_um=0.15))])),
    "slab": dict(
        description="Uniform 0.37 um film of index 2.0 on glass at normal incidence; checks against the Airy formula",
        problem=dict(period_um=[1.0, 1.0], fourier_orders=[1, 1], wavelengths_nm=[540.0],
                     incidence=[dict(theta_deg=0.0, phi_deg=0.0)], eps_in=1.0, eps_out=1.46 ** 2, eps_design=4.0, eps_fill=1.0,
                     layers=[dict(thickness_um=0.37, pattern=dict(type="uniform", value=1.0))])),
    "two_layer_grating": dict(
        description="Two stacked 1D gratings of TiO2-like index 2.4 with a lateral offset, on glass, with an absorbing film below (n = 2 + 0.1i)",
        problem=dict(period_um=[0.8, 0.8], fourier_orders=[15, 0], wavelengths_nm=[600.0, 650.0],
                     incidence=[dict(theta_deg=0.0, phi_deg=0.0), dict(theta_deg=15.0, phi_deg=0.0)],
                     eps_in=1.0, eps_out=1.46 ** 2, eps_design=2.4 ** 2, eps_fill=1.0,
                     layers=[dict(thickness_um=0.3, pattern=dict(type="lamellar", fill_factor=0.4, axis="x", center=0.3)),
                             dict(thickness_um=0.3, pattern=dict(type="lamellar", fill_factor=0.4, axis="x", center=0.6)),
                             dict(thickness_um=0.05, pattern=dict(type="uniform", value=1.0), eps_design=dict(n=2.0, k=0.1))])),
}

PERMITTIVITY = {"description": "Relative permittivity: a number, [real, imag], or {'n': index, 'k': extinction} (eps = (n + ik)^2)",
                "anyOf": [{"type": "number"}, {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                          {"type": "object", "properties": {"n": {"type": "number"}, "k": {"type": "number"}}, "required": ["n"]}]}

SCHEMA = {
    "type": "object",
    "description": "A periodic stack: superstrate (eps_in, lossless) / layers, top first / substrate (eps_out). Light comes from the superstrate. "
                   "Lengths in um, wavelengths in nm, angles in degrees, time dependence exp(-i w t) (absorption: Im eps > 0).",
    "required": ["period_um", "fourier_orders", "wavelengths_nm", "eps_design", "layers"],
    "properties": {
        "period_um": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2,
                      "description": "Periods along x and y. For a 1D grating (fourier_orders [M, 0]) the y period only sets the bookkeeping; any positive value."},
        "fourier_orders": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2,
                           "description": "Orders kept along x and y: (2M + 1)(2N + 1) orders, layer matrices of twice that size. [M, 0] is a 1D grating. Check convergence by raising them."},
        "wavelengths_nm": {"type": "array", "items": {"type": "number"}},
        "incidence": {"type": "array", "items": {"type": "object", "properties": {
            "theta_deg": {"type": "number", "description": "polar angle from z in the superstrate, 0 <= theta < 90"},
            "phi_deg": {"type": "number", "description": "azimuth from x; phi = 0 puts the plane of incidence in x-z"}}},
            "description": "Plane-wave directions; default normal incidence. Every wavelength is solved for every direction, for p and s together."},
        "eps_in": PERMITTIVITY, "eps_out": PERMITTIVITY,
        "eps_design": PERMITTIVITY | {"description": "Permittivity where the density is 1"},
        "eps_fill": PERMITTIVITY | {"description": "Permittivity where the density is 0 (default 1, air)"},
        "grid": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2,
                 "description": "Samples of each layer's density along x and y; at least 4 x orders + 1. Default 512 along an axis with orders, 1 without."},
        "layers": {"type": "array", "minItems": 1, "items": {"type": "object", "required": ["thickness_um", "pattern"], "properties": {
            "thickness_um": {"type": "number"},
            "pattern": {"type": "object", "description":
                        "Density in [0, 1] mixing eps_fill (0) and eps_design (1) linearly. type: 'uniform' {value}; "
                        "'lamellar' {fill_factor, axis 'x' or 'y', center as a fraction of the period, default 0.5}; "
                        "'disk' {radius_um, center_um [x, y], default the cell centre}; 'rectangle' {size_um [a, b], center_um}; "
                        "'array' {density: nested list [grid x][grid y]}; 'npy' {path: a .npy file of shape grid}."},
            "eps_design": PERMITTIVITY | {"description": "Optional: other materials for this layer. Such a layer gets no gradient."},
            "eps_fill": PERMITTIVITY}}},
        "rayleigh_nudge": {"type": "number", "description": "Every solve runs at the frequency (1 + rayleigh_nudge) / wavelength, which keeps the orders off "
                           "their Rayleigh points (grazing orders). Default 1e-4, the library default: a 633 nm solve is at 632.94 nm. Set 0 to compare "
                           "with an analytic result away from a Rayleigh point."},
        "precision": {"enum": ["double", "single"], "description": "double (complex128, default on the CPU) or single (complex64, default on a GPU)"},
        "device": {"enum": ["auto", "cpu", "cuda"], "description": "auto: CUDA if available with torch >= 2.8, otherwise the CPU"},
    },
}


def example(name: str) -> dict:
    if name not in EXAMPLES:
        raise ValueError(f"Unknown example {name!r}; choose one of {', '.join(EXAMPLES)}.")
    return copy.deepcopy(EXAMPLES[name]["problem"])


def permittivity(value) -> complex:
    if isinstance(value, dict):
        return complex(float(value["n"]), float(value.get("k", 0.0))) ** 2
    if isinstance(value, (list, tuple)):
        return complex(float(value[0]), float(value[1]))
    return complex(value)


def _orders(problem) -> tuple[int, int]:
    o = tuple(int(x) for x in problem["fourier_orders"])
    if len(o) != 2 or min(o) < 0:
        raise ValueError("fourier_orders must be two non-negative integers [M, N]")
    return o


def _grid(problem) -> tuple[int, int]:
    if "grid" in problem:
        return tuple(int(x) for x in problem["grid"])
    return tuple(512 if o > 0 else 1 for o in _orders(problem))


def density(pattern: dict, period_um, grid) -> np.ndarray:
    """Density samples of a pattern; sample [i, j] sits at x = i period_x / H, y = j period_y / W."""
    h, w = grid
    lx, ly = period_um
    x = (np.arange(h) / h * lx)[:, None]
    y = (np.arange(w) / w * ly)[None, :]
    kind = pattern.get("type")

    def wrapped(u, c, period):                            # distance to the centre on the periodic cell
        d = (u - c) % period
        return np.minimum(d, period - d)

    def inside(u, c, width, period):                      # u in [c - width / 2, c + width / 2) on the periodic cell
        return (u - c + width / 2) % period < width

    if kind == "uniform":
        rho = np.full((h, w), float(pattern.get("value", 1.0)))
    elif kind == "lamellar":
        axis = pattern.get("axis", "x")
        f = float(pattern["fill_factor"])
        c = float(pattern.get("center", 0.5))
        u, period = (x, lx) if axis == "x" else (y, ly)
        rho = np.broadcast_to(inside(u, c * period, f * period, period), (h, w)).astype(float)
    elif kind == "disk":
        cx, cy = pattern.get("center_um", (lx / 2, ly / 2))
        rho = ((wrapped(x, cx, lx) ** 2 + wrapped(y, cy, ly) ** 2) < float(pattern["radius_um"]) ** 2).astype(float)
    elif kind == "rectangle":
        a, b = pattern["size_um"]
        cx, cy = pattern.get("center_um", (lx / 2, ly / 2))
        rho = (inside(x, cx, a, lx) & inside(y, cy, b, ly)).astype(float)
    elif kind == "array":
        rho = np.asarray(pattern["density"], dtype=float)
        if rho.ndim == 1:
            rho = rho[:, None]
    elif kind == "npy":
        rho = np.load(Path(pattern["path"]), allow_pickle=False).astype(float)
        if rho.ndim == 1:
            rho = rho[:, None]
    else:
        raise ValueError(f"unknown pattern type {kind!r}: uniform, lamellar, disk, rectangle, array or npy")
    if rho.shape != (h, w):
        raise ValueError(f"pattern density has shape {rho.shape}, the grid is {(h, w)}")
    return np.ascontiguousarray(rho)


def cuda_kernels_available() -> bool:
    """Single precision on CUDA compiles its kernels with torch.cuda._compile_kernel (torch >= 2.8)."""
    major, minor = (int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    return torch.cuda.is_available() and (major, minor) >= (2, 8)


def _device_dtype(problem):
    device = problem.get("device", "auto")
    if device == "auto":
        device = "cuda" if cuda_kernels_available() else "cpu"
    precision = problem.get("precision", "double" if device == "cpu" else "single")
    if precision not in ("double", "single"):
        raise ValueError("precision is 'double' or 'single'")
    return device, (torch.complex128 if precision == "double" else torch.complex64)


def check(problem: dict) -> dict:
    """Errors and warnings of a problem, and its size; raises nothing."""
    errors, warnings = [], []
    try:
        ox, oy = _orders(problem)
        grid = _grid(problem)
        n_orders = (2 * ox + 1) * (2 * oy + 1)
        if any(g < 4 * o + 1 for g, o in zip(grid, (ox, oy))):
            errors.append(f"grid {grid} is too coarse for fourier_orders {(ox, oy)}: need at least {(4 * ox + 1, 4 * oy + 1)}")
        eps = {k: permittivity(problem.get(k, 1.0)) for k in ("eps_in", "eps_out", "eps_design", "eps_fill")}
        for layer in problem["layers"]:
            for k in ("eps_design", "eps_fill"):
                if k in layer:
                    eps[f"layer {k}"] = permittivity(layer[k])
        if eps["eps_in"].imag != 0 or eps["eps_in"].real <= 0:
            errors.append("eps_in (the superstrate) must be real and positive: light comes from a lossless medium")
        for k, v in eps.items():
            if v.imag < 0:
                errors.append(f"{k} = {v} has Im(eps) < 0, which is gain in the exp(-i w t) convention; write an absorber as {{'n': n, 'k': k}} with k > 0")
            if v.real < 0:
                warnings.append(f"{k} = {v} is metallic: the Laurent rule converges slowly in the orders for metals; check convergence_in_orders")
        incid = problem.get("incidence") or [dict(theta_deg=0.0, phi_deg=0.0)]
        for inc in incid:
            if not 0 <= float(inc.get("theta_deg", 0.0)) < 90:
                errors.append(f"theta_deg {inc.get('theta_deg')} must be in [0, 90)")
        if not problem["layers"]:
            errors.append("need at least one layer; a bare interface is a layer of any pattern with thickness 0")
        rhos = []
        for i, layer in enumerate(problem["layers"]):
            if float(layer["thickness_um"]) < 0:
                errors.append(f"layer {i}: negative thickness")
            try:
                rho = density(layer["pattern"], problem["period_um"], grid)
                rhos.append(rho)
                if rho.min() < 0 or rho.max() > 1:
                    warnings.append(f"layer {i}: density outside [0, 1] extrapolates the permittivity linearly")
            except Exception as exc:  # noqa: BLE001 - reported to the caller
                errors.append(f"layer {i}: {exc}")
        size = 2 * n_orders
        solves = len(problem["wavelengths_nm"]) * len(incid)
        if size > 1200 and _device_dtype(problem)[0] == "cpu":
            warnings.append(f"layer matrices of {size} x {size} on the CPU take seconds to minutes per solve; a CUDA GPU with precision 'single' is much faster")
        for wl in problem["wavelengths_nm"]:
            for inc in incid:
                gap = _rayleigh_gap(problem, wl, inc)
                if gap < 1e-3:
                    warnings.append(f"{wl} nm at theta {inc.get('theta_deg', 0)}: an order is within {gap:.1e} of grazing (Rayleigh anomaly); "
                                    "efficiencies change steeply there")
        return dict(ok=not errors, errors=errors, warnings=warnings, orders=n_orders, layer_matrix_size=size, solves=solves,
                    grid=list(grid), device=_device_dtype(problem)[0], precision="double" if _device_dtype(problem)[1] == torch.complex128 else "single")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"malformed problem: {exc!r}; see problem_schema")
        return dict(ok=False, errors=errors, warnings=warnings)


def _wave_vector(problem, wl_nm, inc, m, n):
    n_in = permittivity(problem.get("eps_in", 1.0)).real ** 0.5
    th, ph = math.radians(float(inc.get("theta_deg", 0.0))), math.radians(float(inc.get("phi_deg", 0.0)))
    lx, ly = problem["period_um"]
    wl = wl_nm / 1000.0
    return n_in * math.sin(th) * math.cos(ph) + m * wl / lx, n_in * math.sin(th) * math.sin(ph) + n * wl / ly


def _rayleigh_gap(problem, wl_nm, inc):
    ox, oy = _orders(problem)
    gap = math.inf
    for medium in ("eps_in", "eps_out"):
        nr = permittivity(problem.get(medium, 1.0)).real ** 0.5
        for m in range(-ox, ox + 1):
            for n in range(-oy, oy + 1):
                kx, ky = _wave_vector(problem, wl_nm, inc, m, n)
                gap = min(gap, abs(math.hypot(kx, ky) - nr) / nr)
    return gap


def _forward(problem, eps_design=None, eps_fill=None, orders=None):
    device, dtype = _device_dtype(problem)
    orders = tuple(orders) if orders is not None else _orders(problem)
    grid = _grid(problem)
    n_in = permittivity(problem.get("eps_in", 1.0)).real ** 0.5
    stack = Stack(period_um=tuple(float(p) for p in problem["period_um"]), thickness_um=float(problem["layers"][0]["thickness_um"]),
                  density_shape=grid, fourier_orders=orders,
                  eps_design=permittivity(problem["eps_design"] if eps_design is None else eps_design),
                  eps_fill=permittivity(problem.get("eps_fill", 1.0) if eps_fill is None else eps_fill),
                  eps_in=permittivity(problem.get("eps_in", 1.0)), eps_out=permittivity(problem.get("eps_out", 1.0)), detector_samples=(8, 8),
                  rayleigh_nudge=float(problem.get("rayleigh_nudge", Stack.rayleigh_nudge)))
    incid = problem.get("incidence") or [dict(theta_deg=0.0, phi_deg=0.0)]
    rays = tuple(Ray(math.radians(float(i.get("theta_deg", 0.0))), math.radians(float(i.get("phi_deg", 0.0))), 1.0, n_in) for i in incid)
    return ColorRouterForward(stack=stack, rays=rays, wavelengths_nm=[float(w) for w in problem["wavelengths_nm"]],
                              device=device, dtype=dtype, config=DEFAULT)


def _layers(problem, fwd, densities):
    """stack_field layers: (density, thickness) for layers of the problem's materials, stored blocks for the others."""
    out = []
    for layer, rho in zip(problem["layers"], densities):
        if "eps_design" in layer or "eps_fill" in layer:
            other = _forward(problem, layer.get("eps_design", problem["eps_design"]), layer.get("eps_fill", problem.get("eps_fill", 1.0)),
                             orders=fwd.orders)
            with torch.no_grad():
                out.append(other.layer_block(rho.detach(), float(layer["thickness_um"])))
        else:
            out.append((rho, float(layer["thickness_um"])))
    return out


def _efficiencies_ps(fwd, layers):
    """(T, R) [solves, orders, 2] for p and s incidence (columns), from the x-like and y-like columns of the solver."""
    g = fwd._solves(0, None)
    below, top = fwd.stack_field(layers, return_top=True)
    phi = torch.tensor([s[4] for s in fwd.solve_spec], dtype=fwd.rdt, device=fwd.device)
    c, s = torch.cos(phi)[:, None, None].to(below.dtype), torch.sin(phi)[:, None, None].to(below.dtype)

    def ps(a):                                       # columns p cos(phi) - s sin(phi), p sin(phi) + s cos(phi) -> p, s
        return torch.cat((c * a[..., :1] + s * a[..., 1:], -s * a[..., :1] + c * a[..., 1:]), -1)

    n = fwd.n_orders

    def flux(a, y):
        h = y.left(a)
        return (a[:, :n] * h[:, n:].conj() - a[:, n:] * h[:, :n].conj()).real

    e_in = ps(g.e_in)
    incident = flux(e_in, g.vi).sum(1, keepdim=True)
    return flux(ps(below), g.vo) / incident, flux(ps(top) - e_in, g.vi) / incident


def _tensors(problem, fwd, requires_grad=False):
    rdt = fwd.rdt
    out = []
    for layer in problem["layers"]:
        rho = torch.tensor(density(layer["pattern"], problem["period_um"], fwd.shape), dtype=rdt, device=fwd.device)
        out.append(rho.requires_grad_(requires_grad))
    return out


def order_list(orders):
    ox, oy = orders
    return [(m, n) for m in range(-ox, ox + 1) for n in range(-oy, oy + 1)]


def _raise_if_invalid(problem):
    report = check(problem)
    if not report["ok"]:
        raise ValueError("; ".join(report["errors"]))
    return report


def solve(problem: dict, threshold: float = 1e-6) -> dict:
    """Efficiencies of every propagating diffraction order for p and s incidence, per wavelength and direction.

    Returns a dict: ``results`` (one entry per wavelength and direction, orders with an efficiency above ``threshold``),
    ``arrays`` (numpy T and R [wavelength, direction, order, (p, s)], and the order list) and the report of :func:`check`."""
    report = _raise_if_invalid(problem)
    fwd = _forward(problem)
    with torch.no_grad():
        rhos = _tensors(problem, fwd)
        t, r = _efficiencies_ps(fwd, _layers(problem, fwd, rhos))
    t, r = t.double().cpu().numpy(), r.double().cpu().numpy()
    n_wl, n_dir = len(fwd.wavelengths_nm), len(fwd.rays)
    t = t.reshape(n_wl, n_dir, fwd.n_orders, 2)
    r = r.reshape(n_wl, n_dir, fwd.n_orders, 2)
    orders = order_list(fwd.orders)
    incid = problem.get("incidence") or [dict(theta_deg=0.0, phi_deg=0.0)]
    lossless = all(permittivity(v).imag == 0 for v in
                   [problem.get("eps_out", 1.0), problem["eps_design"], problem.get("eps_fill", 1.0)]
                   + [layer[k] for layer in problem["layers"] for k in ("eps_design", "eps_fill") if k in layer])
    results = []
    for li, wl in enumerate(fwd.wavelengths_nm):
        for di, inc in enumerate(incid):
            entry = dict(wavelength_nm=wl, solved_at_nm=wl / (1 + fwd.stack.rayleigh_nudge), theta_deg=float(inc.get("theta_deg", 0.0)), phi_deg=float(inc.get("phi_deg", 0.0)))
            for side, arr, medium in (("transmitted", t, "eps_out"), ("reflected", r, "eps_in")):
                rows = []
                nr = permittivity(problem.get(medium, 1.0))
                for k, (m, n) in enumerate(orders):
                    p, s = arr[li, di, k]
                    if max(abs(p), abs(s)) <= threshold:
                        continue
                    kx, ky = _wave_vector(problem, wl, inc, m, n)
                    row = dict(order=[m, n], p=float(p), s=float(s), unpolarised=float((p + s) / 2))
                    if nr.imag == 0 and math.hypot(kx, ky) < nr.real ** 0.5:
                        row["angle_deg"] = math.degrees(math.asin(math.hypot(kx, ky) / nr.real ** 0.5))
                        row["direction_phi_deg"] = math.degrees(math.atan2(ky, kx))
                    rows.append(row)
                entry[side] = rows
            tt, rr = t[li, di].sum(0), r[li, di].sum(0)
            entry["total"] = dict(T=dict(p=float(tt[0]), s=float(tt[1])), R=dict(p=float(rr[0]), s=float(rr[1])),
                                  absorbed=dict(p=float(1 - tt[0] - rr[0]), s=float(1 - tt[1] - rr[1])))
            results.append(entry)
    energy_error = float(np.abs(t.sum(2) + r.sum(2) - 1).max())
    checks = dict(energy_error=energy_error if lossless else None,
                  note="max |T + R - 1| over all solves; lossless stacks should give about 1e-12 in double and 1e-5 in single precision"
                  if lossless else "an absorbing stack: 1 - T - R is the absorbed power")
    return dict(results=results, checks=checks, report=report,
                arrays=dict(T=t, R=r, orders=np.asarray(orders), wavelengths_nm=np.asarray(fwd.wavelengths_nm)))


def gradient(problem: dict, order=(0, 0), side: str = "T", polarisation: str = "unpolarised", fd_check: bool = False) -> dict:
    """Mean over wavelengths and directions of the efficiency of one order, and its gradient with respect to the density of every layer.

    Layers with their own eps_design / eps_fill get no gradient (None).  With ``fd_check`` a central difference along the gradient
    direction is compared with the gradient's norm (use precision 'double' for that)."""
    _raise_if_invalid(problem)
    if side not in ("T", "R") or polarisation not in ("p", "s", "unpolarised"):
        raise ValueError("side is 'T' or 'R'; polarisation is 'p', 's' or 'unpolarised'")
    fwd = _forward(problem)
    orders = order_list(fwd.orders)
    if tuple(order) not in orders:
        raise ValueError(f"order {tuple(order)} is outside the orders kept, {fwd.orders}")
    k = orders.index(tuple(order))
    cols = {"p": [0], "s": [1], "unpolarised": [0, 1]}[polarisation]

    def objective(rhos):
        t, r = _efficiencies_ps(fwd, _layers(problem, fwd, rhos))
        return (t if side == "T" else r)[:, k, cols].mean()

    rhos = _tensors(problem, fwd, requires_grad=True)
    value = objective(rhos)
    grads = torch.autograd.grad(value, rhos, allow_unused=True)
    free = ["eps_design" not in layer and "eps_fill" not in layer for layer in problem["layers"]]
    grads = [g.double().cpu().numpy() if (g is not None and f) else None for g, f in zip(grads, free)]
    out = dict(value=float(value), order=list(order), side=side, polarisation=polarisation, gradients=grads)
    if fd_check:
        direction = [torch.tensor(g if g is not None else np.zeros(fwd.shape), dtype=fwd.rdt, device=fwd.device) for g in grads]
        norm = float(sum((d.double() ** 2).sum() for d in direction)) ** 0.5
        if norm > 0:
            h = 1e-5 if fwd.cdt == torch.complex128 else 1e-2
            with torch.no_grad():
                base = [x.detach() for x in rhos]
                plus = float(objective([b + h * d / norm for b, d in zip(base, direction)]))
                minus = float(objective([b - h * d / norm for b, d in zip(base, direction)]))
            out["fd_check"] = dict(central_difference=(plus - minus) / (2 * h), gradient_norm=norm, step=h,
                                   note="the two agree when the gradient is right")
    return out


def convergence(problem: dict, orders_list=None) -> dict:
    """Solve at increasing orders and report how much the efficiencies of the low orders change."""
    ox, oy = _orders(problem)
    if orders_list is None:
        orders_list = [[ox, oy], [round(ox * 1.5) + (ox > 0), round(oy * 1.5) + (oy > 0)], [2 * ox, 2 * oy]]
    rows, previous = [], None
    for o in orders_list:
        p = dict(problem, fourier_orders=[int(x) for x in o])
        if "grid" in problem:
            p["grid"] = [max(g, 4 * x + 1) if x > 0 else g for g, x in zip(problem["grid"], o)]
        sol = solve(p)
        a = sol["arrays"]
        keep = {(int(m), int(n)): i for i, (m, n) in enumerate(a["orders"]) if abs(m) <= ox and abs(n) <= oy}
        cur = {key: (a["T"][:, :, i], a["R"][:, :, i]) for key, i in keep.items()}
        row = dict(fourier_orders=list(o), energy_error=sol["checks"]["energy_error"])
        if previous is not None:
            row["max_change_from_previous"] = float(max(max(np.abs(cur[key][0] - previous[key][0]).max(), np.abs(cur[key][1] - previous[key][1]).max())
                                                        for key in cur))
        rows.append(row)
        previous = cur
    return dict(rows=rows, note="largest change of any efficiency of the orders up to the first truncation; report the result with the most orders "
                                "and this change as its truncation error")
