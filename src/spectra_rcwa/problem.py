"""Grating problems described as plain dictionaries (JSON), for the command line and the MCP server.

A problem names the lattice, the Fourier orders, the wavelengths, the incidence directions, the media above and below,
and the layers, top first, each a pattern of two materials.  :func:`solve` returns the power in every diffraction order
for p and s incidence, :func:`gradient` the derivative of one of them with respect to the density of every layer.
Lengths are in um, wavelengths in nm, angles in degrees; the time dependence is exp(-i w t), so absorption is Im(eps) > 0.
"""
from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import numpy as np
import torch

from . import lapack
from .config import DEFAULT
from .forward import ColorRouterForward, Ray, Stack
from .geometry import reciprocal, truncated_orders
from .normals import in_plane

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
    "hexagonal_pillars": dict(
        description="Hexagonal array of index 2.0 disks (radius 0.15 um, height 0.6 um, pitch 0.5 um) on glass, 450 to 650 nm at normal incidence",
        problem=dict(period_um=[0.5, 0.5], lattice_angle_deg=60.0, fourier_orders=[5, 5], wavelengths_nm=[450.0, 550.0, 650.0],
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
                      "description": "Periods along x and y: the lengths of the lattice vectors. For a 1D grating (fourier_orders [M, 0]) the y period only sets the bookkeeping; any positive value."},
        "lattice_angle_deg": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 180,
                              "description": "Angle between the lattice vectors a1 = (period_x, 0) and a2 = period_y (cos, sin)(angle); default 90. "
                              "60 or 120 with equal periods is hexagonal. An oblique lattice needs orders along both vectors. Patterns: center_um and size_um stay "
                              "Cartesian x, y (default center: the middle of the cell, (a1 + a2) / 2); lamellar stripes and arrays follow the lattice vectors."},
        "truncation": {"enum": ["auto", "rectangular", "circular"],
                       "description": "Orders kept: rectangular |m| <= M, |n| <= N; circular: the ellipse through (M, 0) and (0, N) with the symmetry of the lattice. "
                       "Default auto: rectangular at 90 degrees, circular otherwise (a hexagonal pattern keeps its symmetry only in the circular one)."},
        "factorization": {"enum": ["auto", "laurent", "li", "normal"],
                          "description": "How the in-plane permittivity is truncated. auto (default): Li's rule for a 1D grating, Laurent's for a crossed one. "
                          "normal: for crossed gratings of high index or metal, far fewer orders for the same error. li: 90 degrees and rectangular orders only."},
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
                 "description": "Samples of each layer's density along x and y; at least 4 x orders + 1. Default 512 (more if the orders need it) along an axis "
                                "with orders, 1 without. solve uses exact Fourier coefficients, not samples, for lamellar layers and, in a crossed grating, "
                                "for rectangles and disks."},
        "layers": {"type": "array", "minItems": 1, "items": {"type": "object", "required": ["thickness_um", "pattern"], "properties": {
            "thickness_um": {"type": "number"},
            "pattern": {"type": "object", "description":
                        "Density in [0, 1] mixing eps_fill (0) and eps_design (1) linearly. type: 'uniform' {value}; "
                        "'lamellar' {fill_factor, axis 'x' or 'y', center as a fraction of the period, default 0.5}; "
                        "'disk' {radius_um, center_um [x, y], default the cell centre}; 'rectangle' {size_um [a, b], center_um}; "
                        "'array' {density: nested list [grid x][grid y]}; 'npy' {path: a .npy file of shape grid}. An array is point samples at "
                        "(i period_x / H, j period_y / W); with pixels: true it is constant over rectangular pixels centred there, of any shape "
                        "(not tied to grid), and its Fourier coefficients are exact."},
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
    return tuple(max(512, 4 * o + 1) if o > 0 else 1 for o in _extent(problem))


def _angle(problem) -> float:
    return float(problem.get("lattice_angle_deg", 90.0))


def _order_list(problem, orders=None):
    return truncated_orders(tuple(_orders(problem) if orders is None else (int(o) for o in orders)), _angle(problem), problem.get("truncation", "auto"))


def _extent(problem, orders=None) -> tuple[int, int]:
    """Largest |m| and |n| of the orders kept."""
    kept = _order_list(problem, orders)
    return max(m for m, _ in kept), max(n for _, n in kept)


def _shortest(period_um, angle: float) -> float:
    """Distance between a point and its nearest image on the lattice."""
    reach = 1 + int(max(period_um) / (min(period_um) * math.sin(math.radians(angle))))
    return min(math.hypot(x, y) for x, y in _translations(period_um, angle, reach) if (x, y) != (0.0, 0.0))


def _translations(period_um, angle: float, reach: int):
    lx, ly = (float(p) for p in period_um)
    c, s = (0.0, 1.0) if angle == 90 else (math.cos(math.radians(angle)), math.sin(math.radians(angle)))
    return [(i * lx + j * ly * c, j * ly * s) for i in range(-reach, reach + 1) for j in range(-reach, reach + 1)]


def _array(pattern: dict) -> np.ndarray:
    rho = np.asarray(pattern["density"], dtype=float) if pattern["type"] == "array" else np.load(Path(pattern["path"]), allow_pickle=False).astype(float)
    return rho[:, None] if rho.ndim == 1 else rho


def _pixels(pattern: dict) -> bool:
    return pattern.get("type") in ("array", "npy") and bool(pattern.get("pixels"))


def _center(pattern: dict, period_um, angle: float) -> tuple[float, float]:
    """The centre of a disk or rectangle, by default the middle of the cell."""
    lx, ly = (float(p) for p in period_um)
    c, s = (0.0, 1.0) if angle == 90 else (math.cos(math.radians(angle)), math.sin(math.radians(angle)))
    x, y = pattern.get("center_um", ((lx + ly * c) / 2, ly * s / 2))
    return float(x), float(y)


def density(pattern: dict, period_um, grid, angle: float = 90.0) -> np.ndarray:
    """Density samples of a pattern; sample [i, j] sits at (i / H) a1 + (j / W) a2, which is x = i period_x / H, y = j period_y / W at 90 degrees.
    An array with ``pixels`` is returned as it is."""
    h, w = grid
    lx, ly = period_um
    if angle != 90 and pattern.get("type") in ("disk", "rectangle"):
        cx, cy = _center(pattern, period_um, angle)
        c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        u, v = np.arange(h)[:, None] / h * lx, np.arange(w)[None, :] / w * ly
        px, py = u + v * c - cx, v * s - cy
        rho = np.zeros((h, w), dtype=bool)
        size = 2 * float(pattern["radius_um"]) if pattern["type"] == "disk" else max(pattern["size_um"])
        height = min(lx, ly) * s                           # the images that can reach into the cell
        reach = 2 + int((size + math.hypot(cx, cy)) / height)
        for tx, ty in _translations(period_um, angle, reach):
            if pattern["type"] == "disk":
                rho |= (px - tx) ** 2 + (py - ty) ** 2 < float(pattern["radius_um"]) ** 2
            else:
                a, b = (float(q) for q in pattern["size_um"])
                rho |= (px - tx >= -a / 2) & (px - tx < a / 2) & (py - ty >= -b / 2) & (py - ty < b / 2)
        return rho.astype(float)
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
    elif kind in ("array", "npy"):
        rho = _array(pattern)
        if pattern.get("pixels"):
            return np.ascontiguousarray(rho)
    else:
        raise ValueError(f"unknown pattern type {kind!r}: uniform, lamellar, disk, rectangle, array or npy")
    if rho.shape != (h, w):
        raise ValueError(f"pattern density has shape {rho.shape}, the grid is {(h, w)}")
    return np.ascontiguousarray(rho)


def _exact(pattern: dict, period_um, orders, angle: float = 90.0, rule: str = "auto") -> bool:
    """Whether the Fourier coefficients of a pattern are used exactly instead of from samples.

    Any lamellar pattern; a rectangle or a disk of a crossed grating that does not overlap its images (a rectangle may at 90 degrees).  With orders along one axis only,
    Li's rule needs the pattern to vary along one axis."""
    kind = pattern.get("type")
    if rule == "li" and min(orders) > 0:                   # Li's rules of a crossed grating work on samples
        return False
    if kind == "disk" and 2 * float(pattern["radius_um"]) > _shortest(period_um, angle) * (1 + 1e-9):
        return False
    if kind == "rectangle" and angle != 90:
        a, b = (float(q) for q in pattern["size_um"])
        reach = 2 + int(math.hypot(a, b) / (min(period_um) * math.sin(math.radians(angle))))
        if any(abs(x) < a * (1 - 1e-12) and abs(y) < b * (1 - 1e-12) for x, y in _translations(period_um, angle, reach) if (x, y) != (0.0, 0.0)):
            return False
    return kind == "lamellar" or _pixels(pattern) or (kind in ("rectangle", "disk") and min(orders) > 0)


def coefficients(pattern: dict, period_um, m, n, angle: float = 90.0) -> torch.Tensor:
    """Fourier coefficients (1 / cell) int rho exp(-2 pi i (m b1 + n b2) . r) of a lamellar, rectangle or disk pattern, b1 = (1 / period_x, 0) and
    b2 = (0, 1 / period_y) at 90 degrees."""
    lx, ly = (float(p) for p in period_um)
    (b1x, b1y), (_, b2y) = reciprocal((lx, ly), angle)
    m, n = torch.as_tensor(m, dtype=torch.float64), torch.as_tensor(n, dtype=torch.float64)

    def bar(k, fill, center):
        fill = min(max(fill, 0.0), 1.0)
        return fill * torch.sinc(k * fill) * torch.exp(-2j * math.pi * k * center)

    kind = pattern["type"]
    if kind == "lamellar":
        k, other = (m, n) if pattern.get("axis", "x") == "x" else (n, m)
        return bar(k, float(pattern["fill_factor"]), float(pattern.get("center", 0.5))) * (other == 0)
    cx, cy = _center(pattern, period_um, angle)
    gx, gy = m * b1x, m * b1y + n * b2y
    area = 1 / (b1x * b2y)
    if kind == "rectangle":
        a, b = (float(q) for q in pattern["size_um"])
        if angle == 90:
            return bar(m, a / lx, cx / lx) * bar(n, b / ly, cy / ly)
        return a * b / area * torch.sinc(gx * a) * torch.sinc(gy * b) * torch.exp(-2j * math.pi * (gx * cx + gy * cy))
    if kind == "disk":
        radius = float(pattern["radius_um"])
        u = 2 * math.pi * radius * torch.hypot(gx, gy)
        shape = torch.where(u == 0, 0.5, torch.special.bessel_j1(u) / torch.where(u == 0, 1.0, u))        # J1(u) / u
        return 2 * math.pi * radius ** 2 / area * shape * torch.exp(-2j * math.pi * (gx * cx + gy * cy))
    raise ValueError(f"no closed form for pattern type {kind!r}")


def _pixel_coefficients(values: torch.Tensor, m, n) -> torch.Tensor:
    """Fourier coefficients [..., orders] of values [..., H, W] constant over pixels (parallelograms along a1 and a2) centred on the samples."""
    h, w = values.shape[-2:]
    spec = torch.fft.fft2(values) / (h * w)
    return spec[..., m % h, n % w] * torch.sinc(m.double() / h) * torch.sinc(n.double() / w)


def _matrices(fwd, pattern: dict, period_um, rho: torch.Tensor):
    """Permittivity matrices (E [L, 2, N, N], E^-1 [L, N, N]) of a layer from exact Fourier coefficients."""
    ox, oy = (max(o[i] for o in fwd.order_list) for i in (0, 1))
    m, n = torch.meshgrid(torch.arange(-2 * ox, 2 * ox + 1), torch.arange(-2 * oy, 2 * oy + 1), indexing="ij")
    orders = torch.tensor(fwd.order_list)
    dm, dn = (orders[:, None] - orders[None, :]).unbind(-1)
    row, col = (dm + 2 * ox).to(fwd.device), (dn + 2 * oy).to(fwd.device)
    lo, hi = fwd.eps_bg.to(torch.complex128)[:, None, None], fwd.eps_design.to(torch.complex128)[:, None, None]
    if _pixels(pattern):
        m, n = m.to(fwd.device), n.to(fwd.device)
        values = lo + rho.to(torch.complex128) * (hi - lo)
        eps, inv = _pixel_coefficients(values, m, n), _pixel_coefficients(1 / values, m, n)
        along_y = rho.shape[0] == 1
    else:
        one = ((m == 0) & (n == 0)).to(torch.complex128).to(fwd.device)
        c = coefficients(pattern, period_um, m, n, fwd.stack.lattice_angle_deg).to(fwd.device)
        eps, inv = lo * one + (hi - lo) * c, one / lo + (1 / hi - 1 / lo) * c
        along_y = pattern.get("axis") == "y"
    laurent = eps[:, row, col]
    rule = fwd.config.factorization
    if rule == "normal" and min(fwd.orders) > 0:
        samples = rho
        if _pixels(pattern):                               # the pixel that each sample of the grid is in
            i, j = (((torch.arange(g, device=fwd.device) * s * 2 + g) // (2 * g)) % s for g, s in zip(fwd.shape, rho.shape))
            samples = rho[i][:, j]
        tensor = fwd._tensor(samples.real[None]).to(torch.complex128)
        e = in_plane(laurent, lapack.inv(inv[:, row, col]), tensor)
    elif rule == "laurent" or min(fwd.orders) > 0:
        e = laurent[:, None].expand(-1, 2, -1, -1)
    else:                                                  # Li: the inverse rule for the component across the edges
        across = lapack.inv(inv[:, row, col])
        e = torch.stack((laurent, across) if along_y else (across, laurent), 1)
    return e.to(fwd.cdt), lapack.inv(laurent.to(fwd.cdt))


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
        angle = _angle(problem)
        if not 0 < angle < 180:
            raise ValueError("lattice_angle_deg must be between 0 and 180")
        if problem.get("truncation", "auto") not in ("auto", "rectangular", "circular"):
            raise ValueError("truncation is 'auto', 'rectangular' or 'circular'")
        rule = problem.get("factorization", "auto")
        if rule not in ("auto", "laurent", "li", "normal"):
            raise ValueError("factorization is 'auto', 'laurent', 'li' or 'normal'")
        if rule == "li" and (angle != 90 or _order_list(problem) != truncated_orders(_orders(problem))):
            raise ValueError("factorization 'li' needs lattice_angle_deg 90 and the rectangular truncation")
        if angle != 90 and min(ox, oy) == 0:
            raise ValueError("an oblique lattice needs orders along both lattice vectors; a 1D grating has lattice_angle_deg 90")
        if not 60 - 1e-9 <= angle <= 120 + 1e-9:
            warnings.append(f"lattice_angle_deg {angle}: the same lattice has shorter vectors between 60 and 120 degrees, which need fewer orders")
        grid = _grid(problem)
        n_orders = len(_order_list(problem))
        ex, ey = _extent(problem)
        if any(g < 4 * o + 1 for g, o in zip(grid, (ex, ey))):
            errors.append(f"grid {grid} is too coarse for fourier_orders {(ox, oy)}: need at least {(4 * ex + 1, 4 * ey + 1)}")
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
                warnings.append(f"{k} = {v} is metallic: metals converge slowly in the orders, above all in a crossed grating; check convergence_in_orders")
        incid = problem.get("incidence") or [dict(theta_deg=0.0, phi_deg=0.0)]
        for inc in incid:
            if not 0 <= float(inc.get("theta_deg", 0.0)) < 90:
                errors.append(f"theta_deg {inc.get('theta_deg')} must be in [0, 90)")
        if not problem["layers"]:
            errors.append("need at least one layer; a bare interface is a layer of any pattern with thickness 0")
        for i, layer in enumerate(problem["layers"]):
            if float(layer["thickness_um"]) < 0:
                errors.append(f"layer {i}: negative thickness")
            try:
                rho = density(layer["pattern"], problem["period_um"], grid, angle)
                if _pixels(layer["pattern"]) and min(ox, oy) == 0 and min(rho.shape) > 1:
                    errors.append(f"layer {i}: a pixel array that varies along both axes needs orders along both")
                if layer["pattern"].get("type") in ("rectangle", "disk") and min(ox, oy) == 0:
                    warnings.append(f"layer {i}: with orders along one axis a {layer['pattern']['type']} is sampled on the grid {grid}, not exact; "
                                    "with one sample along an axis only the line at 0 is seen. Use a lamellar pattern or orders along both axes")
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
    (b1x, b1y), (_, b2y) = reciprocal(problem["period_um"], _angle(problem))
    wl = wl_nm / 1000.0
    return n_in * math.sin(th) * math.cos(ph) + m * wl * b1x, n_in * math.sin(th) * math.sin(ph) + wl * (m * b1y + n * b2y)


def _rayleigh_gap(problem, wl_nm, inc):
    gap = math.inf
    for medium in ("eps_in", "eps_out"):
        nr = permittivity(problem.get(medium, 1.0)).real ** 0.5
        for m, n in _order_list(problem):
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
                  rayleigh_nudge=float(problem.get("rayleigh_nudge", Stack.rayleigh_nudge)),
                  lattice_angle_deg=_angle(problem), truncation=problem.get("truncation", "auto"))
    incid = problem.get("incidence") or [dict(theta_deg=0.0, phi_deg=0.0)]
    rays = tuple(Ray(math.radians(float(i.get("theta_deg", 0.0))), math.radians(float(i.get("phi_deg", 0.0))), 1.0, n_in) for i in incid)
    return ColorRouterForward(stack=stack, rays=rays, wavelengths_nm=[float(w) for w in problem["wavelengths_nm"]],
                              device=device, dtype=dtype, config=DEFAULT.replace(factorization=problem.get("factorization", "auto")))


def _layers(problem, fwd, densities, exact=True):
    """stack_field layers: (density or permittivity matrices, thickness) for layers of the problem's materials, stored blocks for the others.
    Without ``exact`` the shapes are sampled, so that the gradient with respect to the samples exists; pixel arrays never are."""
    out = []

    def content(f, pattern, rho):
        if _pixels(pattern) or (exact and _exact(pattern, problem["period_um"], f.orders, _angle(problem), f.config.factorization)):
            return _matrices(f, pattern, problem["period_um"], rho)
        return rho

    for layer, rho in zip(problem["layers"], densities):
        if "eps_design" in layer or "eps_fill" in layer:
            other = _forward(problem, layer.get("eps_design", problem["eps_design"]), layer.get("eps_fill", problem.get("eps_fill", 1.0)),
                             orders=fwd.orders)
            with torch.no_grad():
                out.append(other.layer_block(content(other, layer["pattern"], rho.detach()), float(layer["thickness_um"])))
        else:
            out.append((content(fwd, layer["pattern"], rho), float(layer["thickness_um"])))
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
        rho = torch.tensor(density(layer["pattern"], problem["period_um"], fwd.shape, _angle(problem)), dtype=rdt, device=fwd.device)
        out.append(rho.requires_grad_(requires_grad))
    return out


def order_list(orders, lattice_angle_deg: float = 90.0, truncation: str = "auto"):
    return truncated_orders(tuple(orders), lattice_angle_deg, truncation)


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
    orders = fwd.order_list
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

    The density samples on ``grid`` are the variables, so shapes are sampled here and the value differs from :func:`solve` by the sampling error;
    the variables of an array with ``pixels`` are its pixels, and the value is that of :func:`solve`.
    Layers with their own eps_design / eps_fill get no gradient (None).  With ``fd_check`` a central difference along the gradient
    direction is compared with the gradient's norm (use precision 'double' for that)."""
    _raise_if_invalid(problem)
    if side not in ("T", "R") or polarisation not in ("p", "s", "unpolarised"):
        raise ValueError("side is 'T' or 'R'; polarisation is 'p', 's' or 'unpolarised'")
    fwd = _forward(problem)
    orders = fwd.order_list
    if tuple(order) not in orders:
        raise ValueError(f"order {tuple(order)} is outside the orders kept, {fwd.orders}")
    k = orders.index(tuple(order))
    cols = {"p": [0], "s": [1], "unpolarised": [0, 1]}[polarisation]

    def objective(rhos):
        t, r = _efficiencies_ps(fwd, _layers(problem, fwd, rhos, exact=False))
        return (t if side == "T" else r)[:, k, cols].mean()

    rhos = _tensors(problem, fwd, requires_grad=True)
    value = objective(rhos)
    grads = torch.autograd.grad(value, rhos, allow_unused=True)
    free = ["eps_design" not in layer and "eps_fill" not in layer for layer in problem["layers"]]
    grads = [g.double().cpu().numpy() if (g is not None and f) else None for g, f in zip(grads, free)]
    out = dict(value=float(value), order=list(order), side=side, polarisation=polarisation, gradients=grads)
    if fd_check:
        direction = [torch.tensor(g if g is not None else np.zeros(r.shape), dtype=fwd.rdt, device=fwd.device) for g, r in zip(grads, rhos)]
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


def _embed(sol, size):
    """T and R [wavelength, direction, order, (p, s)] of a solution on the orders |m| <= size[0], |n| <= size[1]; those left out carry no power."""
    m, n = sol["arrays"]["orders"].T
    both = np.stack((sol["arrays"]["T"], sol["arrays"]["R"]))
    out = np.zeros((*both.shape[:3], 2 * size[0] + 1, 2 * size[1] + 1, 2))
    out[:, :, :, m + size[0], n + size[1]] = both
    return out


def _estimate(changes):
    """Error left after the last of the changes, as a geometric series with the ratio of the last two, and at least the last change."""
    last, before = changes[-1], changes[-2]
    q = min(last / before, 0.9) if before > 0 else 0.0
    return last * max(1.0, q / (1 - q))


def _growing_orders(problem, max_matrix_size):
    """The problem's orders and then 1.5x at each step (at least one more along an axis with orders), as far as the matrix size allows."""
    out, o = [], list(_orders(problem))
    while not out or 2 * len(_order_list(problem, o)) <= max_matrix_size:
        out.append(o)
        if max(o) == 0:
            break
        o = [(3 * x + 1) // 2 for x in o]
    return out


def convergence(problem: dict, orders_list=None, tolerance: float | None = None, max_matrix_size: int = 2000, max_seconds: float = 300.0) -> dict:
    """Solve at increasing orders, all on one grid, and report how much the efficiencies change.

    Without ``tolerance`` the orders are ``orders_list``, or the problem's, 1.5x and 2x, at least one more per step along an axis with orders.
    With ``tolerance`` the orders grow by 1.5x from the problem's until the last two changes and the estimated error are all at most
    ``tolerance``; ``converged`` is False, with a ``reason``, when the layer matrix would exceed ``max_matrix_size``, the next solve is expected
    to end after ``max_seconds``, the GPU memory runs out, or a sampled array has no room for more orders.  The change is the largest over all
    efficiencies, an order left out of a truncation counting as 0.  The grid is the problem's, refined to 4 x orders + 1 of the largest orders."""
    reason = "max_matrix_size reached"
    if tolerance is not None:
        if orders_list is not None:
            raise ValueError("give orders_list or tolerance, not both")
        if not tolerance > 0:
            raise ValueError("tolerance must be positive")
        orders_list = _growing_orders(problem, max_matrix_size)
        if max(orders_list[0]) == 0:
            reason = "no orders to refine"
        if any(la["pattern"].get("type") in ("array", "npy") and not _pixels(la["pattern"]) for la in problem["layers"]):
            fits = [o for o in orders_list if all(g >= 4 * x + 1 for g, x in zip(_grid(problem), _extent(problem, o)))] or orders_list[:1]
            if len(fits) < len(orders_list):
                orders_list, reason = fits, "a sampled array has no room for more orders (4 x orders + 1 samples); give more samples, or pixels: true"
    elif orders_list is None:
        o = _orders(problem)
        orders_list = [list(o)]
        if max(o) > 0:                                    # without orders there is nothing to refine
            orders_list += [[(3 * x + 1) // 2 for x in o], [max(2 * x, x + 2) if x > 0 else 0 for x in o]]
    largest = [max(int(o[i]) for o in orders_list) for i in (0, 1)]
    extent = [max(_extent(problem, o)[i] for o in orders_list) for i in (0, 1)]
    grid = [max(g, 4 * x + 1) if x > 0 else g for g, x in zip(_grid(dict(problem, fourier_orders=largest)), extent)]
    rows, changes, previous, converged, seconds, start = [], [], None, False, 0.0, time.perf_counter()
    for k, o in enumerate(orders_list):
        if tolerance is not None and k > 0:
            size, before = (len(_order_list(problem, q)) for q in (o, orders_list[k - 1]))
            expected = seconds * (size / before) ** 3 if k > 1 else 0.0           # the first solve includes the start-up of the GPU kernels
            if time.perf_counter() - start + expected > max_seconds:
                reason = "max_seconds reached"
                break
        began = time.perf_counter()
        try:
            sol = solve(dict(problem, fourier_orders=[int(x) for x in o], grid=grid))
        except torch.cuda.OutOfMemoryError:
            if tolerance is None or not rows:
                raise
            reason = "out of GPU memory"
            break
        seconds = time.perf_counter() - began
        row = dict(fourier_orders=list(o), energy_error=sol["checks"]["energy_error"])
        if previous is not None:
            both = [max(int(np.abs(s["arrays"]["orders"][:, i]).max()) for s in (sol, previous)) for i in (0, 1)]
            changes.append(float(np.abs(_embed(sol, both) - _embed(previous, both)).max()))
            row["max_change_from_previous"] = changes[-1]
        rows.append(row)
        previous = sol
        if tolerance is not None and len(changes) > 1 and max(*changes[-2:], _estimate(changes)) <= tolerance:
            converged = True
            break
    out = dict(rows=rows, grid=grid,
               note="largest change of any efficiency between one truncation and the next on the same grid, an order left out counting as 0; an estimate of "
                    "the truncation error, not a bound on it, and it does not include the error of sampled patterns (see grid)")
    if tolerance is not None:
        out.update(converged=converged, tolerance=tolerance, fourier_orders=rows[-1]["fourier_orders"],
                   estimated_error=_estimate(changes) if len(changes) > 1 else None)
        if not converged:
            out["reason"] = reason
    return out
