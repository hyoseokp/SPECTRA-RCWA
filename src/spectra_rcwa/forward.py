"""Batched RCWA response of a single patterned layer under a partially coherent pupil.

For every (wavelength, pupil ray) the layer is solved for both polarisations at once, all solves of a call in
one batch.  The output is the power reaching each quadrant of the detector plane ("wells" R, G2, G1, B of a
2 x 2 colour-router pixel), incoherently averaged over the pupil and the two polarisations: Tabs[well, lambda].

With e' = i w P h and h' = i w Q e (z downward), a step h of a layer needs only entire functions of A = P Q:

    C = cos(w h A^(1/2)),   S = sin(w h A^(1/2)) / A^(1/2),   G = (C - I) / A
    e(0) = C e(h) - i S P h(h),   h(0) = -i Q S e(h) + (I + Q G P) h(h)

(cos(w h (QP)^(1/2)) = I + Q G P).  From the output medium upwards, with h(h) = Y e(h) and K = P Y:

    D = (C - i S K)^-1,   e(h) = D e(0),   Y_top = (Y + Q (G K - i S)) D

There is no eigenvector, no square root, no branch and no P^-1, so nothing is singular at an order at cutoff.  An evanescent
order makes C and S large, so a layer is cut in steps with w h max(k) at most SolverConfig.max_step_phase; a thick layer goes through its
S-matrix, built by doubling one step (:mod:`spectra_rcwa.blocks`), which also can be stored and reused.  The gradient comes from one
adjoint field and needs no graph (:mod:`spectra_rcwa.adjoint`).  The routes through Gamma = (PQ)^(1/2) and E = exp(i w d Gamma)
(:mod:`spectra_rcwa.matfun`, :mod:`spectra_rcwa.boundary`) are the earlier route, kept as an option and as a reference.
"""
from __future__ import annotations

import dataclasses
import math
import warnings
from dataclasses import dataclass
from typing import Callable, Sequence, Union

import torch

from . import adjoint, blocks, lapack, matfun
from .boundary import boundary_direct, transmitted_smatrix
from .config import DEFAULT, SolverConfig
from .geometry import BDD, SolveGeometry, build_geometry, reciprocal, truncated_orders
from .kernels import lincomb as _lincomb
from .kernels.bdd_pair import bdd_pair
from .normals import in_plane, normal_tensor

Permittivity = Union[complex, float, Callable[[float], complex]]

ENERGY_GUARD_EPS = 0.02
ENERGY_GUARD_RETRY_NUDGES = (0.0, 5.0e-4)
WELL_ORDER = ("R", "G2", "G1", "B")


def _take(x: torch.Tensor, *index) -> torch.Tensor:
    """x[:, *index] of a complex tensor, gathered as its real and imaginary parts."""
    return torch.complex(x.real[(slice(None), *index)], x.imag[(slice(None), *index)])


def _eps(value: Permittivity, wl_um: float) -> complex:
    return complex(value(wl_um) if callable(value) else value)


@dataclass(frozen=True)
class Stack:
    """Periodic stack: superstrate / patterned layer / substrate, detector plane in the substrate.

    Lengths are in um.  Permittivities are numbers or callables of the wavelength in um; the time dependence is exp(-i w t), so an absorbing
    material has Im(eps) > 0, and the superstrate is lossless.  The design density rho in [0, 1] mixes ``eps_fill`` (rho = 0) and ``eps_design``
    (rho = 1) linearly.  ``density[i, j]`` is the sample at x = i period_x / H, y = j period_y / W (first index along x), and H, W are at least
    4 * fourier_orders + 1, the harmonics the layer matrices use.  The detector samples (even counts) sit at (k + 0.5) period / samples, and each
    well is one quadrant of the cell.  Every solve runs at the frequency (1 + rayleigh_nudge) / wavelength, which keeps the orders of the two
    half-spaces off their Rayleigh points.

    ``lattice_angle_deg`` is the angle between the lattice vectors a1 = (period_x, 0) and a2 = period_y (cos, sin) of an oblique lattice (60 or 120 with
    equal periods: hexagonal); ``density[i, j]`` is then the sample at (i / H) a1 + (j / W) a2, and order (m, n) adds m b1 + n b2 of the reciprocal
    lattice.  ``truncation`` picks the orders kept ("auto": rectangular at 90 degrees, circular otherwise; see
    :func:`spectra_rcwa.geometry.truncated_orders`).  An oblique lattice has efficiencies, fields and their
    gradients, but no wells and no Li rule.
    """

    period_um: tuple[float, float]
    thickness_um: float
    density_shape: tuple[int, int]
    fourier_orders: tuple[int, int]
    eps_design: Permittivity
    eps_fill: Permittivity = 1.0
    eps_in: Permittivity = 1.0
    eps_out: Permittivity = 1.0
    detector_z_um: float = 1.0
    detector_samples: tuple[int, int] = (24, 24)
    rayleigh_nudge: float = 1.0e-4
    lattice_angle_deg: float = 90.0
    truncation: str = "auto"

    def order_list(self) -> list[tuple[int, int]]:
        """The diffraction orders (m, n) in the order of the solver."""
        return truncated_orders(tuple(int(o) for o in self.fourier_orders), self.lattice_angle_deg, self.truncation)

    def __post_init__(self):
        for name in ("period_um", "density_shape", "fourier_orders", "detector_samples"):
            v = getattr(self, name)
            if not hasattr(v, "__len__") or len(v) != 2:
                raise ValueError(f"{name} must be a pair (x, y), got {v!r}")
        if min(self.period_um) <= 0 or self.thickness_um < 0:
            raise ValueError("period_um must be positive and thickness_um non-negative")
        if any(int(o) != o or o < 0 for o in self.fourier_orders):
            raise ValueError("fourier_orders must be non-negative integers")
        if any(int(s) != s or s < 2 or s % 2 for s in self.detector_samples):
            raise ValueError("detector_samples must be even, so that the four wells get the same number of samples")
        if not 0 < self.lattice_angle_deg < 180:
            raise ValueError("lattice_angle_deg is the angle between the lattice vectors in degrees, 0 < angle < 180")
        if self.truncation not in ("auto", "rectangular", "circular"):
            raise ValueError("truncation is 'auto', 'rectangular' or 'circular'")
        if self.lattice_angle_deg != 90 and min(self.fourier_orders) == 0:
            raise ValueError("an oblique lattice needs orders along both lattice vectors; a 1D grating has lattice_angle_deg 90")
        need = tuple(4 * max(abs(o[i]) for o in self.order_list()) + 1 for i in (0, 1))
        if any(s < x for s, x in zip(self.density_shape, need)):
            raise ValueError(f"density_shape {tuple(self.density_shape)} is too coarse for fourier_orders {tuple(self.fourier_orders)}: "
                             f"need at least {need}")


@dataclass(frozen=True)
class Ray:
    """One incoherent pupil direction in a reference medium of index ``reference_index`` (refracted into the
    superstrate per wavelength), with its intensity weight."""

    theta_rad: float
    phi_rad: float
    weight: float
    reference_index: float = 1.0

    def __post_init__(self):
        if not 0 <= self.theta_rad < math.pi / 2:
            raise ValueError("Ray.theta_rad is the polar angle in radians, 0 <= theta < pi / 2; the direction around z goes in phi_rad")


class ColorRouterForward:
    """Tabs[well, wavelength] (float64) of a density map, differentiable with respect to the density.

    Args:
        stack: the layer stack.
        rays: pupil rays (see :func:`spectra_rcwa.pupil.shifted_disk_pupil`).
        wavelengths_nm: wavelengths.
        device, dtype: torch device (None: CUDA if there is one and it can run ``dtype``, otherwise the CPU; complex64 on CUDA needs torch >= 2.8) and complex dtype (complex64 for speed, complex128 for
            reference values).  One GPU per process: the kernels belong to the current CUDA device.
        config: solver options; defaults to DEFAULT.
        chunk: maximum number of solves per batch (None = all solves of a call in one batch).
    """

    _tf32_warned = False

    def _tf32_backward(self) -> bool:
        return self.config.boundary == "direct"

    def __init__(self, *, stack: Stack, rays: Sequence[Ray], wavelengths_nm: Sequence[float], device=None,
                 dtype: torch.dtype = torch.complex64, config: SolverConfig = DEFAULT, chunk: int | None = None,
                 verbose_guard: bool = False):
        self.stack = stack
        self.config = config
        if device is None:                                             # complex64 on CUDA compiles its kernels with torch.cuda._compile_kernel (torch >= 2.8)
            device = "cuda" if torch.cuda.is_available() and (dtype == torch.complex128 or hasattr(torch.cuda, "_compile_kernel")) else "cpu"
        self.device = torch.device(device)
        if dtype not in (torch.complex64, torch.complex128):
            raise ValueError("dtype must be torch.complex64 or torch.complex128")
        if self.device.type == "cuda" and self.device.index not in (None, torch.cuda.current_device()):
            raise ValueError("the kernels belong to the current CUDA device: select the GPU with CUDA_VISIBLE_DEVICES or torch.cuda.set_device first")
        self.cdt = dtype
        self.rdt = torch.float64 if dtype == torch.complex128 else torch.float32
        self.wavelengths_nm = tuple(float(w) for w in (wavelengths_nm if hasattr(wavelengths_nm, "__iter__") else (wavelengths_nm,)))
        self.rays = tuple(rays)
        self.chunk = chunk
        self.verbose_guard = verbose_guard
        self.guard_events: list[dict] = []
        self.orders = tuple(int(o) for o in stack.fourier_orders)
        ox, oy = self.orders
        self.order_list = stack.order_list()
        self.n_orders = len(self.order_list)
        if config.factorization == "li" and (stack.lattice_angle_deg != 90 or self.order_list != truncated_orders(self.orders)):
            raise ValueError("factorization='li' needs lattice_angle_deg 90 and the rectangular truncation")
        self.shape = tuple(int(s) for s in stack.density_shape)

        self.solve_spec = []
        for li, wl_nm in enumerate(self.wavelengths_nm):
            wl = wl_nm / 1000.0
            n_s = _eps(stack.eps_in, wl) ** 0.5
            if abs(n_s.imag) > 1e-12 * abs(n_s):
                raise ValueError("the superstrate (eps_in) must be lossless")
            n_s = float(n_s.real)
            for ri, ray in enumerate(self.rays):
                s_in = float(ray.reference_index) * math.sin(ray.theta_rad) / n_s
                if not 0 <= s_in < 1:
                    raise ValueError("pupil ray cannot refract into the superstrate")
                self.solve_spec.append((li, ri, wl, math.asin(s_in), float(ray.phi_rad)))
        self.ray_weight = torch.tensor([r.weight for r in self.rays], dtype=self.rdt, device=self.device)

        cdt = self.cdt
        self.eps_bg = torch.tensor([_eps(stack.eps_fill, w / 1000) for w in self.wavelengths_nm], dtype=cdt,
                                   device=self.device)
        self.eps_design = torch.tensor([_eps(stack.eps_design, w / 1000) for w in self.wavelengths_nm], dtype=cdt,
                                       device=self.device)
        if bool((self.eps_bg.imag < 0).any()) or bool((self.eps_design.imag < 0).any()):
            raise ValueError("Im(eps) < 0 is gain in the exp(-i w t) convention used here; write an absorbing index as n + ik")

        # Toeplitz gather indices of the permittivity convolution matrix
        gx, gy = torch.tensor(self.order_list, device=self.device).unbind(1)
        self.conv_ix = (gx[:, None] - gx[None, :]) % self.shape[0]
        self.conv_iy = (gy[:, None] - gy[None, :]) % self.shape[1]
        self.li_mx = (torch.arange(2 * ox + 1, device=self.device)[:, None] - torch.arange(2 * ox + 1, device=self.device)[None, :]) % self.shape[0]
        self.li_my = (torch.arange(2 * oy + 1, device=self.device)[:, None] - torch.arange(2 * oy + 1, device=self.device)[None, :]) % self.shape[1]
        self.li_gx, self.li_gy = gx + ox, gy + oy
        self.box_index = (gx + ox) * (2 * oy + 1) + gy + oy              # of the orders in the full rectangle, for the detector plane

        # detector quadrants
        sx, sy = stack.detector_samples
        lx, ly = stack.period_um
        ax = (torch.arange(sx, device=self.device, dtype=self.rdt) + 0.5) * (lx / sx)
        ay = (torch.arange(sy, device=self.device, dtype=self.rdt) + 0.5) * (ly / sy)
        low_x = (ax < lx / 2).reshape(-1, 1)
        low_y = (ay < ly / 2).reshape(1, -1)
        masks = {"R": low_x & low_y, "G2": ~low_x & low_y, "G1": low_x & ~low_y, "B": ~low_x & ~low_y}
        self.masks = torch.stack([masks[w].to(self.rdt) for w in WELL_ORDER])
        self.area_element = (lx / sx) * (ly / sy)
        self._geo_cache: dict[tuple, SolveGeometry] = {}

    # ------------------------------------------------------------------ geometry
    def geometry(self, indices: Sequence[int], nudges: Sequence[float]) -> SolveGeometry:
        key = (tuple(indices), tuple(nudges))
        if key not in self._geo_cache:
            st = self.stack
            solves = [(*self.solve_spec[i], nudge) for i, nudge in zip(indices, nudges)]
            self._geo_cache[key] = build_geometry(                     # in complex128: eps - kx^2 - ky^2 cancels near a Rayleigh point
                solves, orders=self.orders, detector_samples=st.detector_samples, period=st.period_um,
                thickness=st.thickness_um, detector_z=st.detector_z_um, device=self.device,
                eps_in_fn=lambda wl: _eps(st.eps_in, wl), eps_out_fn=lambda wl: _eps(st.eps_out, wl), cdt=torch.complex128,
                order_list=self.order_list, lattice_angle_deg=st.lattice_angle_deg).to(self.cdt)
        return self._geo_cache[key]

    def _geometry_pairs(self, pairs, nudges) -> SolveGeometry:
        """Geometry of (density slot, solve) pairs: the solve's constants, E rows offset by slot."""
        key = ("pairs", tuple(pairs), tuple(nudges))
        if key not in self._geo_cache:
            base = self.geometry([b for _, b in pairs], nudges)
            offset = torch.as_tensor([d * len(self.wavelengths_nm) for d, _ in pairs],
                                     dtype=base.lam_index.dtype, device=base.lam_index.device)
            self._geo_cache[key] = dataclasses.replace(base, lam_index=base.lam_index + offset)
        return self._geo_cache[key]

    # ------------------------------------------------------------------ physics
    def _econv(self, densities: torch.Tensor):
        """Permittivity matrices per (density, wavelength), row d * L + lambda: [E_xx, E_yy] [rows, 2, N, N], or [E_xx, E_yy, E_xy = E_yx] with the
        rule "normal", and E^-1 = [[eps]]^-1 (for E_zz).  The rule is SolverConfig.factorization (see :class:`spectra_rcwa.config.SolverConfig`)."""
        rho = densities.to(self.cdt)
        eps_map = (self.eps_bg[None, :, None, None]
                   + rho[:, None] * (self.eps_design - self.eps_bg)[None, :, None, None])
        spec = (torch.fft.fft2(eps_map) / (self.shape[0] * self.shape[1])).reshape(-1, *self.shape)
        e = _take(spec, self.conv_ix, self.conv_iy)
        rule = self.config.factorization
        if rule == "laurent" or (rule == "auto" and min(self.orders) > 0):
            return e[:, None].expand(-1, 2, -1, -1), lapack.inv(e)
        if rule == "normal" and min(self.orders) > 0:
            inv = torch.fft.fft2(1.0 / eps_map) / (self.shape[0] * self.shape[1])
            inv = lapack.inv(_take(inv.reshape(-1, *self.shape), self.conv_ix, self.conv_iy))
            return in_plane(e, inv, self._tensor(densities).repeat_interleave(len(self.wavelengths_nm), 0)), lapack.inv(e)
        sx, sy = self.shape
        inv_map = (1.0 / eps_map).reshape(-1, sx, sy)
        tx = _take(torch.fft.fft(inv_map, dim=1) / sx, self.li_mx)                   # [[1/eps]]_x at each y: [rows, Mx, Mx, Sy]
        fx = torch.fft.fft(lapack.inv(tx.permute(0, 3, 1, 2)), dim=1) / sy           # its inverse, then the Fourier series in y: [rows, Sy, Mx, Mx]
        ty = _take(torch.fft.fft(inv_map, dim=2).transpose(1, 2) / sy, self.li_my)   # [[1/eps]]_y at each x: [rows, My, My, Sx]
        fy = torch.fft.fft(lapack.inv(ty.permute(0, 3, 1, 2)), dim=1) / sx           # [rows, Sx, My, My]
        gx, gy = self.li_gx, self.li_gy
        exx = _take(fx, self.conv_iy, gx[:, None], gx[None, :])
        eyy = _take(fy, self.conv_ix, gy[:, None], gy[None, :])
        return torch.stack((exx, eyy), 1), lapack.inv(e)

    def _tensor(self, densities: torch.Tensor):
        """Toeplitz matrices [D, 3, N, N] of the normal tensor of densities [D, H, W]."""
        st = self.stack
        t = normal_tensor(densities, st.period_um, reciprocal(st.period_um, st.lattice_angle_deg))
        spec = torch.fft.fft2(t) / (t.shape[-2] * t.shape[-1])
        return _take(spec.reshape(-1, *t.shape[-2:]), self.conv_ix, self.conv_iy).reshape(-1, 3, self.n_orders, self.n_orders).to(self.cdt)

    def _rows(self, layers):
        """:meth:`_econv` of a list of layers, each a density [H, W] or its permittivity matrices (E [L, 2, N, N], E^-1 [L, N, N])."""
        sampled = [x for x in layers if not isinstance(x, tuple)]
        if len(sampled) == len(layers):
            return self._econv(torch.stack(sampled))
        n_wl = len(self.wavelengths_nm)
        rows = iter(zip(*(a.split(n_wl) for a in self._econv(torch.stack(sampled))))) if sampled else iter(())
        pairs = [x if isinstance(x, tuple) else next(rows) for x in layers]
        return torch.cat([e for e, _ in pairs]), torch.cat([einv for _, einv in pairs])

    def _p(self, einv_all, g: SolveGeometry):
        n = self.n_orders
        ei = einv_all[g.lam_index]
        kxc, kyc = g.kx[:, :, None], g.ky[:, :, None]
        kxr, kyr = g.kx[:, None, :], g.ky[:, None, :]
        eye = torch.eye(n, dtype=ei.dtype, device=ei.device).expand_as(ei)
        return torch.cat((torch.cat((kxc * ei * kyr, eye - kxc * ei * kxr), -1),
                          torch.cat((kyc * ei * kyr - eye, -kyc * ei * kxr), -1)), -2)

    def _pq(self, e_all, einv_all, g: SolveGeometry):
        """P and Q."""
        e = e_all[g.lam_index]
        exx, eyy = e[:, 0], e[:, 1]
        kx, ky = g.kx, g.ky
        exy = e[:, 2] if e.shape[1] == 3 else torch.zeros_like(exx)
        q = torch.cat((torch.cat((torch.diag_embed(-kx * ky) - exy, torch.diag_embed(kx ** 2) - eyy), -1),
                       torch.cat((exx - torch.diag_embed(ky ** 2), torch.diag_embed(ky * kx) + exy), -1)), -2)
        return self._p(einv_all, g), q

    def _layer(self, e_all, einv_all, g: SolveGeometry):
        """P and the layer matrix A = P Q."""
        p, q = self._pq(e_all, einv_all, g)
        return p, p @ q

    def _index_phase(self, g: SolveGeometry) -> float:
        """w max(|eps|^(1/2)) over the solves of ``g``: per unit thickness, how fast the field of a propagating order turns in the layer."""
        return float(torch.maximum(self.eps_design.abs(), self.eps_bg.abs()).amax().sqrt() * g.omega.abs().max())

    def _prepare(self):
        """Process-wide settings that the kernels and the remaining torch.linalg calls run with."""
        cfg = self.config
        if self.device.type == "cuda":
            if cfg.linalg_library:
                torch.backends.cuda.preferred_linalg_library(cfg.linalg_library)
            if self.cdt == torch.complex64 and torch.backends.cuda.matmul.allow_tf32 and not ColorRouterForward._tf32_warned:
                ColorRouterForward._tf32_warned = True
                warnings.warn("torch.backends.cuda.matmul.allow_tf32 is on: TF32 products cost this solver up to 8e-2 in the gradient", stacklevel=3)
        _lincomb.ENABLED = cfg.fused_kernels

    @staticmethod
    def _max_phase(g: SolveGeometry) -> float:
        """w max(k) over the solves of ``g``: per unit thickness, the largest growth rate of an evanescent order."""
        return float(((g.kx.abs() ** 2 + g.ky.abs() ** 2).amax(1).sqrt() * g.omega.abs()).max())

    def _layer_block(self, e_all, einv_all, g: SolveGeometry, thickness: float):
        ops = adjoint.LayerOps(e_all[g.lam_index], einv_all[g.lam_index], g.kx, g.ky)
        doublings = blocks.doublings_for(self._max_phase(g) * thickness, self.config)
        return blocks.layer_block(ops, g.omega, thickness, blocks.reference_admittance(g), self.config, doublings)

    def _solves(self, slot: int, solves) -> SolveGeometry:
        solves = range(len(self.solve_spec)) if solves is None else solves
        return self._geometry_pairs([(slot, b) for b in solves], [float(self.stack.rayleigh_nudge)] * len(solves))

    def layer_block(self, density: torch.Tensor, thickness_um: float, solves=None):
        """S-matrix (T, R) of a patterned layer of uniform thickness (S11 = S22 = T, S12 = S21 = R, each [b, 2N, 2N]) in a fixed
        reference basis, for all solves or those listed.  It depends on this layer alone: keep it and pass it to stack_field, with the same
        ``solves``, for as long as the layer does not change."""
        self._prepare()
        e_all, einv_all = self._rows([density])
        return self._layer_block(e_all, einv_all, self._solves(0, solves), float(thickness_um))

    def stack_field(self, layers, solves=None, return_top: bool = False):
        """Field below a stack for both polarisations, [b, 2N, 2], with return_top also (below, top), between the superstrate and
        substrate of ``self.stack``.  ``layers``, top first: (density [H, W], thickness in um) or a stored layer_block.
        In place of a density, the permittivity matrices of :meth:`_rows`."""
        self._prepare()
        stored = [torch.is_tensor(x[1]) and x[1].ndim == 3 for x in layers]          # a stored block is (T, R); anything else is (density, thickness)
        densities = [x[0] for x, s in zip(layers, stored) if not s]
        e_all, einv_all = self._rows(densities) if densities else (None, None)
        items, slot = [], 0
        for x, s in zip(layers, stored):
            if not s:
                items.append((self._solves(slot, solves), float(x[1])))
                slot += 1
            else:
                items.append((self._solves(0, solves), tuple(x)))
        return self._through_layers(e_all, einv_all, items, return_top)

    def efficiencies(self, layers, solves=None):
        """(T, R), each [b, N, 2]: the power in every diffraction order of the substrate and of the superstrate over the incident power, for the
        stack of :meth:`stack_field`.  The orders are those of ``order_list``; in the rectangular truncation (m, n) has index
        (m + ox) * (2 oy + 1) + (n + oy).  The columns are the two incident polarisations."""
        g = self._solves(0, solves)
        below, top = self.stack_field(layers, solves, return_top=True)
        n = self.n_orders

        def flux(a, y):
            h = y.left(a)
            return (a[:, :n] * h[:, n:].conj() - a[:, n:] * h[:, :n].conj()).real

        incident = flux(g.e_in, g.vi).sum(1, keepdim=True)
        return flux(below, g.vo) / incident, flux(top - g.e_in, g.vi) / incident

    def _hand_adjoint(self) -> bool:
        return self.config.handwritten_backward

    def _through_layers(self, e_all, einv_all, layers, return_top: bool = False):
        """Field at the bottom of a stack for both polarisations, [b, 2N, 2]; with return_top also at its top.  ``layers`` = [(geometry,
        thickness or a stored layer_block)], top first, the geometries differing in the row of the permittivity matrices only (see the top
        of this module).  On a stored block that needs a gradient, autograd goes through the whole stack."""
        stored = [x for _, layer in layers if isinstance(layer, tuple) for x in layer]
        if self._hand_adjoint() and e_all is not None and not any(x.requires_grad for x in stored):
            keep = torch.is_grad_enabled() and (e_all.requires_grad or einv_all.requires_grad)
            out, top = adjoint.StackSolve.apply(e_all, einv_all, self, layers, keep)
        else:
            out, top, _ = adjoint.solve_stack(self, e_all, einv_all, layers)
        return (out, top) if return_top else out

    def _gamma_exp(self, a, g: SolveGeometry):
        gam = matfun.gamma(a, self.config)
        e = matfun.expm_taylor((1j * g.omega[:, None, None] * g.thickness) * gam, self.config)
        return gam, e

    def _transmitted(self, p, a, g: SolveGeometry):
        """Transmitted field at the bottom interface for both polarisations: [b, 2N, 2]."""
        cfg = self.config
        gam, e = self._gamma_exp(a, g)
        if cfg.boundary == "smatrix":
            return transmitted_smatrix(p, gam, e, g)
        ki, ko = bdd_pair(p, g.vi, g.vo, fused=cfg.fused_kernels)
        kind = cfg.boundary_inverse
        return boundary_direct(gam, e, ki, ko, g.e_in, lambda x: matfun.raw_inverse(x, kind))

    def _wells(self, out: torch.Tensor, g: SolveGeometry, strict: bool = True):
        """Quadrant fractions x exact transmission for both polarisations: [b, 4, 2], T [b, 2]."""
        if g.phase_x is None:
            raise ValueError("the wells of a colour router need lattice_angle_deg 90; use efficiencies")
        n = self.n_orders
        ox, oy = self.orders
        exy = out * g.out_phase[:, :, None]
        hxy = g.vo.left(exy)
        ex, ey = exy[:, :n], exy[:, n:]
        hx, hy = hxy[:, :n], hxy[:, n:]
        ez = (g.ky[:, :, None] * hx - g.kx[:, :, None] * hy) / g.eps_out[:, None, None]
        comp = torch.stack((ex, ey, ez), 1)
        if n != (2 * ox + 1) * (2 * oy + 1):
            comp = comp.new_zeros(out.shape[0], 3, (2 * ox + 1) * (2 * oy + 1), 2).index_copy(2, self.box_index, comp)
        comp = comp.reshape(out.shape[0], 3, 2 * ox + 1, 2 * oy + 1, 2)
        grid = torch.einsum("bxi,bcijp,byj->bcxyp", g.phase_x, comp, g.phase_y)
        intensity = (grid.real ** 2 + grid.imag ** 2).sum(1)
        field = torch.einsum("bxyp,wxy->bwp", intensity, self.masks) * self.area_element
        total = field.sum(1, keepdim=True)
        ok = torch.isfinite(total.detach()) & (total.detach() > 0)
        if strict and not bool(ok.all()):
            raise RuntimeError("detector-plane field power is not finite and positive: an order of the superstrate or substrate may sit on a "
                               "Rayleigh point (Stack.rayleigh_nudge > 0 moves the frequency off it)")
        fractions = field / torch.where(ok, total, torch.ones_like(total))
        exo, eyo = out[:, :n], out[:, n:]
        c = g.coef[:, :, :, None]
        op = c[:, 0] * exo + c[:, 1] * eyo
        os_ = c[:, 2] * exo + c[:, 3] * eyo
        trans = (op.real ** 2 + op.imag ** 2 + os_.real ** 2 + os_.imag ** 2).sum(1)
        if not strict:                                                 # a retry of the energy guard: a failed solve is one that is not finite
            trans = torch.where(ok[:, 0], trans, torch.full_like(trans, float("nan")))
        return fractions * trans[:, None, :], trans

    def _run(self, pairs, nudges, e_all, einv_all, strict: bool = True):
        """Wells [n, 4, 2] and transmission [n, 2] for (slot, solve) pairs, in order."""
        cfg = self.config
        self._prepare()
        chunk = self.chunk or len(pairs)
        wells, trans = [], []
        for s in range(0, len(pairs), chunk):
            e_ = min(s + chunk, len(pairs))
            g = self._geometry_pairs(pairs[s:e_], nudges[s:e_])
            if cfg.boundary == "entire":
                out = self._through_layers(e_all, einv_all, [(g, g.thickness)])
            else:
                p, a = self._layer(e_all, einv_all, g)
                out = self._transmitted(p, a, g)
            wl, tr = self._wells(out, g, strict)
            wells.append(wl)
            trans.append(tr)
        return torch.cat(wells), torch.cat(trans)

    # ------------------------------------------------------------------ public
    def response(self, density: torch.Tensor) -> torch.Tensor:
        """Tabs[well, wavelength] (float64) for one density [H, W] with values in [0, 1]."""
        return self.responses(density[None])[0]

    def responses(self, densities: torch.Tensor) -> torch.Tensor:
        """Tabs[density, well, wavelength] for D densities [D, H, W], all solves in one batch."""
        if densities.ndim != 3 or tuple(densities.shape[1:]) != self.shape:
            raise ValueError("densities must be [D, *density_shape]")
        n_d = densities.shape[0]
        e_all, einv_all = self._econv(densities)
        n_solves = len(self.solve_spec)
        pairs = [(d, b) for d in range(n_d) for b in range(n_solves)]
        main = [float(self.stack.rayleigh_nudge)] * len(pairs)
        wells, trans = self._run(pairs, main, e_all, einv_all)
        bad = (~torch.isfinite(trans.detach())) | (trans.detach() > 1.0 + ENERGY_GUARD_EPS)
        if bool(bad.any()):
            wells = self._guarded(pairs, wells, trans, bad, e_all, einv_all)
        return torch.stack([self._aggregate(wells[d * n_solves:(d + 1) * n_solves]) for d in range(n_d)])

    def _guarded(self, pairs, wells, trans, bad, e_all, einv_all):
        """Energy guard: re-solve only failing (solve, polarisation) at alternate Rayleigh nudges."""
        idx_bad = torch.nonzero(bad.any(1)).flatten().tolist()
        chosen: dict[tuple[int, int], float] = {}
        with torch.no_grad():
            for nudge in ENERGY_GUARD_RETRY_NUDGES:
                pending = [(r, p) for r in idx_bad for p in range(2) if bool(bad[r, p]) and (r, p) not in chosen]
                if not pending:
                    break
                rs = sorted({r for r, _ in pending})
                _, tr = self._run([pairs[r] for r in rs], [nudge] * len(rs), e_all, einv_all, strict=False)
                for row, r in enumerate(rs):
                    for p in range(2):
                        ok = bool(torch.isfinite(tr[row, p])) and float(tr[row, p]) <= 1.0 + ENERGY_GUARD_EPS
                        if (r, p) in pending and ok:
                            chosen[(r, p)] = nudge
                            event = dict(density=pairs[r][0], wavelength_um=self.solve_spec[pairs[r][1]][2],
                                         ray=self.solve_spec[pairs[r][1]][1], polarization="xy"[p],
                                         transmission=float(trans[r, p]), nudge_used=nudge,
                                         transmission_retry=float(tr[row, p]))
                            self.guard_events.append(event)
                            if self.verbose_guard:
                                print("energy guard retry: " + ", ".join(f"{k}={v}" for k, v in event.items()),
                                      flush=True)
        unresolved = [(r, p) for r in idx_bad for p in range(2) if bool(bad[r, p]) and (r, p) not in chosen]
        if unresolved:
            r, p = unresolved[0]
            raise RuntimeError(f"energy guard: solve {self.solve_spec[pairs[r][1]]} pol {'xy'[p]} is not finite or exceeds "
                               f"{1 + ENERGY_GUARD_EPS} at every retry nudge {ENERGY_GUARD_RETRY_NUDGES}")
        combos = sorted({(r, n) for (r, _), n in chosen.items()})
        rw, _ = self._run([pairs[r] for r, _ in combos], [n for _, n in combos], e_all, einv_all)
        pos = {c: i for i, c in enumerate(combos)}
        cols = []
        for p in range(2):
            src = wells[:, :, p]
            rows = [rw[pos[(r, chosen[(r, p)])], :, p] if (r, p) in chosen else None for r in range(wells.shape[0])]
            if any(x is not None for x in rows):
                src = torch.stack([x if x is not None else src[r] for r, x in enumerate(rows)])
            cols.append(src)
        return torch.stack(cols, -1)

    def _aggregate(self, wells: torch.Tensor) -> torch.Tensor:
        per = wells.sum(-1).reshape(len(self.wavelengths_nm), len(self.rays), 4)
        tabs = (per * (self.ray_weight / 2.0)[None, :, None]).sum(1).T
        tabs = tabs.to(torch.float64)
        if not bool(torch.isfinite(tabs.detach()).all()):
            raise RuntimeError("response contains non-finite values")
        if bool((tabs.detach() < 0).any()):
            raise RuntimeError("response contains negative power")
        return tabs


def value_and_grad(forward: ColorRouterForward, density: torch.Tensor, objective: Callable[[torch.Tensor], torch.Tensor],
                   tf32_backward: bool | None = None, objective_on_cpu: bool = True):
    """Objective value and its gradient with respect to the density, as benchmarked.

    ``objective`` maps Tabs [4, L] (float64) to a scalar.  With ``objective_on_cpu`` the small table is moved to
    the host first (a 4 x L objective is latency-bound on a GPU).  With ``tf32_backward`` the backward pass runs
    with TF32 matmuls allowed, except the Denman-Beavers steps, whose backward always runs in full FP32.  None: only on
    the "direct" route of one layer; on the default route TF32 costs far more gradient accuracy than time, in every stage.
    Returns (tabs, value, grad).
    """
    rdt = torch.float64 if forward.cdt == torch.complex128 else torch.float32
    rho = density.to(device=forward.device, dtype=rdt).detach().clone().requires_grad_(True)
    tabs = forward.response(rho)
    obj = objective(tabs.cpu() if objective_on_cpu else tabs)
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = forward._tf32_backward() if tf32_backward is None else tf32_backward
    try:
        obj.backward()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    return tabs.detach(), float(obj), rho.grad.detach()
