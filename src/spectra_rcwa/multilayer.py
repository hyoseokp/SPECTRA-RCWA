"""Stack of several patterned layers (no spacers), each solved without eigendecomposition.

By default the admittance goes up the stack through entire functions of each layer's A = P Q
(:meth:`spectra_rcwa.forward.ColorRouterForward._through_layers`).  With the other routes every layer gets its own
Gamma = (P Q)^(1/2) and E = exp(i w d Gamma) (:mod:`spectra_rcwa.matfun`) and its S-matrix in the free-space gap basis
(:func:`spectra_rcwa.boundary.layer_smatrix`), the layers are combined top to bottom by Redheffer products and closed by the
input / output half-space S-matrices; the vacuum gap puts real poles at guided modes of the layers, so this route is a
reference only.  A one-layer stack gives the same field as the direct boundary solve, and a layer of thickness d the
same as two layers of thickness d/2 (tests/test_multilayer.py).
"""
from __future__ import annotations

import dataclasses
from typing import Sequence

import torch

from . import lapack, matfun
from .boundary import layer_smatrix
from .forward import ColorRouterForward


def star(a, b):
    """Redheffer product of two blocks (T, Rt, Tb, Rb), ``a`` above ``b``.

    T / Tb: downward / upward transmission, Rt / Rb: reflection seen from the top / bottom."""
    ta, rta, tba, rba = a
    tb, rtb, tbb, rbb = b
    eye = torch.eye(ta.shape[-1], dtype=ta.dtype, device=ta.device)
    m1 = lapack.inv(eye - rba @ rtb)
    m2 = lapack.inv(eye - rtb @ rba)
    return (tb @ m1 @ ta, rta + tba @ rtb @ m1 @ ta, tba @ m2 @ tbb, rbb + tb @ rba @ m2 @ tbb)


def transmitted_through(block, geo):
    """Transmitted field for the incident fields ``geo.e_in`` through input S-matrix, ``block``, output S-matrix."""
    t, rt, tb, rb = block
    eye = torch.eye(t.shape[-1], dtype=t.dtype, device=t.device)
    k = geo.e_in.shape[-1]
    r1 = eye - geo.sin12.left(rt)
    sol = lapack.solve(r1, torch.cat((geo.sin11.left(geo.e_in), geo.sin12.left(tb)), -1))
    t1 = t @ sol[..., :k]
    rb_all = rb + t @ sol[..., k:]
    r2 = eye - geo.sout21.right(rb_all)
    return geo.sout11.left(lapack.solve(r2, t1))


class MultiLayerForward(ColorRouterForward):
    """Like :class:`~spectra_rcwa.ColorRouterForward`, for a stack of patterned layers of the given thicknesses
    (top first; ``stack.thickness_um`` is ignored).  Densities are [layers, H, W]; all layers share the stack's
    design and fill materials.  ``chunk`` bounds the solves per batch on the default route."""

    def _tf32_backward(self) -> bool:
        return False

    def __init__(self, *, thicknesses_um: Sequence[float], **kw):
        super().__init__(**kw)
        self.thicknesses_um = tuple(float(d) for d in thicknesses_um)
        if not self.thicknesses_um:
            raise ValueError("need at least one layer")

    def responses(self, densities: torch.Tensor) -> torch.Tensor:
        """Tabs[stack, well, wavelength] for S stacks [S, layers, H, W], one stack after the other."""
        if densities.ndim != 4:
            raise ValueError("densities must be [stacks, layers, *density_shape]")
        return torch.stack([self.response(d) for d in densities])

    def response(self, density: torch.Tensor) -> torch.Tensor:
        """Tabs[well, wavelength] (float64) for a density [layers, H, W] with values in [0, 1]."""
        n_layers = len(self.thicknesses_um)
        if density.ndim != 3 or density.shape[0] != n_layers or tuple(density.shape[1:]) != self.shape:
            raise ValueError("density must be [layers, *density_shape]")
        cfg = self.config
        self._prepare()
        e_all, einv_all = self._econv(density)                        # row = layer * L + wavelength
        n = len(self.solve_spec)
        nudges = [float(self.stack.rayleigh_nudge)] * n
        if cfg.boundary == "entire":
            chunk, wells = self.chunk or n, []
            for s in range(0, n, chunk):
                idx = range(s, min(s + chunk, n))
                layers = [(self._geometry_pairs([(li, b) for b in idx], nudges[:len(idx)]), d) for li, d in enumerate(self.thicknesses_um)]
                w, trans = self._wells(self._through_layers(e_all, einv_all, layers), layers[0][0])
                if bool(((~torch.isfinite(trans.detach())) | (trans.detach() > 1.02)).any()):
                    raise RuntimeError("transmission exceeds 1 (energy check failed)")
                wells.append(w)
            return self._aggregate(torch.cat(wells))
        block, geo0 = None, None
        for li, d in enumerate(self.thicknesses_um):
            g = dataclasses.replace(self._geometry_pairs([(li, b) for b in range(n)], nudges), thickness=d)
            if geo0 is None:
                geo0 = g
            p, a = self._layer(e_all, einv_all, g)
            gam, e = self._gamma_exp(a, g)
            t, r = layer_smatrix(p, gam, e, g.vf_inv)
            layer = (t, r, t, r)                                       # a single layer is symmetric
            block = layer if block is None else star(block, layer)
        wells, trans = self._wells(transmitted_through(block, geo0), geo0)
        if bool(((~torch.isfinite(trans.detach())) | (trans.detach() > 1.02)).any()):
            raise RuntimeError("transmission exceeds 1 (energy check failed)")
        return self._aggregate(wells)
