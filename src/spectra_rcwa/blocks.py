"""S-matrix of a uniform layer in a fixed reference basis, built by doubling: for thick layers, and to store a layer and reuse it.

A thin step h of the layer, from the entire functions of A = P Q (:func:`spectra_rcwa.matfun.cos_sinc`), is written in the wave basis of
a reference admittance Y0 (e = a + b, h = Y0 (a - b)), and Redheffer products of the layer with itself double the thickness: r doublings give
2^r steps with r + 1 inverses, against one inverse per step for the admittance recursion.  A uniform layer is symmetric, so one
(T, R) describes both sides and each doubling needs one inverse:

    T2 = T (I - R R)^-1 T,   R2 = R + T (I - R R)^-1 R T

Y0 belongs to a homogeneous medium with a lossy fictitious permittivity, so that it has no Rayleigh point and I - R R no real pole at a
guided mode of the layer, as a vacuum basis would have.  It depends on (kx, ky) only, so blocks of different layers share it.
"""
from __future__ import annotations

import math

import torch

from . import matfun
from .config import SolverConfig
from .geometry import BDD, SolveGeometry, homogeneous_v
from .kernels.thin import thin_mn

EPS_REFERENCE = 4.0 + 4.0j


def reference_admittance(g: SolveGeometry) -> BDD:
    """The admittance Y0 of the reference basis for the solves of ``g``."""
    return homogeneous_v(torch.as_tensor(EPS_REFERENCE, dtype=g.kx.dtype, device=g.kx.device), g.kx, g.ky)


def thin_block(ops, e, sc, gc, wh, y0: BDD, cfg: SolverConfig):
    """(T, R) of a step w h of a layer with the matrices ``ops`` (:class:`spectra_rcwa.adjoint.LayerOps`), from the output of cos_sinc, in the basis of ``y0``."""
    z0 = y0.inverse()
    s = 1j * wh * sc
    sp, qs, qgp = ops.p_right(s), ops.q_left(s), ops.q_left(ops.p_right((wh * wh) * gc))
    if e.is_cuda and e.dtype == torch.complex64 and cfg.fused_kernels and not torch.is_grad_enabled():
        m, k = thin_mn(e, sp, qs, qgp, y0, z0)
    else:                                                              # (C + C') / 2, (C - C') / 2 with C = I + E, C' = I + Y0^-1 Q G P Y0; i S P Y0, Y0^-1 i Q S
        ss, qq, gg = y0.right(sp), z0.left(qs), y0.right(z0.left(qgp))
        m, k = torch.eye(e.shape[-1], dtype=e.dtype, device=e.device) + 0.5 * (e + gg - ss - qq), 0.5 * (e - gg - ss + qq)
    t = matfun.inverse(m, cfg.layer_inverse)
    return t, k @ t


def double_block(t, r, doublings: int, cfg: SolverConfig, keep: bool = False):
    """(T, R, tree) of ``doublings`` Redheffer products of a symmetric block with itself; with ``keep``, tree = [(t, r, x = (I - r r)^-1)] of the halves, thinnest first."""
    tree = []
    for _ in range(doublings):
        m = torch.baddbmm(r, r, r, beta=0.0, alpha=-1.0)               # I - r r
        m.diagonal(dim1=-2, dim2=-1).add_(1.0)
        x = matfun.inverse(m, cfg.layer_inverse)
        if keep:
            tree.append((t, r, x))
        w = t @ x
        t, r = w @ t, torch.baddbmm(r, w, r @ t)
    return t, r, tree


def layer_block(ops, omega, thickness: float, y0: BDD, cfg: SolverConfig, doublings: int):
    """(T, R), both [B, 2N, 2N], of a uniform layer with the matrices ``ops`` in the basis of ``y0``; ``omega`` [B]."""
    wh = omega[:, None, None] * (thickness / 2 ** doublings)
    e, sc, gc = matfun.cos_sinc(ops.pq(), wh, cfg)
    return double_block(*thin_block(ops, e, sc, gc, wh, y0, cfg), doublings, cfg)[:2]


def doublings_for(phase: float, cfg: SolverConfig) -> int:
    """Doublings so that a thin step has w h max(k) at most cfg.max_step_phase; ``phase`` = w d max(k)."""
    return max(1, int(math.ceil(math.log2(max(phase / cfg.max_step_phase, 1.0)))))
