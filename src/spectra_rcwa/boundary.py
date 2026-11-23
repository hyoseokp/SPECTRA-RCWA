"""Boundary conditions of one patterned layer between two homogeneous half-spaces, through Gamma and E (not the default
route; see :meth:`spectra_rcwa.forward.ColorRouterForward._through_layers`).

Inside the layer the transverse fields are E(z) = exp(iwGz) u + exp(iwG(d - z)) v and H = Y_L(...), with
P Y_L = Gamma (Gamma = (PQ)^(1/2), E = exp(iwd Gamma)).  Matching E and H at the top (incident field J, input
admittance Y_i) and at the bottom (output admittance Y_o), and multiplying the H equations by P (K = P Y):

    [ G + Ki        (Ki - G) E ] [u]   [2 Ki J]
    [ (Ko - G) E     G + Ko    ] [v] = [  0   ],          t = E u + v   (transmitted field)

Block elimination with two inverses:

    Z = (G + Ko)^-1 Bo,   S = G + Ki - Bi Z,   u = S^-1 2 Ki J,   v = -Z u      (Bi = (Ki - G)E, Bo = (Ko - G)E)

Adjoint (r right-hand sides; O(n^2 r) besides the two saved inverses):

    lu = S^-H (E - Z)^H g_t,          lv = (G + Ko)^-H (g_t - Bi^H lu)
    gG  = lu (Ev - u)^H + lv (Eu - v)^H
    gKi = lu (2J - u - Ev)^H,         gKo = -lv t^H
    gE  = (g_t - (Ko - G)^H lv) u^H - ((Ki - G)^H lu) v^H

:func:`transmitted_smatrix` is the reference route through the layer S-matrix and Redheffer products
(torcwa's formulation with a free-space gap basis); both give the same field.
"""
from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from . import lapack


class BoundaryDirect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gam, e, ki, ko, j, inv_fn):
        dg_i, dg_o = ki - gam, ko - gam
        bi, bo = dg_i @ e, dg_o @ e
        xo = inv_fn(gam + ko)
        z = xo @ bo
        s = torch.baddbmm(gam + ki, bi, z, beta=1.0, alpha=-1.0)
        sinv = inv_fn(s)
        u = sinv @ (2.0 * (ki @ j))
        v = -(z @ u)
        eu = e @ u
        t = eu + v
        ctx.save_for_backward(e, dg_i, dg_o, bi, xo, z, sinv, j, u, v, eu)
        return t

    @staticmethod
    @once_differentiable
    def backward(ctx, g):
        e, dg_i, dg_o, bi, xo, z, sinv, j, u, v, eu = ctx.saved_tensors
        lu = sinv.mH @ (e.mH @ g - z.mH @ g)
        lv = xo.mH @ (g - bi.mH @ lu)
        ev = e @ v
        t = eu + v
        lam = torch.cat((lu, lv), -1)
        g_gam = lam @ torch.cat((ev - u, eu - v), -1).mH
        g_ki = lu @ (2.0 * j - u - ev).mH
        g_ko = -(lv @ t.mH)
        a1, a2 = dg_i.mH @ lu, dg_o.mH @ lv
        g_e = torch.cat((g - a2, -a1), -1) @ torch.cat((u, v), -1).mH
        return g_gam, g_e, g_ki, g_ko, None, None


def boundary_direct(gam, e, ki, ko, j, inv_fn=None):
    """Transmitted field t [B, n, r] for incident fields j [B, n, r] (``j`` is treated as a constant).

    ``inv_fn`` computes a batched inverse without autograd (default torch.linalg.inv_ex)."""
    if inv_fn is None:
        inv_fn = lapack.inv_unchecked
    return BoundaryDirect.apply(gam, e, ki, ko, j, inv_fn)


def boundary_direct_reference(gam, e, ki, ko, j):
    """The same block elimination written with plain differentiable torch ops (for tests)."""
    bi, bo = (ki - gam) @ e, (ko - gam) @ e
    z = lapack.solve(gam + ko, bo)
    s = gam + ki - bi @ z
    u = lapack.solve(s, 2.0 * (ki @ j))
    v = -(z @ u)
    return e @ u + v


def layer_smatrix(p, gam, e, vf_inv):
    """Layer transmission T and reflection R (symmetric layer: S11 = S22 = T, S12 = S21 = R) in the free-space
    gap basis, from Gamma and E instead of eigenpairs (X diag(f(kz)) X^-1 = f(Gamma)):

        W = V0^-1 P^-1 Gamma,   G+- = (I + W +- (I - W) E)^-1
        T = E (G+ + G-) + (G+ - G-),   R = (G+ + G-) + E (G+ - G-) - I
    """
    n = gam.shape[-1]
    eye = torch.eye(n, dtype=gam.dtype, device=gam.device)
    w = vf_inv.left(lapack.solve(p, gam))
    ap, am = eye + w, eye - w
    ame = am @ e
    gp, gm = lapack.inv(ap + ame), lapack.inv(ap - ame)
    sig, dif = gp + gm, gp - gm
    return e @ sig + dif, sig + e @ dif - eye


def transmitted_smatrix(p, gam, e, geo):
    """Transmitted field through input S-matrix, layer, output S-matrix (Redheffer products, linear solves)."""
    lay11, lay21 = layer_smatrix(p, gam, e, geo.vf_inv)
    n = gam.shape[-1]
    eye = torch.eye(n, dtype=gam.dtype, device=gam.device)
    r1 = eye - geo.sin12.left(lay21)
    rhs = torch.cat((geo.sin11.left(geo.e_in), geo.sin12.left(lay11)), -1)
    sol = lapack.solve(r1, rhs)
    k = geo.e_in.shape[-1]
    s1v = lay11 @ sol[..., :k]
    s12p = lay21 + lay11 @ sol[..., k:]
    r2 = eye - geo.sout21.right(s12p)
    return geo.sout11.left(lapack.solve(r2, s1v))
