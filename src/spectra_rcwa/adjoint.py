"""The default route through a stack, and its gradient from one adjoint field instead of a graph.

Forward, from the output medium upwards: the admittance recursion or, for a thick layer, the S-matrix by doubling
(:mod:`spectra_rcwa.blocks`).  The objective needs the field for two incident polarisations only, so everything the backward pass does
is a product with a few columns.  For a variation of P and Q of a layer, with e' = i w P h and h' = i w Q e,

    dJ = Re int i w (a^H dP h + b^H dQ e) dz,      a' = i w Q^H b,   b' = i w P^H a

where (a, b) is the adjoint field, with a = V_i^H b at the top of the stack and a + V_o^H b = dJ/d(field below) at its bottom.  The factors
saved by the forward pass give (e, h) and (a, b) on the planes between the steps of each layer, stable in both directions, in a few products
with two columns.  Between the planes the integrand is smooth: a short Taylor series in P and Q from each plane (order set by the size of the
step) gives the fields on Gauss-Legendre nodes, and the integral is a low-rank sum.  With P = J + K_L E^-1 K_R and
Q = -K_L K_R + [[0, -E_yy], [E_xx, 0]] a product with P costs a quarter of a dense product and one with Q half of it.

There is no eigenvector, no square root and no division by an eigenvalue here either: an order at cutoff is not special.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.autograd.function import once_differentiable

from . import blocks, matfun
from .geometry import BDD
from .kernels import single, structured as S
from .kernels.bdd_mul import bdd_left, bdd_right
from .kernels.lincomb import lincomb
from .kernels.pivlu import lu_factor, lu_solve, lu_solve_h
from .kernels.product import baddbmm, matmul


class LayerOps:
    """P and Q of a layer for a batch of solves: e = [E_xx, E_yy] [b, 2, N, N] or [E_xx, E_yy, E_xy = E_yx], einv [b, N, N], kx, ky [b, N].  Without a graph by structure; under autograd dense, because the graph of the
    products by structure would hold far more."""

    def __init__(self, e, einv, kx, ky):
        self.e, self.einv = e, einv
        self.exx, self.eyy = e[:, 0], e[:, 1]
        self.exy = e[:, 2] if e.shape[1] == 3 else None
        self.kx, self.ky = kx.real, ky.real                            # the transverse wave vector is real
        self.n = kx.shape[-1]
        self.fused = e.is_cuda and e.dtype == torch.complex64 and not torch.is_grad_enabled()
        self.p, self.q = None, None
        if torch.is_grad_enabled():
            kxc, kyc = self.kx[:, :, None], self.ky[:, :, None]
            kxr, kyr = self.kx[:, None, :], self.ky[:, None, :]
            eye = torch.eye(self.n, dtype=e.dtype, device=e.device).expand_as(einv)
            self.p = torch.cat((torch.cat((kxc * einv * kyr, eye - kxc * einv * kxr), -1), torch.cat((kyc * einv * kyr - eye, -kyc * einv * kxr), -1)), -2)
            exy = self.exy if self.exy is not None else torch.zeros_like(self.exx)
            self.q = torch.cat((torch.cat((torch.diag_embed((-kx * ky).to(e.dtype)) - exy, torch.diag_embed((kx ** 2).to(e.dtype)) - self.eyy), -1),
                                torch.cat((self.exx - torch.diag_embed((ky ** 2).to(e.dtype)), torch.diag_embed((ky * kx).to(e.dtype)) + exy), -1)), -2)

    def e_left(self, x1, x2):
        """(E_yy x2 + E_yx x1, E_xx x1 + E_xy x2)."""
        u1, u2 = self.eyy @ x2, self.exx @ x1
        if self.exy is not None:
            u1, u2 = torch.baddbmm(u1, self.exy, x1), torch.baddbmm(u2, self.exy, x2)
        return u1, u2

    def eh_left(self, x1, x2):
        """(E_xx^H x2 - E_yx^H x1, E_yy^H x1 - E_xy^H x2)."""
        u1, u2 = self.exx_h @ x2, self.eyy_h @ x1
        if self.exy is not None:
            u1, u2 = torch.baddbmm(u1, self.exy_h, x1, alpha=-1.0), torch.baddbmm(u2, self.exy_h, x2, alpha=-1.0)
        return u1, u2

    def q_left(self, x):
        """Q @ x."""
        if self.q is not None:
            return self.q @ x
        n = self.n
        x1, x2 = x[:, :n], x[:, n:]
        if self.fused:
            w = S.proj_rows(x, self.ky, -self.kx)
            one = torch.ones(x.shape[0], dtype=x.dtype, device=x.device)
            return S.lift_rows(w, *self.e_left(x1, x2), -self.kx, -self.ky, one, -1, 1, n * x.shape[2])
        kx, ky = self.kx[:, :, None], self.ky[:, :, None]
        w = ky * x1 - kx * x2
        u1, u2 = self.e_left(x1, x2)
        return torch.cat((-kx * w - u1, -ky * w + u2), 1)

    def p_right(self, x):
        """x @ P."""
        if self.p is not None:
            return x @ self.p
        if self.fused:
            z = S.proj_cols(x, self.kx, self.ky) @ self.einv
            return S.lift_cols(z, x, self.ky, -self.kx, -1, 1)
        n = self.n
        x1, x2 = x[:, :, :n], x[:, :, n:]
        kx, ky = self.kx[:, None, :], self.ky[:, None, :]
        z = (x1 * kx + x2 * ky) @ self.einv
        return torch.cat((z * ky - x2, x1 - z * kx), 2)

    def pq(self):
        """A = P Q = [[E_xx, E_xy], [E_yx, E_yy]] - [[ky^2, -kx ky], [-kx ky, kx^2]] - K_L E^-1 [kx E_xx + ky E_yx, kx E_xy + ky E_yy], since K_R K_L = 0."""
        if self.p is not None:
            return self.p @ self.q
        n, kx, ky = self.n, self.kx, self.ky
        kxc, kyc = kx[:, :, None], ky[:, :, None]
        m1, m2 = kxc * self.exx, kyc * self.eyy
        if self.exy is not None:
            m1, m2 = m1 + kyc * self.exy, m2 + kxc * self.exy
        m = torch.cat((matmul(self.einv, m1), matmul(self.einv, m2)), 2)               # [b, N, 2N]
        a = torch.cat((-kxc * m, -kyc * m), 1)
        a[:, :n, :n] += self.exx
        a[:, n:, n:] += self.eyy
        if self.exy is not None:
            a[:, :n, n:] += self.exy
            a[:, n:, :n] += self.exy
        d = a.diagonal(dim1=1, dim2=2)
        d[:, :n] -= ky * ky
        d[:, n:] -= kx * kx
        a[:, :n, n:].diagonal(dim1=1, dim2=2).add_(kx * ky)
        a[:, n:, :n].diagonal(dim1=1, dim2=2).add_(kx * ky)
        return a


def _mul(y, x):
    return y.left(x) if isinstance(y, BDD) else y @ x


def _mulh(y, x):
    return y.mh().left(x) if isinstance(y, BDD) else y.mH @ x


def _dense(y):
    return y.dense() if isinstance(y, BDD) else y


@dataclass
class _Recursion:
    """A layer solved by the admittance recursion, top to bottom: ys[k] above step k, ys[k + 1] below it; e below = ds[k] e above."""

    ops: LayerOps
    rows: torch.Tensor
    omega: torch.Tensor
    h: float
    x: float
    sp: torch.Tensor
    qgp: torch.Tensor
    ys: list = field(default_factory=list)
    ds: list = field(default_factory=list)


@dataclass
class _Block:
    """A layer through its S-matrix; ``tree`` = [(t, r, x)] of its halves, thinnest first, empty for a stored block.  At the top of the stack ``x_top`` instead of ``inv_top``."""

    t: torch.Tensor
    r: torch.Tensor
    y0: BDD
    rho: object
    xb: torch.Tensor
    tx: torch.Tensor
    rho_top: torch.Tensor
    inv_top: torch.Tensor | None = None
    x_top: object = None
    rho_i: BDD | None = None
    a_top: torch.Tensor | None = None
    ops: LayerOps | None = None
    rows: torch.Tensor | None = None
    omega: torch.Tensor | None = None
    h: float = 0.0
    x: float = 0.0
    tree: list = field(default_factory=list)

    def below(self, a_b):
        """Field below the block from the wave going down there."""
        return a_b + _mul(self.rho, a_b)


@dataclass
class _State:
    """What the backward pass needs: the layers top to bottom; for a top of the stack that is not a block, the inverse of V_i + Y and V_i."""

    segs: list
    top_inv: object
    vi: BDD
    top: torch.Tensor


def _shift(x):
    """I + x for a fresh x."""
    x.diagonal(dim1=-2, dim2=-1).add_(1.0)
    return x


class _Factored:
    """Solves with a matrix that needs pivoting and two right-hand sides: its LU factors on the GPU, in complex64 without a graph; else its inverse."""

    def __init__(self, m, cfg):
        if m.is_cuda and m.dtype == torch.complex64 and not torch.is_grad_enabled():
            self.lib = single.few(m)
            self.lu, self.perm = single.lu_factor(m) if self.lib else lu_factor(m)
            self.inv = None
        else:
            self.inv = matfun.inverse(m, cfg.top_inverse)

    def solve(self, x):
        if self.lu is None:
            return self.inv @ x
        return single.lu_solve(self.lu, self.perm, x) if self.lib else lu_solve(self.lu, self.perm, x)

    def solve_h(self, x):
        if self.lu is None:
            return self.inv.mH @ x
        return single.lu_solve(self.lu, self.perm, x, adjoint=True) if self.lib else lu_solve_h(self.lu, self.perm, x)

    lu = None


def _through_block(block, y, y0, cfg, vi=None):
    """(admittance on top, state of the block) for the admittance ``y`` below it.  With ``vi`` the block is the top of the stack and only the
    field for the two incident waves is needed: a = rho_i b + tau J above it, one factorisation of I - rho_i rho_top and no admittance."""
    t, r = block
    if isinstance(y, BDD):
        rho = (y0 + y).inverse().matmul_bdd(y0 - y)                    # b = rho a below the block
        tx_in, rho_tx = bdd_right(rho, r, -1, 1), lambda x: bdd_left(rho, x)
    else:
        rho = matmul(matfun.inverse(y0.dense() + y, cfg.top_inverse), y0.dense() - y)
        tx_in, rho_tx = _shift(-matmul(r, rho)), lambda x: matmul(rho, x)
    xb = matfun.inverse(tx_in, cfg.layer_inverse)
    tx = matmul(xb, t)                                                   # a below = tx a above
    rho_top = baddbmm(r, t, rho_tx(tx))
    seg = _Block(t, r, y0, rho, xb, tx, rho_top)
    if vi is not None:
        seg.rho_i = (y0 + vi).inverse().matmul_bdd(y0 - vi)
        seg.x_top = _Factored(bdd_left(seg.rho_i, rho_top, -1, 1), cfg)
        return None, seg
    seg.inv_top = matfun.inverse(_shift(rho_top.clone()), cfg.layer_inverse)
    return bdd_left(y0, matmul(_shift(-rho_top), seg.inv_top)), seg


def solve_stack(fwd, e_all, einv_all, layers, keep=False):
    """(field below, field on top, state) for [(geometry, thickness or a stored block)], top first; differentiable if run under autograd.  ``keep``: what the backward pass needs."""
    cfg = fwd.config
    g0 = layers[0][0]
    phase = fwd._max_phase(g0)
    turn = max(phase, fwd._index_phase(g0))                            # the fields turn with w sqrt(|eps|) as well: the adjoint integral follows the larger
    y, segs = g0.vo, []
    for i, (g, layer) in reversed(list(enumerate(layers))):
        vi = g0.vi if i == 0 else None
        if isinstance(layer, tuple):
            y, seg = _through_block(layer, y, blocks.reference_admittance(g), cfg, vi)
            segs.append(seg)
            continue
        ops = LayerOps(e_all[g.lam_index], einv_all[g.lam_index], g.kx, g.ky)
        cut = max(phase, turn / 2.0) * layer                          # w d max(k), and no step with w h |eps|^(1/2) above twice max_step_phase
        steps = max(1, int(math.ceil(cut / cfg.max_step_phase)))
        if steps >= cfg.block_from_steps:
            doublings = blocks.doublings_for(cut, cfg)
            y0 = blocks.reference_admittance(g)
            h = layer / 2 ** doublings
            wh = g.omega[:, None, None] * h
            e, sc, gc = matfun.cos_sinc(ops.pq(), wh, cfg)
            t, r = blocks.thin_block(ops, e, sc, gc, wh, y0, cfg)
            del e, sc, gc
            t, r, tree = blocks.double_block(t, r, doublings, cfg, keep)
            y, seg = _through_block((t, r), y, y0, cfg, vi)
            seg.ops, seg.rows, seg.omega, seg.h, seg.x, seg.tree = ops, g.lam_index, g.omega.real, h, turn * h / 2.0, tree
            segs.append(seg)
            continue
        h = layer / steps
        y = _dense(y)
        wh = g.omega[:, None, None] * h
        e, sc, gc = matfun.cos_sinc(ops.pq(), wh, cfg)
        s = 1j * wh * sc                                              # i S, with S = w h sc; G = (w h)^2 gc
        c, sp, qs, qgp = _shift(e), ops.p_right(s), ops.q_left(s), ops.q_left(ops.p_right((wh * wh) * gc))
        del sc, gc, s
        seg = _Recursion(ops, g.lam_index, g.omega.real, h, turn * h / 2.0, sp, qgp, ys=[y])
        for _ in range(steps):
            d = matfun.inverse(lincomb([c, matmul(sp, y)], [1.0, -1.0]), cfg.step_inverse)
            y = matmul(lincomb([y, matmul(qgp, y), qs], [1.0, 1.0, -1.0]), d)
            if keep:
                seg.ys.insert(0, y)
            seg.ds.insert(0, d)
        segs.append(seg)
    segs.reverse()
    source = 2.0 * g0.vi.left(g0.e_in)
    if y is None:                                                      # a block on top
        seg = segs[0]
        seg.a_top = seg.x_top.solve((seg.y0 + g0.vi).inverse().left(source))
        top = seg.a_top + seg.rho_top @ seg.a_top
        out, top_inv = seg.below(seg.tx @ seg.a_top), None
        rest = segs[1:]
    else:
        top_inv = _Factored(g0.vi.dense() + _dense(y), cfg)
        top = top_inv.solve(source)
        out, rest = top, segs
    for seg in rest:
        if isinstance(seg, _Block):
            out = seg.below(seg.tx @ (seg.inv_top @ out))
        else:
            for d in seg.ds:
                out = d @ out
    return out, top, _State(segs, top_inv, g0.vi, top.detach())       # detached: the state must not hold the output of its own Function


_GAUSS: dict = {}
ADJOINT_STEP = 2.25                                                    # half the default max_step_phase: the size of step the node counts are for


def _gauss(nodes):
    """Half of the Gauss-Legendre rule of ``nodes`` (even) on [0, 1]: t in (0, 1), with s = t / 2, and the weights."""
    if nodes not in _GAUSS:
        x, w = np.polynomial.legendre.leggauss(nodes)
        keep = x < 0
        _GAUSS[nodes] = (x[keep] + 1.0, w[keep] / 2.0)
    return _GAUSS[nodes]


def _order(x, tol):
    """Smallest J with x^(J + 1) / (J + 1)! at most ``tol``."""
    j, term = 0, x
    while term > tol:
        j += 1
        term *= x / (j + 1)
    return j


def _layer_gradient(seg, e_pl, h_pl, a_pl, b_pl, g_e, g_einv, cfg):
    """Add the integral over a layer, from its planes (each [b, 2N, 2], top to bottom), to the gradients with respect to [E_xx, E_yy] and E^-1."""
    ops, n = seg.ops, seg.ops.n
    dt = ops.e.dtype
    nodes = cfg.adjoint_nodes if dt == torch.complex64 else 16        # for a step of ADJOINT_STEP, and in proportion beyond it
    t_nodes, w_nodes = _gauss(2 * int(math.ceil(nodes * max(1.0, seg.x / ADJOINT_STEP) / 2.0)))
    order = _order(seg.x, cfg.adjoint_tol if dt == torch.complex64 else 1e-16)
    rdt = ops.kx.dtype
    sigma = 1j * seg.omega.to(dt)[:, None, None] * (seg.h / 2.0)
    planes = len(e_pl)
    cols = e_pl[0].shape[-1]
    ee, hh, aa, bb = torch.cat(e_pl, -1), torch.cat(h_pl, -1), torch.cat(a_pl, -1), torch.cat(b_pl, -1)
    es, wh, bs, wa = [ee], [], [bb], []
    einv_h, ops.exx_h, ops.eyy_h = ops.einv.mH, ops.exx.mH, ops.eyy.mH
    ops.exy_h = ops.exy.mH if ops.exy is not None else None
    kx, ky = ops.kx, ops.ky
    if ops.fused:
        per = 2 * n * ee.shape[2]
        for j in range(order + 1):
            w_h, w_a = S.proj_rows(hh, ky, -kx), S.proj_rows(aa, kx, ky)
            wh.append(w_h)
            wa.append(w_a)
            if j == order:
                break
            s = (sigma / (j + 1)).reshape(-1)
            new_e = S.lift_rows(ops.einv @ w_h, hh[:, n:], hh[:, :n], kx, ky, s, 1, -1, per)
            hh = S.lift_rows(S.proj_rows(ee, ky, -kx), *ops.e_left(ee[:, :n], ee[:, n:]), -kx, -ky, s, -1, 1, per // 2)
            ee = new_e
            es.append(ee)
            new_b = S.lift_rows(einv_h @ w_a, aa[:, n:], aa[:, :n], ky, -kx, s, -1, 1, per)
            aa = S.lift_rows(S.proj_rows(bb, kx, ky), *ops.eh_left(bb[:, :n], bb[:, n:]), -ky, kx, s, 1, -1, per // 2)
            bb = new_b
            bs.append(bb)
    else:
        kxc, kyc = kx[:, :, None], ky[:, :, None]
        for j in range(order + 1):
            h1, h2 = hh[:, :n], hh[:, n:]
            w_h = kyc * h1 - kxc * h2
            wh.append(w_h)
            a1, a2 = aa[:, :n], aa[:, n:]
            w_a = kxc * a1 + kyc * a2
            wa.append(w_a)
            if j == order:
                break
            s = sigma / (j + 1)
            z = ops.einv @ w_h
            new_e = s * torch.cat((kxc * z + h2, kyc * z - h1), 1)     # (i w t h / 2)^j / j! e_j
            e1, e2 = ee[:, :n], ee[:, n:]
            w_e = kyc * e1 - kxc * e2
            u1, u2 = ops.e_left(e1, e2)
            hh = s * torch.cat((-kxc * w_e - u1, -kyc * w_e + u2), 1)
            ee = new_e
            es.append(ee)
            z = einv_h @ w_a
            new_b = s * torch.cat((kyc * z - a2, a1 - kxc * z), 1)
            b1, b2 = bb[:, :n], bb[:, n:]
            w_b = kxc * b1 + kyc * b2
            u1, u2 = ops.eh_left(b1, b2)
            aa = s * torch.cat((-kyc * w_b + u1, kxc * w_b - u2), 1)
            bb = new_b
            bs.append(bb)
    powers = torch.as_tensor(np.stack([t_nodes ** j for j in range(order + 1)], 1), dtype=dt, device=ops.e.device)     # [q, J]

    def at_nodes(items):
        """[b, rows, m, J] of Taylor terms -> values on the nodes inside the layer, [b, rows, cols * q * 2 * (planes - 1)], the node index fastest."""
        st = torch.stack(items, -1)
        ev, od = st[..., 0::2] @ powers[:, 0::2].T, st[..., 1::2] @ powers[:, 1::2].T
        below, above = (ev + od)[:, :, :cols * (planes - 1)], (ev - od)[:, :, cols:]     # none below the last plane, none above the first
        return torch.cat((below, above), 2).flatten(2)

    en, wn_h, bn, wn_a = at_nodes(es), at_nodes(wh), at_nodes(bs), at_nodes(wa)
    w = torch.as_tensor(np.tile(w_nodes, 2 * cols * (planes - 1)), dtype=rdt, device=ops.e.device)[None, None, :]
    factor = (-1j * seg.omega.to(dt))[:, None, None] * seg.h
    g_einv.index_add_(0, seg.rows, factor * ((wn_a * w) @ wn_h.mH))
    b1, b2, e1, e2 = bn[:, :n], bn[:, n:], en[:, :n], en[:, n:]
    bw1, bw2 = b1 * w, b2 * w
    parts = [bw2 @ e1.mH, -(bw1 @ e2.mH)]
    if ops.exy is not None:
        parts.append(torch.baddbmm(bw2 @ e2.mH, bw1, e1.mH, alpha=-1.0))
    g_e.index_add_(0, seg.rows, factor[:, None] * torch.stack(parts, 1))


def _block_planes(seg, a_t, b_t, a_b, b_b, p_t, q_t, p_b, q_b):
    """Fields on the planes of a block from the wave amplitudes (a, b) and adjoint (p, q) at its ends, by its tree."""
    levels = len(seg.tree)
    count = 2 ** levels
    aa, bb, pp, qq = {0: a_t}, {0: b_t}, {0: p_t}, {0: q_t}
    aa[count], bb[count], pp[count], qq[count] = a_b, b_b, p_b, q_b
    m = a_t.shape[-1]
    for t, r, x in reversed(seg.tree):
        stride = count
        count //= 2
        pos = range(0, 2 ** levels, stride)
        at, bbot = torch.cat([aa[i] for i in pos], -1), torch.cat([bb[i + stride] for i in pos], -1)
        qt, pbot = torch.cat([qq[i] for i in pos], -1), torch.cat([pp[i + stride] for i in pos], -1)
        tb = t @ bbot
        am = x @ (t @ at + r @ tb)
        bm = r @ am + tb
        th = t.mH
        rh = r.mH
        tq = th @ qt
        pm = x.mH @ (th @ pbot - rh @ tq)
        qm = tq - rh @ pm
        for k, i in enumerate(pos):
            aa[i + count], bb[i + count] = am[..., k * m:(k + 1) * m], bm[..., k * m:(k + 1) * m]
            pp[i + count], qq[i + count] = pm[..., k * m:(k + 1) * m], qm[..., k * m:(k + 1) * m]
    y0, z0h = seg.y0, seg.y0.inverse().mh()
    idx = sorted(aa)
    e_pl = [aa[i] + bb[i] for i in idx]
    h_pl = [y0.left(aa[i] - bb[i]) for i in idx]
    a_pl = [0.5 * (pp[i] + qq[i]) for i in idx]
    b_pl = [0.5 * z0h.left(pp[i] - qq[i]) for i in idx]
    return e_pl, h_pl, a_pl, b_pl


def adjoint_gradient(state, g_out, g_top, e_all, einv_all, cfg):
    """Gradients with respect to [E_xx, E_yy(, E_xy)] and E^-1 of each row, for dJ = Re tr(g_out^H d(field below) + g_top^H d(field on top))."""
    segs = state.segs
    g_e, g_einv = torch.zeros_like(e_all), torch.zeros_like(einv_all)
    # forward field on the ends of the layers, and c = a + Y^H b from the bottom up
    e = state.top
    ends = []
    for seg in segs:
        if isinstance(seg, _Block):
            a_t = seg.a_top if seg.x_top is not None else seg.inv_top @ e
            ends.append(a_t)
            e = seg.below(seg.tx @ a_t)
        else:
            pl = [e]
            for d in seg.ds:
                e = d @ e
                pl.append(e)
            ends.append(pl)
    c = g_out if g_out is not None else torch.zeros_like(e)
    cs = []
    for seg in reversed(segs):
        if isinstance(seg, _Block):
            cs.append(c)
            if seg.x_top is None:
                c = seg.inv_top.mH @ (seg.tx.mH @ (c + _mulh(seg.rho, c)))
        else:
            cl = [c]
            for d in reversed(seg.ds):
                c = d.mH @ c
                cl.append(c)
            cs.append(cl[::-1])
    cs.reverse()
    if state.top_inv is not None:
        b = state.top_inv.solve_h(c + g_top if g_top is not None else c)
        a = state.vi.mh().left(b)
        if g_top is not None:
            a = a - g_top
    # the adjoint field from the top down, the integral over each layer
    for seg, end, cb in zip(segs, ends, cs):
        if isinstance(seg, _Block):
            a_t = end
            y0 = seg.y0
            chat = cb + _mulh(seg.rho, cb)                              # (I + rho)^H c below
            if seg.x_top is not None:
                ct = seg.tx.mH @ chat
                if g_top is not None:
                    gi = g_top + seg.rho_i.mh().left(g_top)
                    p_t = seg.x_top.solve_h(ct + seg.rho_top.mH @ gi)
                    q_t = -seg.rho_i.mh().left(p_t) - gi
                else:
                    p_t = seg.x_top.solve_h(ct)
                    q_t = -seg.rho_i.mh().left(p_t)
            else:
                y0h = y0.mh()
                p_t, q_t = a + y0h.left(b), a - y0h.left(b)
            b_t = seg.rho_top @ a_t
            a_b = seg.tx @ a_t
            b_b = _mul(seg.rho, a_b)
            x = seg.t.mH @ q_t - seg.r.mH @ chat
            q_b = x + seg.r.mH @ (seg.xb.mH @ _mulh(seg.rho, x))        # (I - rho R)^-1 = I + rho (I - R rho)^-1 R
            p_b = chat - _mulh(seg.rho, q_b)
            if seg.ops is not None:
                _layer_gradient(seg, *_block_planes(seg, a_t, b_t, a_b, b_b, p_t, q_t, p_b, q_b), g_e, g_einv, cfg)
            a = 0.5 * (p_b + q_b)
            b = 0.5 * y0.inverse().mh().left(p_b - q_b)
        else:
            e_pl, h_pl, a_pl, b_pl = end, [], [], []
            for k, (y, c_k) in enumerate(zip(seg.ys, cb)):
                if k > 0:
                    b = b + seg.qgp.mH @ b - seg.sp.mH @ a              # across a step: the row of M^H
                    a = c_k - _mulh(y, b)
                h_pl.append(_mul(y, e_pl[k]))
                a_pl.append(a)
                b_pl.append(b)
            _layer_gradient(seg, e_pl, h_pl, a_pl, b_pl, g_e, g_einv, cfg)
    return g_e, g_einv


class StackSolve(torch.autograd.Function):
    """Field below and on top of a stack as a function of [E_xx, E_yy(, E_xy)] and E^-1 of each row, with the backward :func:`adjoint_gradient`."""

    @staticmethod
    def forward(ctx, e_all, einv_all, fwd, layers, keep):
        out, top, state = solve_stack(fwd, e_all, einv_all, layers, keep)
        if keep:
            ctx.state, ctx.cfg = state, fwd.config
            ctx.save_for_backward(e_all, einv_all)
        return out, top

    @staticmethod
    @once_differentiable
    def backward(ctx, g_out, g_top):
        e_all, einv_all = ctx.saved_tensors
        return (*adjoint_gradient(ctx.state, g_out, g_top, e_all, einv_all, ctx.cfg), None, None, None)
