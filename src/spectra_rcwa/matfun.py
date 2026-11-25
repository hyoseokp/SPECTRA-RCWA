"""Matrix functions of the layer matrix A = P Q, without eigenvectors.

Entire functions (the route used by default).  With x^2 = (w h)^2 A:  cos(x) - 1, sin(x) / x and (cos(x) - 1) / x^2
are power series in A, with no square root and no branch: :func:`cos_sinc`.  One scaling 4^s for the batch, a polynomial
by Paterson-Stockmeyer, s double-angle steps, GEMMs only.  The double-angle steps work on cos(x) - 1 rather than cos(x),
which would lose digits while x is small, and (cos(x) - 1) / x^2 = -(sin(x/2) / (x/2))^2 / 2 needs no division by A.

The other route goes through Gamma = (P Q)^(1/2) and E = exp(i w d Gamma).  Gamma is not analytic at an eigenvalue 0 of A,
which is where an order has its cutoff: the value is smooth there only through cancellations, which a complex64 A cannot
hold and 4 iterations do not reach (the relative gradient error grows as 1 / lambda^2, also without complex64).
Square root.  The spectrum of A = P Q lies near the positive real axis (propagating orders, kz^2 > 0) and the
negative real axis (evanescent orders).  The principal root of -iA has its spectrum away from the branch cut,
and Gamma = e^(i pi/4) (-iA)^(1/2) gives kz real for propagating and Im kz > 0 for evanescent orders (the
branch torcwa uses).  (-iA)^(1/2) is computed by the product form of the scaled Denman-Beavers iteration

    M_0 = Y_0 = X,   mu_k = (||M_k^-1||_F / ||M_k||_F)^p
    Y_{k+1} = mu_k/2 Y_k + 1/(2 mu_k) Y_k M_k^-1
    M_{k+1} = I/2 + mu_k^2/4 M_k + 1/(4 mu_k^2) M_k^-1

(Y_k -> X^(1/2), M_k -> I), with p = 1/4 and Y_0 M_0^-1 = I used exactly in the first step, and finished by
a short series for M^(-1/2) once M is close to I:  X^(1/2) ~ Y (I + D)^(-1/2),  D = M - I.
The scale factors are treated as constants in the derivative (they do not change the converged result).

Exponential.  One scaling 2^s for the whole batch, Taylor polynomial by Paterson-Stockmeyer, s squarings.

Both have hand-written backward passes for complex64 on CUDA (DBStep, Squarings): every conjugate transpose is
materialised once and reused, every GEMM is in the nn layout, and accumulations ride in baddbmm or one fused
linear-combination kernel.
"""
from __future__ import annotations

import math

import torch
from torch.autograd.function import once_differentiable

from . import lapack
from .config import SolverConfig
from .kernels.blockinv import block_inverse, inv_block
from .kernels.lincomb import lincomb
from .kernels.product import baddbmm, matmul
from .kernels import single

_ROT = complex(math.cos(math.pi / 4), math.sin(math.pi / 4))
_TAIL_COEF = (1.0, -0.5, 0.375, -0.3125, 0.2734375)               # (1 + d)^(-1/2)


def _poly(c, z, z2, z3, f):
    """sum_k c[k] (f z)^k by Paterson-Stockmeyer on z, z2 = z @ z, z3 = z2 @ z; a coefficient that would multiply a scalar block goes in the block below."""
    deg = len(c) - 1
    out, f2, f3 = None, f * f, f * f * f
    for b in reversed(range(0, deg if deg % 3 == 0 else deg + 1, 3)):
        xs, cs = [z], [c[b + 1] * f]
        if b + 2 <= deg:
            xs.append(z2)
            cs.append(c[b + 2] * f2)
        if out is None and b + 3 == deg:
            xs.append(z3)
            cs.append(c[deg] * f3)
        block = lincomb(xs, cs, diag=c[b] if c[b] != 0.0 else None)
        out = block if out is None else baddbmm(block, z3, out, alpha=f3)
    return out


def cos_sinc(a: torch.Tensor, wh: torch.Tensor, cfg: SolverConfig):
    """(cos(x) - 1, sin(x) / x, (cos(x) - 1) / x^2) with x^2 = wh^2 a, for a batch [B, n, n] and wh [B, 1, 1]."""
    deg, theta = (cfg.cos_degree, cfg.cos_theta) if a.dtype == torch.complex64 else (8, 1.0)        # complex128: down to its rounding
    z = lincomb([a], [(wh * wh).reshape(-1)])
    z2 = matmul(z, z)
    z3 = matmul(z2, z)
    norm = max(float(torch.linalg.matrix_norm(z2.detach(), ord=1).max()) ** 0.5, float(torch.linalg.matrix_norm(z3.detach(), ord=1).max()) ** (1.0 / 3.0))
    s = max(1, int(math.ceil(math.log(max(norm / theta, 1.0)) / math.log(4.0))))
    f = 4.0 ** -s
    e = _poly([0.0] + [(-1) ** k / math.factorial(2 * k) for k in range(1, deg + 1)], z, z2, z3, f)
    sc = _poly([(-1) ** k / math.factorial(2 * k + 1) for k in range(deg + 1)], z, z2, z3, f)
    gc = None
    for k in range(s):
        if k == s - 1:
            gc = baddbmm(sc, sc, sc, beta=0.0, alpha=-0.5)
        sc = baddbmm(sc, sc, e)
        e = baddbmm(e, e, e, beta=4.0, alpha=2.0)
    return e, sc, gc


def _cuda64(x: torch.Tensor) -> bool:
    return x.is_cuda and x.dtype == torch.complex64


def _single(m: torch.Tensor, kind: str) -> bool:
    if kind == "block":
        return m.shape[0] == 1 and m.shape[-1] <= single.BLOCK_UP_TO
    return kind in ("pivlu", "permuted") and single.few(m)


def raw_inverse(m: torch.Tensor, kind: str) -> torch.Tensor:
    """Batched inverse without autograd.  ``kind``: "block", "pivlu" or "torch" (CUDA complex64 only;
    anything else uses torch.linalg.inv_ex, which does not synchronise with the host: :mod:`spectra_rcwa.lapack`)."""
    if _cuda64(m):
        if _single(m, kind):
            return single.inverse(m)
        if kind == "block":
            return block_inverse(m.contiguous())
        if kind == "pivlu":
            from .kernels.pivlu import pivoted_inverse
            return pivoted_inverse(m)
    return lapack.inv_unchecked(m)


def inverse(m: torch.Tensor, kind: str) -> torch.Tensor:
    """Batched inverse, differentiable (same choices as :func:`raw_inverse`, and "permuted": :func:`spectra_rcwa.kernels.pivlu.permuted_block_inverse`, under autograd "pivlu")."""
    if _cuda64(m):
        if not torch.is_grad_enabled() and _single(m, kind):
            return single.inverse(m)
        if kind == "permuted" and not torch.is_grad_enabled():
            from .kernels.pivlu import permuted_block_inverse
            return permuted_block_inverse(m)
        if kind == "block":
            return inv_block(m)
        if kind in ("pivlu", "permuted"):
            from .kernels.pivlu import inv_pivot
            return inv_pivot(m)
    return lapack.inv_unchecked(m)


def _db_coefficients(m, minv, scaled, power):
    if scaled:
        with torch.no_grad():
            mu = torch.pow(torch.linalg.matrix_norm(minv) / torch.linalg.matrix_norm(m), power).to(m.dtype)
    else:
        mu = torch.ones(m.shape[0], dtype=m.dtype, device=m.device)
    return 0.5 * mu, 0.5 / mu, 0.25 * mu * mu, 0.25 / (mu * mu)


class DBStep(torch.autograd.Function):
    """One product-form DB step with a hand-written backward.

        minv = M^-1,  Y' = a Y + b Y minv,  M' = c M + d minv + I/2       (a, b, c, d per matrix, constants)
        gY    = conj(a) gY' + conj(b) gY' minv^H
        gminv = conj(b) Y^H gY' + conj(d) gM'
        gM    = conj(c) gM' - minv^H gminv minv^H
    """

    @staticmethod
    def forward(ctx, m, y, scaled, power, inv_kind, first):
        minv = raw_inverse(m, inv_kind)
        a, b, c, d = _db_coefficients(m, minv, scaled, power)
        ctx.first = first
        if first:                                                     # Y0 = M0: Y0 M0^-1 = I exactly
            y2 = lincomb([y], [a], diag=b)
        else:
            y2 = lincomb([y, y @ minv], [a, b])
        m2 = lincomb([m, minv], [c, d], diag=0.5)
        ctx.save_for_backward(minv, y, a, b, c, d)
        return m2, y2

    @staticmethod
    @once_differentiable
    def backward(ctx, gm2, gy2):
        minv, y, a, b, c, d = ctx.saved_tensors
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False                 # TF32 here costs ~1 % gradient accuracy
        minvh = minv.mH.contiguous()
        if gy2 is None:
            gy2 = torch.zeros_like(y)
        if gm2 is None:
            gm2 = torch.zeros_like(minv)
        if ctx.first:
            gy = lincomb([gy2], [a.conj()])
            parts, cs = [gm2], [d.conj()]
        else:
            gy = lincomb([gy2, gy2 @ minvh], [a.conj(), b.conj()])
            parts, cs = [y.mH.contiguous() @ gy2, gm2], [b.conj(), d.conj()]
        gminv = lincomb(parts, cs)
        v = (minvh @ gminv) @ minvh
        gm = lincomb([gm2, v], [c.conj(), -1.0])
        torch.backends.cuda.matmul.allow_tf32 = prev
        return gm, gy, None, None, None, None


def db_step(m, y, k, cfg: SolverConfig):
    """Iteration ``k`` of the scaled product-form DB iteration; returns (M_{k+1}, Y_{k+1})."""
    scaled = k < cfg.db_scaled
    first = k == 0 and cfg.db_first_identity
    kind = "block" if k >= cfg.block_inverse_from else cfg.first_inverse
    if cfg.handwritten_backward and _cuda64(m):
        return DBStep.apply(m, y, scaled, cfg.db_scale_power, kind, first)
    minv = inverse(m, kind)
    a, b, c, d = _db_coefficients(m, minv, scaled, cfg.db_scale_power)
    y2 = lincomb([y], [a], diag=b) if first else lincomb([y, y @ minv], [a, b])
    m2 = lincomb([m, minv], [c, d], diag=0.5)
    return m2, y2


def sqrtm_db(x: torch.Tensor, cfg: SolverConfig) -> torch.Tensor:
    """Principal square root of a batch [B, n, n] whose spectrum avoids the closed negative real axis."""
    n = x.shape[-1]
    eye = torch.eye(n, dtype=x.dtype, device=x.device)
    m, y = x, x
    for k in range(cfg.db_iters):
        m, y = db_step(m, y, k, cfg)
    t = cfg.tail_order
    if t:                                                             # Y (I + D)^(-1/2), Horner without the I product
        d = m - eye
        s = _TAIL_COEF[t - 1] * eye + _TAIL_COEF[t] * d
        for c in reversed(_TAIL_COEF[:t - 1]):
            s = c * eye + d @ s
        y = y @ s
    return y


def gamma(a: torch.Tensor, cfg: SolverConfig) -> torch.Tensor:
    """Gamma = A^(1/2) on the branch Im(kz) >= 0, as e^(i pi/4) (-iA)^(1/2)."""
    return _ROT * sqrtm_db(-1j * a, cfg)


class Squarings(torch.autograd.Function):
    """E -> E^(2^s) with the backward gE = gE' E^H + E^H gE' per squaring (optionally on TF32)."""

    @staticmethod
    def forward(ctx, e, s, tf32_bwd):
        ctx.tf32_bwd = tf32_bwd
        es = [e]
        for _ in range(s):
            es.append(es[-1] @ es[-1])
        ctx.save_for_backward(*es[:-1])
        return es[-1]

    @staticmethod
    @once_differentiable
    def backward(ctx, g):
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = ctx.tf32_bwd
        for ek in reversed(ctx.saved_tensors):
            ekh = ek.mH.contiguous()
            g = torch.baddbmm(g @ ekh, ekh, g)
        torch.backends.cuda.matmul.allow_tf32 = prev
        return g, None, None


def expm_taylor(x: torch.Tensor, cfg: SolverConfig) -> torch.Tensor:
    """exp of a batch [B, n, n]: one scaling 2^s for the batch, Paterson-Stockmeyer Taylor, s squarings."""
    deg, theta = cfg.taylor_degree, cfg.exp_theta
    n = x.shape[-1]
    eye = torch.eye(n, dtype=x.dtype, device=x.device)
    norm = float(torch.linalg.matrix_norm(x.detach(), ord=1).max())
    s = max(0, int(math.ceil(math.log2(max(norm / theta, 1e-30)))))
    x = x / (2.0 ** s)
    q = int(round(math.sqrt(deg)))
    pw = [eye, x]
    for _ in range(q - 1):
        pw.append(pw[-1] @ x)
    xq = pw[q]
    c = [1.0 / math.factorial(k) for k in range(deg + 1)]
    e = None
    for blk in reversed(range(0, deg + 1, q)):
        js = [j for j in range(1, q) if blk + j <= deg]
        if js:
            poly = lincomb([pw[j] for j in js], [c[blk + j] for j in js], diag=c[blk])
        else:
            poly = c[blk] * eye.expand_as(x)
        e = poly if e is None else torch.baddbmm(poly, xq, e)
    if cfg.handwritten_backward and _cuda64(e):
        return Squarings.apply(e, s, cfg.tf32_squaring_backward)
    for _ in range(s):
        e = e @ e
    return e
