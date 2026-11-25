"""CUDA kernels and the complex64 CUDA path (skipped without a GPU)."""
import pytest
import torch

from conftest import layer_inputs, rel, requires_cuda, tiny_density, tiny_forward
from spectra_rcwa import DEFAULT, REFERENCE_C128
from spectra_rcwa import matfun

pytestmark = requires_cuda


def test_lincomb_kernel_vs_torch():
    from spectra_rcwa.kernels import lincomb as lc
    dev = torch.device("cuda")
    b, n = 8, 130
    xs = [torch.randn(b, n, n, dtype=torch.complex64, device=dev, requires_grad=True) for _ in range(3)]
    mu = torch.randn(b, dtype=torch.complex64, device=dev)
    coefs = [0.5 * mu, 1.0, -0.25 / mu]
    y = lc.lincomb(xs, coefs, diag=0.5)
    ref = (0.5 * mu[:, None, None] * xs[0] + xs[1] - (0.25 / mu)[:, None, None] * xs[2]
           + 0.5 * torch.eye(n, dtype=torch.complex64, device=dev))
    assert rel(y, ref) < 1e-6
    g = torch.randn_like(y)
    for u, v in zip(torch.autograd.grad(y, xs, g), torch.autograd.grad(ref, xs, g)):
        assert rel(u, v) < 1e-6


def test_bdd_pair_kernel_vs_torch():
    from spectra_rcwa.kernels.bdd_pair import bdd_pair
    fwd = tiny_forward(dtype=torch.complex64, config=DEFAULT, device="cuda")
    p, _, g = layer_inputs(fwd, tiny_density().float())
    p = p.detach().requires_grad_(True)
    k1, k2 = bdd_pair(p, g.vi, g.vo)
    r1, r2 = g.vi.right(p), g.vo.right(p)
    assert rel(k1, r1) < 1e-6 and rel(k2, r2) < 1e-6
    g1, g2 = torch.randn_like(k1), torch.randn_like(k2)
    assert rel(torch.autograd.grad((k1, k2), p, (g1, g2))[0],
               torch.autograd.grad((r1, r2), p, (g1, g2))[0]) < 1e-6


def test_inverses_vs_identity():
    from spectra_rcwa.kernels.blockinv import block_inverse
    from spectra_rcwa.kernels.pivlu import pivoted_inverse
    dev = torch.device("cuda")
    n = 290
    eye = torch.eye(n, dtype=torch.complex128, device=dev)
    a = torch.randn(16, n, n, dtype=torch.complex64, device=dev, generator=torch.Generator(dev).manual_seed(1))
    r_piv = (a.to(torch.complex128) @ pivoted_inverse(a).to(torch.complex128) - eye).abs().max()
    ad = a + 30 * eye.to(a.dtype)                                    # block inverse: no cross-block pivoting
    r_blk = (ad.to(torch.complex128) @ block_inverse(ad).to(torch.complex128) - eye).abs().max()
    assert float(r_piv) < 1e-3 and float(r_blk) < 1e-4


@pytest.mark.parametrize("dtype, n, batch", [(torch.complex64, 1800, 1), (torch.complex64, 2000, 2), (torch.complex128, 256, 3)])
def test_three_real_products(dtype, n, batch):
    from spectra_rcwa.kernels import product
    g = torch.Generator("cuda").manual_seed(0)
    a, b, c = (torch.randn(batch, n, n, dtype=dtype, device="cuda", generator=g) for _ in range(3))
    tol = 5e-6 if dtype == torch.complex64 else 1e-13
    assert product._as_three(a, b) or dtype == torch.complex128
    assert rel(product._three_real(a, b) if dtype == torch.complex128 else product.matmul(a, b), a @ b) < tol
    for alpha, beta in ((1.0, 1.0), (-0.5, 0.0), (2.0, 4.0), (-1.0, 1.0)):
        assert rel(product.baddbmm(c, a, b, beta=beta, alpha=alpha), torch.baddbmm(c, a, b, beta=beta, alpha=alpha)) < tol


def test_cusolver_route_equals_the_batched_kernels():
    """One solve takes cuSOLVER's factorisations, a batch the kernels: the same response and gradient."""
    from spectra_rcwa.kernels import single
    fwd = tiny_forward(dtype=torch.complex64, config=DEFAULT.replace(block_from_steps=1), device="cuda")
    out = []
    for few in (lambda m: True, lambda m: False):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(single, "few", few)
            r = tiny_density().float().cuda().requires_grad_(True)
            tabs = fwd.response(r)
            weights = torch.arange(12.0, device="cuda", dtype=tabs.dtype).reshape(4, 3)
            out.append((tabs.detach(), torch.autograd.grad((tabs * weights).sum(), r)[0]))
    assert rel(out[0][0], out[1][0]) < 1e-5 and rel(out[0][1], out[1][1]) < 1e-3
    library = torch.backends.cuda.preferred_linalg_library()
    single.inverse(torch.eye(4, dtype=torch.complex64, device="cuda")[None])
    assert torch.backends.cuda.preferred_linalg_library() == library


def test_handwritten_sqrt_backward_vs_plain():
    fwd = tiny_forward(dtype=torch.complex64, config=DEFAULT, device="cuda")
    _, a, _ = layer_inputs(fwd, tiny_density().float())
    x = (-1j * a).detach().requires_grad_(True)
    gout = torch.randn_like(x)
    y_hand = matfun.sqrtm_db(x, DEFAULT)
    y_plain = matfun.sqrtm_db(x, DEFAULT.replace(handwritten_backward=False))
    assert rel(y_hand, y_plain) < 1e-4
    assert rel(torch.autograd.grad(y_hand, x, gout)[0], torch.autograd.grad(y_plain, x, gout)[0]) < 1e-3


def test_cuda_value_and_gradient_at_cutoff():
    from spectra_rcwa import ColorRouterForward, value_and_grad
    from test_near_cutoff import RAY, WEIGHTS, WL_FAR, WL_P, WL_Q, _l2, _reference
    from conftest import tiny_stack
    rho = tiny_density()
    for wl in (WL_P, WL_Q, WL_FAR):
        v_ref, g_ref = _reference(wl, rho)
        fwd = ColorRouterForward(stack=tiny_stack(), rays=RAY, wavelengths_nm=(wl,), device="cuda", dtype=torch.complex64, config=DEFAULT)
        _, v, g = value_and_grad(fwd, rho.float(), lambda t: (t * WEIGHTS.to(t.device)).sum())
        assert abs(v - v_ref) < 2e-5 * abs(v_ref)
        assert _l2(g.double().cpu(), g_ref) < 3e-3


def test_cuda_d1_response_vs_cpu_reference():
    rho = tiny_density()
    ref = tiny_forward(config=REFERENCE_C128).response(rho)
    out = tiny_forward(dtype=torch.complex64, config=DEFAULT, device="cuda").response(rho.float().cuda()).cpu()
    assert rel(out, ref) < 1e-4


def _batch_residual(a, x, b):
    r = a.to(torch.complex128) @ x.to(torch.complex128) - b.to(torch.complex128)
    return float((r.flatten(1).norm(dim=1) / b.to(torch.complex128).flatten(1).norm(dim=1)).max())


@pytest.mark.parametrize("library", ["magma", "default"])
@pytest.mark.parametrize("n", [578, 4])
def test_batched_solvers_at_large_batch(library, n):
    """Every batched solver the routes use, 144 matrices or more that need pivoting.  See the next test for why the batch size is part of the test."""
    from spectra_rcwa.kernels.pivlu import lu_factor, lu_solve, lu_solve_h, permuted_block_inverse, pivoted_inverse
    prev = torch.backends.cuda.preferred_linalg_library()
    torch.backends.cuda.preferred_linalg_library(library)
    try:
        for batch in (144, 288):
            gen = torch.Generator(device="cuda").manual_seed(batch)
            a = torch.randn(batch, n, n, dtype=torch.complex64, device="cuda", generator=gen)
            b = torch.randn(batch, n, 2, dtype=torch.complex64, device="cuda", generator=gen)
            limit = 0.5                                               # the worst of these random matrices has a condition number near 1e5
            with torch.no_grad():
                lu, perm = lu_factor(a)
                assert _batch_residual(a, lu_solve(lu, perm, b), b) < limit
                assert _batch_residual(a.mH, lu_solve_h(lu, perm, b), b) < limit
                assert _batch_residual(a, torch.linalg.solve(a, b), b) < limit
                assert _batch_residual(a, torch.linalg.inv(a) @ b, b) < limit
                assert _batch_residual(a, torch.linalg.inv_ex(a)[0] @ b, b) < limit
                assert _batch_residual(a, pivoted_inverse(a) @ b, b) < limit
                assert _batch_residual(a, permuted_block_inverse(a) @ b, b) < limit
                factors, pivots = torch.linalg.lu_factor(a)
                assert _batch_residual(a, torch.linalg.lu_solve(factors, pivots, b), b) < limit
    finally:
        torch.backends.cuda.preferred_linalg_library(prev)


@pytest.mark.xfail(strict=False, reason="torch.linalg.solve_triangular is wrong for a complex64 batch of 16 or more with factors of a matrix that needs pivoting (torch 2.10.0+cu128, RTX 5090), so the routes do not call it")
def test_solve_triangular_at_large_batch():
    from spectra_rcwa.kernels.pivlu import lu_factor
    a = torch.randn(144, 578, 578, dtype=torch.complex64, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1))
    b = torch.randn(144, 578, 2, dtype=torch.complex64, device="cuda", generator=torch.Generator(device="cuda").manual_seed(2))
    with torch.no_grad():
        lu, perm = lu_factor(a)
        lo = torch.tril(lu, -1) + torch.eye(578, dtype=lu.dtype, device=lu.device)
        pb = b.gather(1, perm[:, :, None].expand(-1, -1, 2))
        x = torch.linalg.solve_triangular(torch.triu(lu), torch.linalg.solve_triangular(lo, pb, upper=False), upper=True)
    assert _batch_residual(a, x, b) < 0.5
