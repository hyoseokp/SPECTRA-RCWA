"""Matrix functions against eigendecomposition / torch references, and the hand-written backward passes."""
import torch

from conftest import eig_gamma, layer_inputs, rel, tiny_density, tiny_forward
from spectra_rcwa import DEFAULT, REFERENCE_C128
from spectra_rcwa import matfun


def _layer_matrix():
    fwd = tiny_forward()
    return layer_inputs(fwd, tiny_density())


def test_gamma_matches_eig_branch():
    _, a, _ = _layer_matrix()
    ref = eig_gamma(a)
    assert rel(matfun.gamma(a, REFERENCE_C128), ref) < 1e-12
    assert rel(matfun.gamma(a, DEFAULT), ref) < 1e-6                      # 4 DB steps + tail, complex128
    g64 = matfun.gamma(a.to(torch.complex64), DEFAULT).to(torch.complex128)
    assert rel(g64, ref) < 1e-5


def test_cos_sinc_matches_eig():
    _, a, g = _layer_matrix()
    wh = g.omega[:, None, None] * 0.11
    e, sc, gc = matfun.cos_sinc(a, wh, DEFAULT)
    vals, x = torch.linalg.eig(a)
    xinv = torch.linalg.inv(x)
    u = wh[:, 0] * torch.sqrt(vals)
    fn = lambda f: (x * f[:, None, :]) @ xinv                              # noqa: E731
    assert rel(e, fn(torch.cos(u) - 1)) < 1e-12
    assert rel(sc, fn(torch.sin(u) / u)) < 1e-12
    assert rel(gc, fn((torch.cos(u) - 1) / u ** 2)) < 1e-12


def test_expm_matches_torch():
    gen = torch.Generator().manual_seed(0)
    x = 1.5 * torch.randn(4, 30, 30, dtype=torch.complex128, generator=gen)
    for cfg in (DEFAULT, REFERENCE_C128):
        assert rel(matfun.expm_taylor(x, cfg), torch.linalg.matrix_exp(x)) < 1e-12


def test_db_step_backward_matches_autograd():
    _, a, _ = _layer_matrix()
    x = (-1j * a[:2]).detach()
    gen = torch.Generator().manual_seed(3)
    for first in (True, False):
        m0 = x.clone().requires_grad_(True)
        y0 = m0 if first else (x + 0.1 * x.mH).clone().requires_grad_(True)
        leaves = (m0,) if first else (m0, y0)
        gm = torch.randn(x.shape, dtype=x.dtype, generator=gen)
        gy = torch.randn(x.shape, dtype=x.dtype, generator=gen)
        m1, y1 = matfun.DBStep.apply(m0, y0, True, 0.25, "torch", first)
        g_hand = torch.autograd.grad((m1, y1), leaves, (gm, gy))
        cfg = DEFAULT.replace(handwritten_backward=False, db_first_identity=first)
        m2, y2 = matfun.db_step(m0, y0, 0, cfg)
        g_auto = torch.autograd.grad((m2, y2), leaves, (gm, gy))
        assert rel(m1, m2) < 1e-13 and rel(y1, y2) < 1e-13
        for u, v in zip(g_hand, g_auto):
            assert rel(u, v) < 1e-11


def test_squarings_backward_matches_autograd():
    gen = torch.Generator().manual_seed(1)
    e = (0.3 * torch.randn(3, 20, 20, dtype=torch.complex128, generator=gen)).requires_grad_(True)
    g = torch.randn(3, 20, 20, dtype=torch.complex128, generator=gen)
    out = matfun.Squarings.apply(e, 4, False)
    ref = e
    for _ in range(4):
        ref = ref @ ref
    assert rel(out, ref) < 1e-13
    assert rel(torch.autograd.grad(out, e, g)[0], torch.autograd.grad(ref, e, g)[0]) < 1e-12


def test_block_inverse_cpu():
    from spectra_rcwa.kernels.blockinv import block_inverse, inv_block
    gen = torch.Generator().manual_seed(2)
    a = torch.randn(3, 200, 200, dtype=torch.complex128, generator=gen) + 20 * torch.eye(200)
    assert rel(block_inverse(a), torch.linalg.inv(a)) < 1e-12
    a.requires_grad_(True)
    g = torch.randn(3, 200, 200, dtype=torch.complex128, generator=gen)
    assert rel(torch.autograd.grad(inv_block(a), a, g)[0], torch.autograd.grad(torch.linalg.inv(a), a, g)[0]) < 1e-11
