"""Li's rules for the in-plane permittivity matrices: TM convergence of a 1D grating, and the gradient with E_xx != E_yy."""
import torch

from conftest import tiny_density, tiny_forward
from spectra_rcwa import AUTOGRAD_ROUTE, DEFAULT
from spectra_rcwa import problem as P


def _lamellar(m):
    return dict(period_um=[0.8, 0.8], fourier_orders=[m, 0], wavelengths_nm=[550.0], incidence=[dict(theta_deg=20.0, phi_deg=0.0)],
                eps_in=1.0, eps_out=1.45 ** 2, eps_design=1.5 ** 2, eps_fill=1.0, rayleigh_nudge=0.0, device="cpu", precision="double",
                grid=[1024, 1], layers=[dict(thickness_um=0.4, pattern=dict(type="lamellar", fill_factor=0.5, axis="x"))])


def _tm(m):
    return {tuple(o["order"]): o["p"] for o in P.solve(_lamellar(m))["results"][0]["transmitted"]}


def test_tm_of_a_lamellar_grating_converges_fast():
    """With the Laurent rule TM moves by 3e-4 from 80 to 160 orders; with Li's rule 20 orders are within 1e-4 of 160."""
    coarse, fine = _tm(20), _tm(160)
    assert max(abs(coarse[k] - fine[k]) for k in fine) < 1e-4


def test_adjoint_gradient_with_li_matches_autograd():
    rho = tiny_density()
    w = torch.randn(4, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    grads = []
    for cfg in (DEFAULT, AUTOGRAD_ROUTE):
        fwd = tiny_forward(config=cfg.replace(factorization="li"))
        r = rho.clone().requires_grad_(True)
        grads.append(torch.autograd.grad((fwd.response(r) * w).sum(), r)[0])
    assert float((grads[0] - grads[1]).norm() / grads[1].norm()) < 1e-10
