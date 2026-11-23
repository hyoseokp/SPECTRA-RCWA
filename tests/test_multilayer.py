"""Multi-layer stack: one layer equals the direct boundary solve, two half layers equal one layer, and the
autograd gradient agrees with central finite differences (complex128)."""
import dataclasses

import torch

from conftest import rel, tiny_density, tiny_stack
from spectra_rcwa import DEFAULT, REFERENCE_C128, ColorRouterForward, shifted_disk_pupil
from spectra_rcwa.multilayer import MultiLayerForward


def _make(cls, config=REFERENCE_C128, dtype=torch.complex128, **kw):
    rays = shifted_disk_pupil(numerical_aperture=0.3, chief_ray_angle_deg=2.5, chief_ray_azimuth_deg=22.5,
                              radial_order=1, azimuth_count=2)
    return cls(stack=tiny_stack(), rays=rays, wavelengths_nm=(450.0, 540.0, 640.0), device="cpu", dtype=dtype,
               config=config, **kw)


def test_one_layer_equals_direct():
    rho = tiny_density()
    direct = _make(ColorRouterForward, REFERENCE_C128.replace(boundary="direct")).response(rho)
    stacked = _make(MultiLayerForward, thicknesses_um=(0.6,)).response(rho[None])
    assert rel(stacked, direct) < 1e-11
    s64 = _make(MultiLayerForward, DEFAULT, torch.complex64, thicknesses_um=(0.6,)).response(rho[None].float())
    assert rel(s64, direct) < 1e-4


def test_two_half_layers_equal_one_layer():
    rho = tiny_density()
    for cfg in (REFERENCE_C128, DEFAULT):
        one = _make(MultiLayerForward, cfg, thicknesses_um=(0.6,)).response(rho[None])
        two = _make(MultiLayerForward, cfg, thicknesses_um=(0.3, 0.3)).response(torch.stack((rho, rho)))
        assert rel(two, one) < 1e-10


def test_block_route_equals_the_recursion():
    rho = tiny_density()
    both = torch.stack((tiny_density(grey=False).flip(0) * 0.7 + 0.15, rho))
    ref = _make(MultiLayerForward, DEFAULT, thicknesses_um=(0.3, 0.4)).response(both)
    assert rel(_make(MultiLayerForward, DEFAULT.replace(block_from_steps=1), thicknesses_um=(0.3, 0.4)).response(both), ref) < 1e-11
    assert rel(_make(ColorRouterForward, DEFAULT.replace(block_from_steps=1)).response(rho), _make(ColorRouterForward, DEFAULT).response(rho)) < 1e-11


def test_stored_block_equals_the_recursion():
    rho = tiny_density()
    fwd = _make(MultiLayerForward, DEFAULT, thicknesses_um=(0.3, 0.4))
    top = tiny_density(grey=False).flip(0) * 0.7 + 0.15
    stored = fwd.layer_block(rho, 0.4)
    field = fwd.stack_field([(top, 0.3), stored])
    g = fwd._solves(0, None)
    assert rel(fwd._aggregate(fwd._wells(field, g)[0]), fwd.response(torch.stack((top, rho)))) < 1e-11
    assert rel(fwd.stack_field([(top, 0.3), (rho, 0.4)]), field) < 1e-11
    some = fwd.stack_field([(top, 0.3), fwd.layer_block(rho, 0.4, solves=[1, 4])], solves=[1, 4])
    assert rel(some, field[[1, 4]]) < 1e-11


def _flux(a, y):
    h = y.left(a)
    n = a.shape[1] // 2
    return (a[:, :n] * h[:, n:].conj() - a[:, n:] * h[:, :n].conj()).real.sum(1)


def test_energy_balance_and_absorption():
    rho = tiny_density()
    fwd = _make(ColorRouterForward, DEFAULT)
    g = fwd._solves(0, None)
    below, top = fwd.stack_field([(rho, 0.6)], return_top=True)
    inc = _flux(g.e_in, g.vi)
    t, r = _flux(below, g.vo) / inc, _flux(top - g.e_in, g.vi) / inc
    assert float((t + r - 1).abs().max()) < 1e-11
    st = dataclasses.replace(tiny_stack(), eps_design=(4.14 + 0.034j) ** 2, eps_fill=2.13, eps_in=2.13, eps_out=2.13, thickness_um=1.5)
    res = []
    for cfg in (DEFAULT.replace(block_from_steps=10 ** 6), DEFAULT.replace(block_from_steps=1)):
        f = ColorRouterForward(stack=st, rays=fwd.rays, wavelengths_nm=(540.0,), device="cpu", dtype=torch.complex128, config=cfg)
        gg = f._solves(0, None)
        below, top = f.stack_field([(rho, 1.5)], return_top=True)
        res.append((_flux(below, gg.vo) / _flux(gg.e_in, gg.vi), _flux(top - gg.e_in, gg.vi) / _flux(gg.e_in, gg.vi)))
    assert rel(res[0][0], res[1][0]) < 1e-10 and rel(res[0][1], res[1][1]) < 1e-10
    assert float((1 - res[0][0] - res[0][1]).min()) > 0.01


def test_gamma_route_of_a_stack_equals_the_default():
    rho = tiny_density()
    both = torch.stack((tiny_density(grey=False).flip(0) * 0.7 + 0.15, rho))
    ref = _make(MultiLayerForward, thicknesses_um=(0.3, 0.4)).response(both)
    assert rel(_make(MultiLayerForward, DEFAULT, thicknesses_um=(0.3, 0.4)).response(both), ref) < 1e-11


def test_multilayer_gradient_central_differences():
    for cfg in (REFERENCE_C128, DEFAULT):
        fwd = _make(MultiLayerForward, cfg, thicknesses_um=(0.3, 0.4))
        w = torch.linspace(-1.0, 1.0, 12, dtype=torch.float64).reshape(4, 3)
        obj = lambda r: (fwd.response(r) * w).sum()                 # noqa: E731
        rho = torch.stack((tiny_density(), tiny_density(grey=False).flip(0) * 0.7 + 0.15)).requires_grad_(True)
        grad = torch.autograd.grad(obj(rho), rho)[0]
        v = torch.randn(rho.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(9))
        v = v / v.norm()
        h = 1e-5
        with torch.no_grad():
            fd = float((obj(rho + h * v) - obj(rho - h * v)) / (2 * h))
        assert abs(float((grad * v).sum()) - fd) < 1e-7 * abs(fd) + 1e-10
