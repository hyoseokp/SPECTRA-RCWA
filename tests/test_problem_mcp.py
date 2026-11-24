"""Problems as JSON, the MCP tools and the command line: examples, energy, the Airy slab, p and s columns, symmetry, gradients."""
import asyncio
import json
import math
import time

import numpy as np
import pytest
import torch

from spectra_rcwa import cli
from spectra_rcwa import mcp_server as tools
from spectra_rcwa import problem as P


@pytest.fixture(autouse=True)
def results(tmp_path, monkeypatch):
    monkeypatch.setenv("SPECTRA_RCWA_RESULTS", str(tmp_path))
    return tmp_path


def cpu(problem, **kw):
    return dict(problem, device="cpu", precision="double", **kw)


@pytest.mark.parametrize("name", sorted(P.EXAMPLES))
def test_every_example_validates_and_conserves_energy(name):
    problem = cpu(P.example(name))
    report = P.check(problem)
    assert report["ok"] and not report["errors"]
    sol = P.solve(problem)
    t, r = sol["arrays"]["T"], sol["arrays"]["R"]
    if sol["checks"]["energy_error"] is not None:
        assert sol["checks"]["energy_error"] < 1e-11
    else:
        assert float((t.sum(2) + r.sum(2)).max()) < 1 - 1e-3          # the absorbing film takes some power


def test_slab_against_airy():
    def airy(wl):
        n1, n2, n3, d = 1.0, 2.0, 1.46, 0.37
        r12, r23, ph = (n1 - n2) / (n1 + n2), (n2 - n3) / (n2 + n3), np.exp(2j * np.pi * n2 * d / wl)
        return abs((r12 + r23 * ph ** 2) / (1 + r12 * r23 * ph ** 2)) ** 2

    exact = P.solve(cpu(P.example("slab"), rayleigh_nudge=0.0))["results"][0]
    assert abs(exact["total"]["R"]["p"] - airy(0.54)) < 1e-8 and abs(exact["total"]["R"]["s"] - airy(0.54)) < 1e-8   # pi as in torcwa
    nudged = P.solve(cpu(P.example("slab")))["results"][0]                       # the default solves at 540 nm / (1 + 1e-4)
    assert abs(nudged["total"]["R"]["p"] - airy(nudged["solved_at_nm"] / 1000)) < 1e-8 < abs(nudged["total"]["R"]["p"] - airy(0.54))


def test_p_and_s_are_the_solver_columns_in_the_plane_of_incidence():
    problem = cpu(P.example("lamellar_grating"))
    fwd = P._forward(problem)
    rho = P._tensors(problem, fwd)
    with torch.no_grad():
        t_ps, r_ps = P._efficiencies_ps(fwd, P._layers(problem, fwd, rho))
        t, r = fwd.efficiencies(P._layers(problem, fwd, rho))
    assert float((t_ps - t).abs().max()) < 1e-13 and float((r_ps - r).abs().max()) < 1e-13


def test_rotating_grating_and_plane_of_incidence_together_keeps_p_and_s():
    along_x = cpu(P.example("lamellar_grating"), fourier_orders=[6, 6], grid=[64, 64], period_um=[1.0, 1.0])
    along_y = json.loads(json.dumps(along_x))
    along_y["layers"][0]["pattern"]["axis"] = "y"
    along_y["incidence"] = [dict(theta_deg=10.0, phi_deg=90.0)]
    a, b = P.solve(along_x)["arrays"], P.solve(along_y)["arrays"]
    order = P.order_list((6, 6))
    swap = [order.index((n, m)) for m, n in order]
    assert np.abs(a["T"] - b["T"][:, :, swap]).max() < 1e-10 and np.abs(a["R"] - b["R"][:, :, swap]).max() < 1e-10


@pytest.mark.parametrize("name, order", [("lamellar_grating", (1, 0)), ("two_layer_grating", (-1, 0))])
def test_gradient_against_central_differences(name, order):
    out = P.gradient(cpu(P.example(name)), order=order, fd_check=True)
    fd = out["fd_check"]
    assert abs(fd["central_difference"] - fd["gradient_norm"]) < 1e-6 * fd["gradient_norm"]
    fixed = ["eps_design" in layer for layer in P.example(name)["layers"]]
    assert [g is None for g in out["gradients"]] == fixed


def test_check_reports_errors_without_raising():
    bad = cpu(P.example("lamellar_grating"), grid=[32, 1], eps_design=[4.0, -0.1])
    bad["incidence"] = [dict(theta_deg=95.0)]
    report = P.check(bad)
    text = " ".join(report["errors"])
    assert not report["ok"] and "too coarse" in text and "gain" in text and "theta" in text
    assert not P.check({"layers": []})["ok"]
    with pytest.raises(ValueError, match="too coarse"):
        P.solve(bad)


def test_metal_warns_about_convergence():
    metal = cpu(P.example("lamellar_grating"), eps_design=dict(n=0.2, k=3.5))
    assert any("metallic" in w for w in P.check(metal)["warnings"])


def test_patterns():
    grid, period = (8, 4), (1.0, 0.5)
    assert P.density(dict(type="lamellar", fill_factor=0.5, axis="x"), period, grid)[:, 0].tolist() == [0, 0, 1, 1, 1, 1, 0, 0]
    disk = P.density(dict(type="disk", radius_um=0.2), (1.0, 1.0), (8, 8))
    assert disk[4, 4] == 1 and disk[0, 0] == 0 and np.array_equal(disk, disk.T)
    assert P.density(dict(type="array", density=[0.5] * 8), period, (8, 1)).shape == (8, 1)
    with pytest.raises(ValueError):
        P.density(dict(type="hexagon"), period, grid)


def test_convergence_in_orders():
    out = tools.convergence_in_orders(cpu(P.example("lamellar_grating"), fourier_orders=[6, 0]), [[6, 0], [10, 0]])
    assert len(out["rows"]) == 2 and 0 < out["rows"][1]["max_change_from_previous"] < 0.1


@pytest.mark.parametrize("orders", [[1, 0], [2, 0], [3, 0], [8, 0], [1, 2]])
def test_default_orders_of_the_convergence_increase(orders):
    """1.3.0 went 2, 4, 4 and 1, 3, 2: the last change was 0 or taken backwards."""
    out = P.convergence(cpu(P.example("lamellar_grating"), fourier_orders=orders))
    used = [row["fourier_orders"] for row in out["rows"]]
    assert len(used) == 3 and used[0] == orders
    for a, b in zip(used, used[1:]):
        assert all((y > x) if x > 0 else y == 0 for x, y in zip(a, b))
    assert all(row["max_change_from_previous"] > 0 for row in out["rows"][1:])


def test_convergence_uses_one_grid():
    problem = cpu(P.example("lamellar_grating"), fourier_orders=[2, 0], grid=[9, 1])
    out = P.convergence(problem, [[2, 0], [4, 0], [6, 0]])
    assert out["grid"] == [25, 1]
    same = P.convergence(dict(problem, grid=[25, 1]), [[2, 0], [4, 0], [6, 0]])
    assert out["rows"] == same["rows"]
    assert P.convergence(dict(problem, fourier_orders=[4, 0]), [[2, 0], [4, 0]])["rows"][1]["max_change_from_previous"] > 0
    big = cpu(P.example("lamellar_grating"), fourier_orders=[80, 0])              # 2x needs more than the default 512 samples
    assert P.convergence(big)["grid"] == [641, 1]


def test_convergence_without_orders_reports_no_change():
    out = P.convergence(cpu(P.example("lamellar_grating"), fourier_orders=[0, 0], grid=[512, 1]))
    assert len(out["rows"]) == 1 and "max_change_from_previous" not in out["rows"][0]


def eff(problem):
    out = P.solve(problem)["arrays"]
    return np.concatenate([out["T"].ravel(), out["R"].ravel()])


def sampled(monkeypatch, problem, grid):
    with monkeypatch.context() as m:
        m.setattr(P, "_exact", lambda *a: False)
        return eff(dict(problem, grid=grid))


EDGES = [                                                      # edges halfway between the samples of a grid of 4 x odd: the sampled error is O(1 / grid^2)
    ([8, 0], dict(type="lamellar", fill_factor=0.5, center=0.375), [4 * 2001, 1]),
    ([0, 8], dict(type="lamellar", fill_factor=0.5, center=0.375, axis="y"), [1, 4 * 2001]),
    ([0, 2], dict(type="lamellar", fill_factor=0.5, center=0.375), [4 * 2001, 9]),
    ([4, 1], dict(type="lamellar", fill_factor=0.5, center=0.375), [4 * 2001, 5]),
    ([2, 2], dict(type="rectangle", size_um=[0.5, 0.25], center_um=[0.375, 0.75]), [4 * 257, 4 * 257]),
    ([2, 2], dict(type="disk", radius_um=0.3, center_um=[0.4, 0.55]), [2048, 2048])]


@pytest.mark.parametrize("orders, pattern, grid", EDGES)
@pytest.mark.parametrize("eps", [6.0, dict(n=0.5, k=2.0)])
def test_exact_coefficients_are_the_limit_of_the_samples(orders, pattern, grid, eps, monkeypatch):
    problem = cpu(P.example("lamellar_grating"), fourier_orders=orders, eps_design=eps, incidence=[dict(theta_deg=20.0, phi_deg=35.0)],
                  layers=[dict(thickness_um=0.3, pattern=pattern)])
    exact = eff(problem)
    assert np.abs(sampled(monkeypatch, problem, grid) - exact).max() < (1e-3 if pattern["type"] == "disk" else 5e-5)
    assert np.array_equal(eff(dict(problem, grid=[4 * o + 1 for o in orders])), exact)          # the grid plays no part


def test_exact_layers_next_to_sampled_layers_and_other_materials(monkeypatch):
    example = P.example("two_layer_grating")
    layers = [dict(thickness_um=0.2, pattern=dict(type="lamellar", fill_factor=0.5, center=0.375)), dict(thickness_um=0.1, pattern=dict(type="uniform", value=0.4)),
              dict(thickness_um=0.2, pattern=dict(type="lamellar", fill_factor=0.5, center=0.625), eps_design=3.0)]
    problem = cpu(example, layers=layers)
    assert np.abs(sampled(monkeypatch, problem, [4 * 2001, 1]) - eff(problem)).max() < 5e-5


@pytest.mark.parametrize("orders, shape", [([8, 0], (6, 1)), ([0, 8], (1, 6)), ([3, 3], (6, 5))])
def test_pixel_arrays(orders, shape):
    """Constant over pixels centred on the samples: the limit of a finer sampling, with a gradient with respect to the pixels."""
    rho = 0.1 + 0.8 * np.random.default_rng(0).random(shape)
    problem = cpu(P.example("lamellar_grating"), fourier_orders=orders, layers=[dict(thickness_um=0.3, pattern=dict(type="array", density=rho.tolist(), pixels=True))])
    fine = rho
    for axis in (0, 1):
        if shape[axis] > 1:
            fine = np.roll(np.repeat(fine, 151, axis), -75, axis)
    other = dict(problem, grid=list(fine.shape), layers=[dict(thickness_um=0.3, pattern=dict(type="array", density=fine))])
    assert np.abs(eff(other) - eff(problem)).max() < 1e-4
    g = P.gradient(problem, order=(0, 0), fd_check=True)
    assert g["gradients"][0].shape == shape
    assert math.isclose(g["fd_check"]["central_difference"], g["fd_check"]["gradient_norm"], rel_tol=1e-6)
    sol = P.solve(problem)["arrays"]
    assert math.isclose(g["value"], sol["T"][0, 0, len(sol["orders"]) // 2].mean(), rel_tol=1e-10)


def test_pixels_that_make_a_bar_are_a_lamellar_layer():
    bar = dict(type="lamellar", fill_factor=3 / 7, center=2 / 7)                                # pixels 1, 2 and 3 of 7, centred on 1/7, 2/7, 3/7
    pixels = dict(type="array", density=[0, 1, 1, 1, 0, 0, 0], pixels=True)
    a, b = (eff(cpu(P.example("lamellar_grating"), layers=[dict(thickness_um=0.5, pattern=p)])) for p in (bar, pixels))
    assert np.abs(a - b).max() < 1e-12


def test_pixel_array_needs_orders_along_both_axes():
    problem = cpu(P.example("lamellar_grating"), layers=[dict(thickness_um=0.3, pattern=dict(type="array", density=[[0, 1], [1, 0]], pixels=True))])
    assert "orders along both" in P.check(problem)["errors"][0]


def test_a_disk_in_a_one_dimensional_problem_warns():
    assert any("lamellar" in w for w in P.check(cpu(P.example("pillar_array"), fourier_orders=[5, 0]))["warnings"])


@pytest.mark.parametrize("start", [[1, 0], [8, 0]])
@pytest.mark.parametrize("eps", [4.0, 12.0])
def test_convergence_to_a_tolerance(start, eps):
    problem = cpu(P.example("lamellar_grating"), fourier_orders=start, eps_design=eps,
                  layers=[dict(thickness_um=0.5, pattern=dict(type="lamellar", fill_factor=0.3))])
    out = tools.convergence_in_orders(problem, tolerance=1e-3)
    assert out["converged"] and out["estimated_error"] <= 1e-3 and "reason" not in out
    ref = [200, 0]
    final, fine = (P._embed(P.solve(dict(problem, fourier_orders=o)), ref) for o in (out["fourier_orders"], ref))
    assert np.abs(final - fine).max() < 1e-3


def test_convergence_says_when_it_stops_short():
    problem = cpu(P.example("lamellar_grating"), fourier_orders=[4, 0], eps_design=12.0)
    out = P.convergence(problem, tolerance=1e-9, max_matrix_size=100)
    assert not out["converged"] and out["reason"] == "max_matrix_size reached" and out["fourier_orders"] == [21, 0]
    out = P.convergence(problem, tolerance=1e-9, max_seconds=0)
    assert not out["converged"] and out["reason"] == "max_seconds reached" and len(out["rows"]) == 1 and out["estimated_error"] is None
    out = P.convergence(dict(problem, fourier_orders=[0, 0]), tolerance=1e-3)
    assert not out["converged"] and out["reason"] == "no orders to refine"
    array = dict(problem, grid=[64, 1], layers=[dict(thickness_um=0.5, pattern=dict(type="array", density=[1.0] * 32 + [0.0] * 32))])
    out = P.convergence(array, tolerance=1e-9)
    assert not out["converged"] and "array" in out["reason"] and out["fourier_orders"] == [14, 0] and out["grid"] == [64, 1]
    with pytest.raises(ValueError):
        P.convergence(problem, [[2, 0]], tolerance=1e-3)


def test_convergence_does_not_extrapolate_the_start_up(monkeypatch):
    solve, calls = P.solve, []

    def slow_start(problem):
        calls.append(problem)
        if len(calls) == 1:
            time.sleep(0.5)
        return solve(problem)

    monkeypatch.setattr(P, "solve", slow_start)
    problem = cpu(P.example("lamellar_grating"), fourier_orders=[4, 0])
    assert P.convergence(problem, tolerance=1e-2, max_seconds=1.2)["converged"] and len(calls) > 2


def test_tools_save_arrays_and_gradients(results):
    out = tools.solve_efficiencies(cpu(tools.example_problem("lamellar_grating")), name="lam/1")
    saved = np.load(out["output"])
    assert out["output"].endswith("lam_1.npz") and saved["T"].shape == (1, 1, 41, 2)
    first = {tuple(row["order"]): row for row in out["results"][0]["transmitted"]}
    assert math.isclose(first[(1, 0)]["p"], float(saved["T"][0, 0, 21, 0]))
    grad = tools.efficiency_gradient(cpu(tools.example_problem("lamellar_grating")), order=[1, 0], polarisation="s")
    assert grad["layers"][0]["shape"] == [512, 1] and len(grad["layers"][0]["values"]) == 512
    assert np.allclose(np.load(grad["layers"][0]["file"]), grad["layers"][0]["values"])
    env = tools.environment()
    assert env["version"] and set(env["examples"]) == set(P.EXAMPLES)


def test_tools_register_with_the_mcp_server():
    pytest.importorskip("mcp")
    names = {tool.name for tool in asyncio.run(tools.build_server().list_tools())}
    assert names == {tool.__name__ for tool in tools.TOOLS}


def test_command_line(tmp_path, capsys):
    cli.main(["example", "slab"])
    problem = json.loads(capsys.readouterr().out)
    path = tmp_path / "slab.json"
    path.write_text(json.dumps(cpu(problem)))
    cli.main(["check", str(path)])
    assert json.loads(capsys.readouterr().out)["ok"]
    cli.main(["solve", str(path)])
    out = json.loads(capsys.readouterr().out)
    assert out["checks"]["energy_error"] < 1e-11 and out["results"][0]["transmitted"][0]["order"] == [0, 0]
