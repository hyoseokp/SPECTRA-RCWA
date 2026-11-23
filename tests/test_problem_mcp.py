"""Problems as JSON, the MCP tools and the command line: examples, energy, the Airy slab, p and s columns, symmetry, gradients."""
import asyncio
import json
import math

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
