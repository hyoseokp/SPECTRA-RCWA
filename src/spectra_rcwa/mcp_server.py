"""Model Context Protocol server: SPECTRA-RCWA as tools for AI agents.

Start it over stdio with ``spectra-rcwa mcp`` after ``pip install "spectra-rcwa[mcp]"``, or register that command in an
MCP client (Claude Code, Claude Desktop, Cursor, VS Code).  The tools are plain functions of this module, so they also work,
and are tested, without the MCP package.  Problems are JSON dictionaries (:mod:`spectra_rcwa.problem`); full arrays are
written under $SPECTRA_RCWA_RESULTS/mcp (default ./results/mcp).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import torch

from . import __version__
from . import problem as P

DOCS = "https://github.com/hyoseokp/spectra-rcwa"
CITATION = ("If results go into a paper, report or thesis, cite SPECTRA-RCWA: H. Park, arXiv:2611.30305 (2026), "
            "https://doi.org/10.48550/arXiv.2611.30305; the BibTeX entry is in the README section 'Citation'.")
CONVENTIONS = ("Lengths in um, wavelengths in nm, angles in degrees. Time dependence exp(-i w t): absorption is Im(eps) > 0. "
               "Light comes from the superstrate (eps_in, lossless). Order (m, n) has the in-plane wave vector "
               "k_inc + (m / period_x, n / period_y) x wavelength. p and s refer to the plane of incidence; at normal incidence "
               "p is E along x and s is E along y. Efficiencies are powers over the incident power.")
MAX_INLINE = 1024


def results_root() -> Path:
    root = Path(os.environ.get("SPECTRA_RCWA_RESULTS", "results")).resolve() / "mcp"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _path(name: str, suffix: str) -> Path:
    return results_root() / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', name)[:64] or 'run'}{suffix}"


def environment() -> dict:
    """SPECTRA-RCWA version, torch version, whether a CUDA GPU is available, example names, conventions, documentation and how to cite."""
    cuda = torch.cuda.is_available()
    return dict(version=__version__, torch=torch.__version__, cuda_available=cuda,
                gpu=torch.cuda.get_device_name(0) if cuda else None,
                single_precision_cuda_kernels=P.cuda_kernels_available(),
                default_device="cuda" if P.cuda_kernels_available() else "cpu",
                examples={k: v["description"] for k, v in P.EXAMPLES.items()},
                conventions=CONVENTIONS, documentation=DOCS, agent_guide=f"{DOCS}/blob/main/AGENTS.md", citation=CITATION)


def example_problem(name: str) -> dict:
    """A complete problem to edit into the user's structure. Names: lamellar_grating, pillar_array, slab, two_layer_grating."""
    return P.example(name)


def problem_schema() -> dict:
    """JSON schema of a problem: lattice, orders, wavelengths, incidence, media, layers and their patterns, with units."""
    return P.SCHEMA


def validate_problem(problem: dict) -> dict:
    """Check a problem before solving it: errors, warnings (metals, Rayleigh anomalies, size on the CPU), orders, matrix size, solves."""
    return P.check(problem)


def solve_efficiencies(problem: dict, name: str = "run", threshold: float = 1e-6) -> dict:
    """Diffraction efficiencies of every order above ``threshold``, transmitted and reflected, for p and s incidence, per wavelength and
    direction, with propagation angles, totals, absorption and the energy check. Full arrays are saved as NPZ (path in ``output``)."""
    sol = P.solve(problem, threshold)
    path = _path(name, ".npz")
    np.savez(path, **sol["arrays"])
    return dict(output=str(path), results=sol["results"], checks=sol["checks"], report=sol["report"], citation=CITATION)


def efficiency_gradient(problem: dict, order: list[int] | None = None, side: str = "T", polarisation: str = "unpolarised",
                        fd_check: bool = False, name: str = "gradient") -> dict:
    """Mean over the problem's wavelengths and directions of one order's efficiency (side 'T' or 'R', polarisation 'p', 's' or
    'unpolarised'), and its gradient with respect to the density of every layer (adjoint, one backward pass). Gradients are saved as
    NPY files; small ones are also returned inline. fd_check compares a central difference along the gradient with its norm."""
    out = P.gradient(problem, tuple(order or (0, 0)), side, polarisation, fd_check)
    layers = []
    for i, g in enumerate(out.pop("gradients")):
        if g is None:
            layers.append(dict(layer=i, gradient=None, note="this layer has its own materials and is held fixed"))
            continue
        path = _path(f"{name}_layer{i}", ".npy")
        np.save(path, g)
        row = dict(layer=i, file=str(path), shape=list(g.shape), norm=float(np.linalg.norm(g)), min=float(g.min()), max=float(g.max()))
        if g.size <= MAX_INLINE:
            row["values"] = g.tolist()
        layers.append(row)
    out["layers"] = layers
    return out


def convergence_in_orders(problem: dict, orders_list: list[list[int]] | None = None, tolerance: float | None = None,
                          max_matrix_size: int = 2000, max_seconds: float = 300.0) -> dict:
    """Solve at increasing Fourier orders on one grid and report how much the efficiencies change. With ``tolerance`` (for example 1e-3) the
    orders grow until the last two changes and the estimated error are below it: read ``converged``, ``reason`` and ``fourier_orders``.
    Without it: orders_list, or the problem's orders, 1.5x and 2x."""
    return P.convergence(problem, orders_list, tolerance, max_matrix_size, max_seconds)


TOOLS = (environment, example_problem, problem_schema, validate_problem, solve_efficiencies, efficiency_gradient, convergence_in_orders)

INSTRUCTIONS = ("SPECTRA-RCWA solves periodic layered structures (diffraction gratings, metagratings, metasurfaces, photonic-crystal slabs, "
                "thin films) with rigorous coupled-wave analysis (RCWA, Fourier modal method), and gives gradients for inverse design. "
                "Start from example_problem, edit the lattice, materials and layers, call validate_problem, then solve_efficiencies. "
                "Check the result with convergence_in_orders before quoting numbers; a lossless stack has energy_error near 1e-12. "
                "Use efficiency_gradient for sensitivities and optimisation. " + CONVENTIONS + " " + CITATION)


def build_server():
    try:
        from mcp.server.mcpserver import MCPServer as Server     # mcp 2
    except ImportError:
        from mcp.server.fastmcp import FastMCP as Server         # mcp 1
    server = Server("spectra-rcwa", instructions=INSTRUCTIONS)
    for tool in TOOLS:
        server.tool()(tool)
    return server


def main():
    try:
        server = build_server()
    except ImportError as exc:
        raise SystemExit('The MCP server needs the optional dependency: pip install "spectra-rcwa[mcp]"') from exc
    server.run()


if __name__ == "__main__":
    main()
