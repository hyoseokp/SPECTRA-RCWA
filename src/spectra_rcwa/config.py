"""Solver configuration.

The defaults (DEFAULT) are the configuration used for all benchmarks: complex64 with a value
error below 1e-5.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class SolverConfig:
    """Options of the eigendecomposition-free layer solve.

    Matrix square root (scaled Denman-Beavers, product form):
        db_iters:            number of DB iterations (each costs one batched inverse).
        db_scaled:           how many of them use norm scaling (the first ``db_scaled``).
        db_scale_power:      scaling mu = (||M^-1||_F / ||M||_F) ** db_scale_power.  0.25 balances
                             mu^2 M against mu^-2 M^-1 in the M update; 0.5 is the classical choice.
        db_first_identity:   use Y0 M0^-1 = I exactly in the first step (saves one GEMM and removes
                             a cancelling term from the derivative).
        tail_order:          order of the M^(-1/2) series (M close to I) that finishes the
                             iteration; 0 disables it.
    Matrix exponential (scaling and squaring):
        taylor_degree:       Taylor degree, evaluated by Paterson-Stockmeyer.
        exp_theta:           scale so that ||X / 2^s||_1 <= exp_theta.
    Route through the layer:
        boundary:            "entire" (default): cos(w h A^(1/2)) and sin(w h A^(1/2)) / A^(1/2), entire in A, no square
                             root, no branch, nothing singular at an order at cutoff.  The other two go through
                             Gamma = A^(1/2) and E, whose value and gradient are not reliable at a cutoff with the
                             iterations of DEFAULT: "direct" (two inverses, hand-written adjoint) and "smatrix" (layer
                             S-matrix and Redheffer products with linear solves; reference with REFERENCE_C128).
        max_step_phase:      "entire": a layer is cut in steps with w h max(k) at most this (largest |C|, |S| about
                             cosh of it).
        cos_degree, cos_theta: "entire": polynomial degree in A of C and S, and the scale 4^s with max(||z^2||_1^(1/2), ||z^3||_1^(1/3)) / 4^s <= cos_theta, z = (w h)^2 A.
        block_from_steps:    "entire": a layer that would need this many steps goes through its S-matrix, built by doubling
                             (:mod:`spectra_rcwa.blocks`), instead of the admittance recursion.
        adjoint_nodes, adjoint_tol: "entire", complex64: Gauss-Legendre nodes (even) per step of a layer with w h max(k, |eps|^(1/2)) up to 4.5 (in proportion beyond it), and the size of the first
                             term left out of the Taylor series between the planes, for the gradient (:mod:`spectra_rcwa.adjoint`).
    CUDA / complex64 implementation choices (ignored on CPU or for complex128):
        first_inverse:       "pivlu" (in-house sync-free pivoted LU) or "torch" (torch.linalg.inv).
        block_inverse_from:  DB iterations k >= this use the unpivoted block-recursive inverse
                             (all GEMMs).  The early iterates (M close to A) need pivoting.
        boundary_inverse:    "block", "pivlu" or "torch" for the boundary solve of the routes through Gamma.
        layer_inverse:       "entire": inverse of a thin step and in the doubling of a block, and below a block.
        step_inverse:        "entire": inverse of a step of the admittance recursion; "permuted": the rows of partial pivoting, then the block recursion.  An
                             unpivoted inverse here gives a gradient wrong by 2.5e-3 on an 8-layer design at 420 nm.
        top_inverse:         "entire": what needs pivoting, V_i + Y and I - rho_i rho_top at the top of the stack (LU factors, two right-hand sides) and Y_0 + Y below a block.
        fused_kernels:       NVRTC kernels for linear combinations, the K_i / K_o products and the thin step.
        handwritten_backward: "entire": the gradient from one adjoint field (:mod:`spectra_rcwa.adjoint`), no graph, any dtype; False: autograd.  Routes through Gamma, complex64
                             on CUDA: hand-written backward for the DB steps and the exp squarings.
        tf32_squaring_backward: run the backward of the exp squarings on TF32 tensor cores.
        linalg_library:      torch.backends.cuda.preferred_linalg_library for remaining torch.linalg calls.
    """

    db_iters: int = 4
    db_scaled: int = 4
    db_scale_power: float = 0.25
    db_first_identity: bool = True
    tail_order: int = 3
    taylor_degree: int = 18
    exp_theta: float = 4.0
    boundary: str = "entire"
    max_step_phase: float = 4.5
    cos_degree: int = 6
    cos_theta: float = 6.0
    block_from_steps: int = 7
    adjoint_nodes: int = 6
    adjoint_tol: float = 1e-5
    first_inverse: str = "pivlu"
    block_inverse_from: int = 1
    boundary_inverse: str = "pivlu"
    layer_inverse: str = "block"
    step_inverse: str = "permuted"
    top_inverse: str = "pivlu"
    fused_kernels: bool = True
    handwritten_backward: bool = True
    tf32_squaring_backward: bool = True
    linalg_library: str = "magma"

    def __post_init__(self):
        if {self.step_inverse, self.layer_inverse, self.top_inverse} - {"block", "pivlu", "torch", "permuted"}:
            raise ValueError("unknown inverse")
        if self.db_iters < 1 or not 0 <= self.db_scaled <= self.db_iters:
            raise ValueError("need db_iters >= 1 and 0 <= db_scaled <= db_iters")
        if not 0 <= self.tail_order <= 4:
            raise ValueError("tail_order must lie in 0..4")
        if self.adjoint_nodes < 2 or self.adjoint_nodes % 2:
            raise ValueError("adjoint_nodes must be even and at least 2")
        if min(self.max_step_phase, self.cos_theta, self.adjoint_tol) <= 0 or self.cos_degree < 1 or self.block_from_steps < 1:
            raise ValueError("max_step_phase, cos_theta and adjoint_tol must be positive, cos_degree and block_from_steps at least 1")
        if self.boundary not in ("entire", "direct", "smatrix"):
            raise ValueError("boundary must be 'entire', 'direct' or 'smatrix'")
        if self.first_inverse not in ("pivlu", "torch"):
            raise ValueError("first_inverse must be 'pivlu' or 'torch'")
        if self.boundary_inverse not in ("block", "pivlu", "torch"):
            raise ValueError("boundary_inverse must be 'block', 'pivlu' or 'torch'")

    def replace(self, **kw) -> "SolverConfig":
        return replace(self, **kw)


DEFAULT = SolverConfig()

# The default route as it was before the adjoint field and the kernels (commit 777f630), for comparison: autograd through dense P and Q, pivoted inverses, a higher polynomial degree.
AUTOGRAD_ROUTE = DEFAULT.replace(handwritten_backward=False, cos_degree=8, cos_theta=1.0, layer_inverse="pivlu", step_inverse="pivlu")

# Converged reference for complex128 (used for accuracy checks; matches central finite differences).
REFERENCE_C128 = SolverConfig(db_iters=14, db_scaled=14, db_scale_power=0.5, db_first_identity=False, tail_order=0,
                              exp_theta=2.0, boundary="smatrix", first_inverse="torch", block_inverse_from=99, boundary_inverse="torch",
                              fused_kernels=False, handwritten_backward=False, tf32_squaring_backward=False)
