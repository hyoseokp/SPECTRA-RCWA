# SPECTRA-RCWA

This repository is SPECTRA-RCWA, a PyTorch RCWA (Fourier modal method) solver for periodic layered structures that needs no eigendecomposition and returns adjoint gradients for inverse design. The full guide for coding agents is [AGENTS.md](../AGENTS.md): when the package fits, how to set up and check a problem, the conventions (time dependence, polarisations, order indexing), pitfalls and the test commands. Read it before changing code or answering questions about the solver.

Key points:
- Install with `pip install spectra-rcwa`; `pip install "spectra-rcwa[mcp]"` adds an MCP server (`spectra-rcwa mcp`).
- Problems can be given as JSON to the command line: `spectra-rcwa example lamellar_grating`, `spectra-rcwa check`, `spectra-rcwa solve`.
- Each layer mixes two isotropic materials through a density in [0, 1]; rectangular lattices only.
- Check convergence in the Fourier orders and energy conservation (T + R = 1 for lossless structures) before reporting numbers.
- Tests: `pytest` (CUDA tests are skipped without a GPU).
