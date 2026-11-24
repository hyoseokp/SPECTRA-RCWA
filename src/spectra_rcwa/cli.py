"""Command line: ``spectra-rcwa info | example NAME | check FILE | solve FILE | mcp``."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog="spectra-rcwa", description="RCWA (Fourier modal method) solver without eigendecomposition, with adjoint gradients.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("info", help="version, torch, GPU, examples and conventions as JSON")
    ex = sub.add_parser("example", help="print an example problem as JSON")
    ex.add_argument("name", help="lamellar_grating, pillar_array, hexagonal_pillars, slab or two_layer_grating")
    for name, text in (("check", "validate a problem JSON file"), ("solve", "diffraction efficiencies of a problem JSON file")):
        p = sub.add_parser(name, help=text)
        p.add_argument("file", help="problem JSON ('-' for standard input)")
    sub.add_parser("mcp", help="serve the MCP tools over stdio (needs spectra-rcwa[mcp])")
    args = parser.parse_args(argv)

    if args.command == "mcp":
        from .mcp_server import main as serve
        serve()
        return
    from . import mcp_server, problem as P
    if args.command == "info":
        out = mcp_server.environment()
    elif args.command == "example":
        out = P.example(args.name)
    else:
        text = sys.stdin.read() if args.file == "-" else open(args.file, encoding="utf-8").read()
        problem = json.loads(text)
        out = P.check(problem) if args.command == "check" else {k: v for k, v in P.solve(problem).items() if k != "arrays"}
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
