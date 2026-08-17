"""Entry point for ``python -m tactus.losses``.

Runs the full loss battery (every registered loss x every adversarial batch
scenario) and exits non-zero on the first failure, so it can be dropped into CI
or used as a pre-flight check before launching a 40-fold sweep.

    python -m tactus.losses                 # strict: raise on first failure
    python -m tactus.losses --no-strict     # run everything, tabulate failures
    python -m tactus.losses --names protonce clisa
"""

from __future__ import annotations

import argparse
import sys

from . import list_losses, selftest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tactus.losses")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help=f"subset of losses to test (default: all -> {list_losses()})",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="tabulate every failure instead of raising on the first",
    )
    args = parser.parse_args(argv)

    results = selftest(
        batch_size=args.batch_size,
        dim=args.dim,
        names=args.names,
        strict=not args.no_strict,
        verbose=True,
    )
    failed = [
        (name, tag)
        for name, scenarios in results.items()
        for tag, value in scenarios.items()
        if not isinstance(value, float)
    ]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
