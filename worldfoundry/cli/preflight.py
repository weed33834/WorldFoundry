"""CLI registration for optional runtime preflight commands."""

from __future__ import annotations

import argparse


def register_preflight_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "preflight",
        help=argparse.SUPPRESS,
    )
    preflight_subparsers = parser.add_subparsers(dest="preflight_command", required=True)
    runtime_parser = preflight_subparsers.add_parser(
        "runtime",
        help=argparse.SUPPRESS,
    )
    runtime_parser.set_defaults(func=_runtime_preflight_removed)


def _runtime_preflight_removed(args: argparse.Namespace) -> int:
    del args
    print("runtime preflight checks are not part of the in-tree benchmark execution path")
    return 2
