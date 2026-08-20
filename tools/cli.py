#!/usr/bin/env python3
"""Command line interface for the agent tools (design in progress).

This is the entry point that the subprocess launcher (launcher.py) triggers.
Individual tools and subcommands will be added here step by step.

Usage:
    python tools/cli.py <tool> [args...]
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="tools-cli",
        description="Agent tools command line interface.",
    )
    parser.add_argument(
        "tool",
        nargs="?",
        help="tool to run (tools get added as we design them)",
    )
    parser.add_argument(
        "args",
        nargs="*",
        help="arguments passed to the selected tool",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to a tool (placeholder for now)."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.tool is None:
        parser.print_help()
        return 0

    # TODO: dispatch to real tools in a later design step.
    print(f"tools-cli: tool '{args.tool}' is not implemented yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
