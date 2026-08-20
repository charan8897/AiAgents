#!/usr/bin/env python3
"""Trigger the tools CLI in a subprocess.

This is the subprocess integration step: instead of calling tool functions
in-process, we launch cli.py as a separate process and hand back its result.
The agent will later use this to invoke tools in an isolated process.

Usage:
    python tools/launcher.py <tool> [args...]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

CLI_PATH = Path(__file__).with_name("cli.py")


def trigger_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the tools CLI as a subprocess and return the completed result."""
    return subprocess.run(
        [sys.executable, str(CLI_PATH), *args],
        capture_output=True,
        text=True,
    )


def main(argv: list[str] | None = None) -> int:
    """Trigger the CLI with the given arguments and print its output."""
    args = list(sys.argv[1:] if argv is None else argv)

    result = trigger_cli(*args)

    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
