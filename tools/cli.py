#!/usr/bin/env python3
"""CLI tool frontend — Layer 1: flags, config & normalized request.

This is the entry point that the subprocess launcher (tools/launcher.py) triggers.
It reads tools/cli.yaml, runs directory discovery, resolves all flags/config, and
outputs a normalized JSON request envelope to stdout for the next pipeline stage.

Flow:
    cli.py [objective] [--cwd DIR] [--max-steps N] ...
      → loads cli.yaml defaults
      → (optional) directory discovery prompt
      → merges CLI flags > YAML defaults
      → prints normalized request JSON → stdout

Usage:
    python tools/cli.py "Find all TODO comments" --cwd /project
    python tools/cli.py  # interactive (prompts for objective + directory)
    python tools/cli.py "Analyze code" --max-steps 20 --permission-mode auto
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path


CLI_YAML_PATH = Path(__file__).with_name("cli.yaml")


# ============================================================================
# Minimal YAML loader (zero external dependencies)
#
# Handles the subset of YAML used in cli.yaml:
#   - key: scalar value  (int, float, bool, str, null)
#   - nested sections via indentation
#   - literal blocks  (|)
#   - folded blocks   (>)
#   - comments        (#)
# ============================================================================

def _strip_inline_yaml_comment(raw: str) -> str:
    """Strip an inline YAML comment (``# ...``) that appears outside quotes."""
    in_quote: str | None = None
    for i, ch in enumerate(raw):
        if ch in ('"', "'") and in_quote is None:
            in_quote = ch
        elif ch == in_quote and in_quote is not None:
            in_quote = None
        elif ch == "#" and in_quote is None:
            return raw[:i].rstrip()
    return raw


def _parse_scalar(raw: str):
    """Convert a YAML scalar token to a Python value.

    Handles inline ``#`` comments by stripping everything from the first
    unquoted ``#`` to end-of-line.
    """
    v = _strip_inline_yaml_comment(raw)
    if not v:
        return ""
    if v.lower() in ("true", "yes", "on"):
        return True
    if v.lower() in ("false", "no", "off"):
        return False
    if v.lower() in ("null", "~", "none"):
        return None
    # Integer
    try:
        return int(v)
    except ValueError:
        pass
    # Float
    try:
        return float(v)
    except ValueError:
        pass
    # Quoted string
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
        return v[1:-1]
    return v


def load_cli_config(path: Path = CLI_YAML_PATH) -> dict:
    """Load cli.yaml into a nested Python dict.

    This is purpose-built for the specific YAML structure of cli.yaml and
    does NOT aim to be a general-purpose YAML parser.
    """
    if not path.exists():
        raise RuntimeError(f"CLI config not found: {path}")

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    root: dict = {}
    # Stack of (indent_level, parent_dict) — tracks nesting
    stack: list[tuple[int, dict]] = [(-1, root)]

    line_idx = 0
    while line_idx < len(lines):
        raw_line = lines[line_idx]
        line_idx += 1

        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip())

        # Pop stack to reach the correct parent for this indent level
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()

        if ":" not in stripped:
            continue

        colon = stripped.index(":")
        key = stripped[:colon].strip()
        rest = stripped[colon + 1:].strip()

        parent = stack[-1][1]

        # --- Empty value → sub-map ---
        if not rest:
            sub: dict = {}
            parent[key] = sub
            stack.append((indent, sub))

        # --- Literal block (|) → preserve newlines ---
        elif rest == "|":
            block_lines: list[str] = []
            while line_idx < len(lines):
                next_line = lines[line_idx]
                next_s = next_line.strip()
                next_indent = len(next_line) - len(next_line.lstrip())

                # Empty lines inside the block are preserved
                if not next_s:
                    block_lines.append("")
                    line_idx += 1
                    continue

                if next_s.startswith("#"):
                    line_idx += 1
                    continue

                # End of block: next line at or above the key's indent
                if next_indent <= indent:
                    break

                block_lines.append(next_s)
                line_idx += 1

            # Strip trailing blank lines
            while block_lines and block_lines[-1] == "":
                block_lines.pop()

            parent[key] = "\n".join(block_lines)

        # --- Folded block (>, >-) → join with spaces ---
        elif rest.startswith(">"):
            block_lines = []
            while line_idx < len(lines):
                next_line = lines[line_idx]
                next_s = next_line.strip()
                next_indent = len(next_line) - len(next_line.lstrip())

                if not next_s:
                    line_idx += 1
                    continue
                if next_s.startswith("#"):
                    line_idx += 1
                    continue
                if next_indent <= indent:
                    break

                block_lines.append(next_s)
                line_idx += 1

            parent[key] = " ".join(block_lines).strip()

        # --- Inline value ---
        else:
            parent[key] = _parse_scalar(rest)

    return root


# ============================================================================
# Argument parser
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with all available flags."""
    parser = argparse.ArgumentParser(
        prog="tools-cli",
        description="Agent tools CLI — Layer 1: flags, config & normalized request.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python tools/cli.py \"Find all TODO comments\"\n"
            "  python tools/cli.py \"Analyze code\" --cwd /home/user/my-project\n"
            "  python tools/cli.py --max-steps 20 --permission-mode auto\n"
            "  python tools/cli.py --dry-run \"Count lines of code\" --cwd .\n"
        ),
    )

    # Positional: the user's objective / task description
    parser.add_argument(
        "objective",
        nargs="?",
        default=None,
        help="The task objective (omit for interactive mode)",
    )

    # Working directory — if omitted, run discovery prompt
    parser.add_argument(
        "--cwd",
        type=str,
        default=None,
        help="Working directory (skip discovery prompt; default: current dir)",
    )

    # Limit overrides
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max execution steps (default: from cli.yaml)",
    )
    parser.add_argument(
        "--max-retries-per-step",
        type=int,
        default=None,
        help="Override max retries per step (default: from cli.yaml)",
    )
    parser.add_argument(
        "--max-replans",
        type=int,
        default=None,
        help="Override max replan cycles (default: from cli.yaml)",
    )
    parser.add_argument(
        "--timeout",
        type=str,
        default=None,
        help='Timeout strategy: "flexible" or an integer seconds (default: from cli.yaml)',
    )

    # Permission mode
    parser.add_argument(
        "--permission-mode",
        type=str,
        choices=["ask", "auto", "plan_approval"],
        default=None,
        help="Permission model for execution (default: from cli.yaml)",
    )

    # Output format
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["markdown", "json", "plain"],
        default=None,
        help="Desired output format (default: markdown)",
    )

    # Dry run — just print the normalized request, do NOT execute
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the normalized request JSON and exit",
    )

    return parser


# ============================================================================
# Directory discovery
# ============================================================================

def prompt_for_directory(prompt_text: str, default_cwd: str) -> str:
    """Run the interactive directory discovery prompt.

    Shows the prompt from cli.yaml, reads user input, and resolves
    to an absolute path. Falls back to *default_cwd* if the user
    presses Enter without typing.
    """
    print(f"\n{prompt_text}", file=sys.stderr, end="\n\n")
    try:
        raw = input("Directory path: ").strip()
    except (EOFError, KeyboardInterrupt):
        print(file=sys.stderr)
        raw = ""

    resolved = raw if raw else default_cwd
    resolved = os.path.expanduser(resolved)
    resolved = os.path.abspath(resolved)

    if not os.path.isdir(resolved):
        print(
            f"Warning: '{resolved}' is not a valid directory. Using current directory.",
            file=sys.stderr,
        )
        resolved = os.path.abspath(os.getcwd())

    return resolved


def discover_directory(config: dict, cli_cwd: str | None) -> str:
    """Resolve the working directory.

    Priority:
      1. CLI ``--cwd`` flag (if provided)
      2. Interactive prompt (if enabled in YAML)
      3. YAML default_cwd
      4. Current process working directory
    """
    if cli_cwd:
        resolved = os.path.expanduser(cli_cwd)
        resolved = os.path.abspath(resolved)
        if not os.path.isdir(resolved):
            print(
                f"Warning: '{resolved}' is not a valid directory. Using current directory.",
                file=sys.stderr,
            )
            resolved = os.path.abspath(os.getcwd())
        return resolved

    discovery_cfg = config.get("directory_discovery", {})
    if discovery_cfg.get("enabled", True):
        prompt_text = discovery_cfg.get("prompt", "Which directory should we work in?")
        default_cwd = discovery_cfg.get("default_cwd", "{cwd}")

        # Resolve default_cwd placeholder
        if default_cwd == "{cwd}":
            default_cwd = os.path.abspath(os.getcwd())

        return prompt_for_directory(prompt_text, default_cwd)

    # Discovery disabled — fall back to current directory
    return os.path.abspath(os.getcwd())


# ============================================================================
# Normalized request builder
# ============================================================================

def build_normalized_request(
    objective: str,
    cwd: str,
    config: dict,
    cli_args: argparse.Namespace,
) -> dict:
    """Build the normalized request JSON object.

    Merge priority:  CLI flags  >  YAML defaults  >  hardcoded fallbacks
    """
    # --- Limits ---
    defaults = config.get("limits", {})
    timeout_cfg = defaults.get("timeout", {})

    # Resolve timeout
    if cli_args.timeout is not None:
        try:
            timeout_val = int(cli_args.timeout)
        except ValueError:
            timeout_val = cli_args.timeout  # "flexible"
    else:
        timeout_val = timeout_cfg.get("strategy", "flexible")

    limits = {
        "max_steps": cli_args.max_steps if cli_args.max_steps is not None
                     else defaults.get("max_steps", 15),
        "max_retries_per_step": cli_args.max_retries_per_step if cli_args.max_retries_per_step is not None
                                else defaults.get("max_retries_per_step", 2),
        "max_replans": cli_args.max_replans if cli_args.max_replans is not None
                       else defaults.get("max_replans", 1),
        "timeout_seconds": timeout_val,
    }

    # --- Permission mode ---
    permission_mode = cli_args.permission_mode if cli_args.permission_mode is not None \
                      else config.get("permission_mode", {}).get("mode", "ask")

    # --- Output format ---
    output_format = cli_args.output_format if cli_args.output_format is not None \
                    else "markdown"

    # --- Generate request_id ---
    timestamp = int(time.time())
    short_id = uuid.uuid4().hex[:8]
    request_id = f"req_{timestamp}_{short_id}"

    # --- Assemble ---
    request = {
        "request_id": request_id,
        "objective": objective,
        "cwd": cwd,
        "output_format": output_format,
        "permission_mode": permission_mode,
        "limits": limits,
    }

    return request


def print_normalized_request(request: dict) -> None:
    """Print the normalized request JSON to stdout (for the orchestrator to capture)."""
    print(json.dumps(request, indent=2))


# ============================================================================
# Main entry point
# ============================================================================

def main(argv: list[str] | None = None) -> int:
    """Run the CLI tool frontend.

    Returns 0 on success, 1 on error.
    """
    # 1. Load YAML config
    try:
        config = load_cli_config()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # 2. Parse CLI flags
    parser = build_parser()
    args = parser.parse_args(argv)

    # 3. Directory discovery
    try:
        cwd = discover_directory(config, args.cwd)
    except Exception as exc:
        print(f"Directory discovery failed: {exc}", file=sys.stderr)
        return 1

    # 4. Get objective (prompt if interactive)
    objective = args.objective
    if not objective:
        # Interactive: prompt for the task
        try:
            objective = input("Task objective: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return 0

        if not objective:
            print("No objective provided. Exiting.", file=sys.stderr)
            return 0

    # 5. Build normalized request
    request = build_normalized_request(objective, cwd, config, args)

    # 6. Output
    if args.dry_run:
        print("[DRY RUN] Normalized request:", file=sys.stderr)
    print_normalized_request(request)

    # If not dry-run, the orchestrator (chat.py) captures this JSON and
    # proceeds to the evaluator↔executor loop.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())