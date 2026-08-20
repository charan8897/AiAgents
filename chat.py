#!/usr/bin/env python3
"""Minimal chat client for Google's Gemma model through the Gemini API.

This uses a Gemini Console / Google AI Studio API key.

Usage:
    python chat.py          # sends "hi"
    python chat.py "Hello" # sends a custom one-shot message
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# Direct Gemini API model id. In OpenRouter this may be shown as
# "google/gemma-4-31b-it:free", but Gemini API uses "gemma-4-31b-it".
MODEL = "gemma-4-31b-it"
ENV_PATH = Path(__file__).with_name(".env")


def load_env(path: Path = ENV_PATH) -> None:
    """Load simple KEY=VALUE lines from .env without external dependencies."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def get_gemini_api_key() -> str:
    """Return the Gemini API key, accepting common env var names."""
    load_env()

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        return api_key

    raise RuntimeError(
        f"GEMINI_API_KEY is missing. Create {ENV_PATH} in the same folder "
        "as chat.py and add: GEMINI_API_KEY=your_gemini_console_api_key"
    )


def ask(message: str) -> str:
    api_key = get_gemini_api_key()
    encoded_model = urllib.parse.quote(MODEL, safe="")
    encoded_key = urllib.parse.quote(api_key, safe="")
    api_url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/{encoded_model}:generateContent?key={encoded_key}"
    )

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": message}],
            }
        ]
    }

    request = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API request failed with HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach API: {exc.reason}") from exc

    try:
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(part.get("text", "") for part in parts).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected API response: {json.dumps(data, indent=2)}") from exc


def main() -> int:
    message = " ".join(sys.argv[1:]).strip() or "hi"
    try:
        print(ask(message))
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
