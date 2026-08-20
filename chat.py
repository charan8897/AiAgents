#!/usr/bin/env python3
"""Multi-agent Gemma chat client through the Gemini API.

Workflow:
    user query -> evaluator (gemma-4-31b-it) reads intent_prompt.yaml
    -> streams its intent analysis in real time
    -> responder (gemma-4-26b-a4b-it) produces the final answer

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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Direct Gemini API model ids.
# In OpenRouter these may be shown with provider/free suffixes, but Gemini API
# uses direct model ids like "gemma-4-31b-it" and "gemma-4-26b-a4b-it".
RESPONDER_MODEL = "gemma-4-26b-a4b-it"
EVALUATOR_MODEL = "gemma-4-31b-it"

ENV_PATH = Path(__file__).with_name(".env")
INTENT_PROMPT_PATH = Path(__file__).with_name("intent_prompt.yaml")

# ANSI styling for evaluator-only terminal output. Terminals cannot render true
# transparency, so this uses dim gray text to make evaluator thinking feel muted.
ANSI_DIM_GRAY = "\033[2;90m"
ANSI_RESET = "\033[0m"
_WINDOWS_ANSI_ENABLED: bool | None = None


@dataclass(frozen=True)
class IntentPromptTemplate:
    """Raw YAML prompt template plus the parsed one-line prompt instruction."""

    raw_yaml: str
    prompt: str


def enable_windows_ansi() -> bool:
    """Enable ANSI escape handling for Windows terminals when possible."""
    global _WINDOWS_ANSI_ENABLED

    if os.name != "nt":
        return True
    if _WINDOWS_ANSI_ENABLED is not None:
        return _WINDOWS_ANSI_ENABLED

    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        stderr_handle = kernel32.GetStdHandle(-12)
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(stderr_handle, ctypes.byref(mode)):
            _WINDOWS_ANSI_ENABLED = False
            return False

        enable_virtual_terminal_processing = 0x0004
        new_mode = mode.value | enable_virtual_terminal_processing
        _WINDOWS_ANSI_ENABLED = bool(kernel32.SetConsoleMode(stderr_handle, new_mode))
        return _WINDOWS_ANSI_ENABLED
    except Exception:
        _WINDOWS_ANSI_ENABLED = False
        return False


def evaluator_color_enabled() -> bool:
    """Return True when evaluator output should use ANSI dim-gray styling."""
    if os.getenv("NO_COLOR") is not None or os.getenv("CHAT_EVALUATOR_COLOR") == "0":
        return False
    if not sys.stderr.isatty():
        return False
    if os.name != "nt" and os.getenv("TERM") == "dumb":
        return False
    return enable_windows_ansi()


def start_evaluator_style() -> bool:
    """Start muted evaluator styling; return whether styling was enabled."""
    if not evaluator_color_enabled():
        return False
    print(ANSI_DIM_GRAY, end="", file=sys.stderr, flush=True)
    return True


def stop_evaluator_style(enabled: bool) -> None:
    """Stop muted evaluator styling if it was enabled."""
    if enabled:
        print(ANSI_RESET, end="", file=sys.stderr, flush=True)


def print_evaluator(text: str, *, end: str = "", flush: bool = True) -> None:
    """Print evaluator text to stderr only."""
    print(text, end=end, file=sys.stderr, flush=flush)


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


def load_intent_prompt_template(path: Path = INTENT_PROMPT_PATH) -> IntentPromptTemplate:
    """Load the YAML template and parse its one-line `prompt` entry.

    This intentionally avoids a PyYAML dependency because the template is small,
    but it still passes the raw YAML text to the evaluator so the model sees the
    actual file contents instead of only a detached string value.
    """
    if not path.exists():
        raise RuntimeError(f"Intent prompt template is missing: {path}")

    raw_yaml = path.read_text(encoding="utf-8").strip()
    if not raw_yaml:
        raise RuntimeError(f"Intent prompt template is empty: {path}")

    for raw_line in raw_yaml.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.strip() == "prompt":
            prompt = value.strip().strip('"').strip("'")
            if not prompt:
                raise RuntimeError(f"The 'prompt' entry in {path} is empty")
            return IntentPromptTemplate(raw_yaml=raw_yaml, prompt=prompt)

    raise RuntimeError(f"No 'prompt' entry found in {path}")


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


def build_api_url(model: str, *, stream: bool = False) -> str:
    """Build the Gemini REST API URL for normal or streaming generation."""
    api_key = get_gemini_api_key()
    encoded_model = urllib.parse.quote(model, safe="")
    encoded_key = urllib.parse.quote(api_key, safe="")
    action = "streamGenerateContent" if stream else "generateContent"
    api_url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/{encoded_model}:{action}?key={encoded_key}"
    )
    if stream:
        api_url += "&alt=sse"
    return api_url


def build_payload(prompt: str) -> dict[str, object]:
    """Build a Gemini generateContent request payload."""
    return {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ]
    }


def extract_text(data: dict[str, object]) -> str:
    """Extract text from a Gemini generateContent response or stream chunk."""
    try:
        candidates = data.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            return ""
        content = candidates[0].get("content")
        if not isinstance(content, dict):
            return ""
        parts = content.get("parts")
        if not isinstance(parts, list):
            return ""
        return "".join(
            part.get("text", "") for part in parts if isinstance(part, dict)
        )
    except AttributeError:
        return ""


def generate_content(model: str, prompt: str) -> str:
    """Call the Gemini generateContent endpoint for a single model/prompt."""
    request = urllib.request.Request(
        build_api_url(model),
        data=json.dumps(build_payload(prompt)).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"{model} API request failed with HTTP {exc.code}: {body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach {model} API: {exc.reason}") from exc

    text = extract_text(data).strip()
    if not text:
        raise RuntimeError(f"{model} returned an empty response: {json.dumps(data, indent=2)}")
    return text


def stream_generate_content(
    model: str,
    prompt: str,
    on_text: Callable[[str], None] | None = None,
) -> str:
    """Stream Gemini output in real time and return the complete response text."""
    request = urllib.request.Request(
        build_api_url(model, stream=True),
        data=json.dumps(build_payload(prompt)).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    chunks: list[str] = []

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue

                event_data = line.removeprefix("data:").strip()
                if event_data == "[DONE]":
                    break

                try:
                    data = json.loads(event_data)
                except json.JSONDecodeError:
                    continue

                text = extract_text(data)
                if not text:
                    continue

                chunks.append(text)
                if on_text:
                    on_text(text)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"{model} streaming API request failed with HTTP {exc.code}: {body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach {model} streaming API: {exc.reason}") from exc

    text = "".join(chunks).strip()
    if not text:
        raise RuntimeError(f"{model} returned an empty streaming response")
    return text


def stream_to_stderr(text: str) -> None:
    """Print evaluator tokens immediately without mixing them into final stdout."""
    print_evaluator(text)


def evaluate_intent(user_query: str, intent_template: IntentPromptTemplate) -> str:
    """Stream 31B evaluator intent analysis using the YAML prompt."""
    evaluator_input = f"""You are the evaluator agent in a two-agent workflow.

Read this exact YAML prompt template loaded from intent_prompt.yaml:
```yaml
{intent_template.raw_yaml}
```

Parsed prompt instruction from YAML key `prompt`:
{intent_template.prompt}

User query:
{user_query}

Analyze the user's query intent according to the YAML prompt template. The YAML file content was provided above; do not say it is missing or implied. Return your full analysis for the responder agent."""

    style_enabled = start_evaluator_style()
    try:
        print_evaluator(f"\n--- Streaming evaluator ({EVALUATOR_MODEL}) ---\n")
        evaluator_response = stream_generate_content(
            EVALUATOR_MODEL,
            evaluator_input,
            on_text=stream_to_stderr,
        )
        print_evaluator("\n--- Evaluator complete; generating final answer ---\n\n")
    finally:
        stop_evaluator_style(style_enabled)

    return evaluator_response



def clean_final_answer(text: str) -> str:
    """Extract the end-user answer from responder output if it leaks analysis."""
    marker = "FINAL_ANSWER:"
    answer = text.rsplit(marker, 1)[-1] if marker in text else text
    answer = answer.strip()

    # Remove common wrapping quotes/markdown left by the model, but keep the
    # actual response content intact.
    while answer.startswith(("*", "-", "•")):
        answer = answer[1:].strip()
    if len(answer) >= 2 and answer[0] == answer[-1] and answer[0] in {'"', "'"}:
        answer = answer[1:-1].strip()
    return answer

def respond(user_query: str, intent_template: IntentPromptTemplate, evaluator_response: str) -> str:
    """Ask the 26B responder to produce the final end-user response."""
    responder_input = f"""You are the final responder agent. Produce exactly one final answer for the end user.

Strict output rules:
- Your entire response must be exactly one line in this format: FINAL_ANSWER: <answer>
- Do not write anything before FINAL_ANSWER.
- Do not mention roles, evaluator, intent recognition, routing, YAML, prompt templates, or this task.
- Do not quote the final answer.
- Do not provide alternatives, analysis, bullet points, labels, or explanations.

Original user query:
{user_query}

Intent recognition YAML prompt template loaded from intent_prompt.yaml:
```yaml
{intent_template.raw_yaml}
```

Parsed YAML prompt instruction:
{intent_template.prompt}

Full evaluator response from {EVALUATOR_MODEL}:
{evaluator_response}

Now write only the final answer for the end user."""
    return clean_final_answer(generate_content(RESPONDER_MODEL, responder_input))


def ask(message: str) -> str:
    """Run the complete multi-agent workflow and return the final response."""
    intent_template = load_intent_prompt_template()
    evaluator_response = evaluate_intent(message, intent_template)
    return respond(message, intent_template, evaluator_response)


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