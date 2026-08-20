#!/usr/bin/env python3
"""Multi-agent Gemma chat client through the Gemini API.

Workflow:
    user query -> evaluator (gemma-4-31b-it) reads intent_prompt.yaml
    -> streams its intent analysis in real time
    -> responder (gemma-4-26b-a4b-it) streams the final answer

Tool mode (--tool):
    user query -> tools/cli.py -> normalized request JSON
    -> evaluator (31B) <-> responder (26B) <-> executor loop:
      - Evaluator (31B): plans, instructs 26B, quality-gates feedback
      - Responder (26B): turns 31B instructions into shell commands
      - Executor (subprocess): runs those commands
      - stdout/stderr (error or success) is sent back to 31B as feedback

History context:
    Conversation turns are kept as structured session data and sent to the API
    as native multi-turn contents (role 'user'/'model'), with the static agent
    instructions carried in systemInstruction — history is never flattened
    into prompt text. An interactive session keeps the session in memory;
    --history FILE persists turns as JSON across runs.

This uses a Gemini Console / Google AI Studio API key.

Usage:
    python chat.py                              # interactive session
    python chat.py "Hello"                      # one-shot message
    python chat.py --history chat.json "Hi"     # one-shot with saved history
    python chat.py --history chat.json          # interactive, auto-saved
    python chat.py --history chat.json --reset  # ignore existing history
    python chat.py --tool "Find all TODOs"      # tool mode (evaluator<->executor loop)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# Direct Gemini API model ids.
# In OpenRouter these may be shown with provider/free suffixes, but Gemini API
# uses direct model ids like "gemma-4-31b-it" and "gemma-4-26b-a4b-it".
RESPONDER_MODEL = "gemma-4-26b-a4b-it"
EVALUATOR_MODEL = "gemma-4-31b-it"

# Socket timeout in seconds for each API request/read. Streaming keeps the
# connection alive between tokens, so this only caps idle/stalled requests.
# Override with the CHAT_TIMEOUT environment variable (in seconds).
try:
    REQUEST_TIMEOUT = int(os.getenv("CHAT_TIMEOUT", "120"))
except ValueError:
    REQUEST_TIMEOUT = 120

ENV_PATH = Path(__file__).with_name(".env")
INTENT_PROMPT_PATH = Path(__file__).with_name("intent_prompt.yaml")

# ANSI styling for evaluator-only terminal output. Terminals cannot render true
# transparency, so this uses dim gray text to make evaluator thinking feel muted.
ANSI_DIM_GRAY = "\033[2;90m"
ANSI_RESET = "\033[0m"
ANSI_BOLD_CYAN = "\033[1;36m"
ANSI_YELLOW = "\033[33m"
ANSI_GREEN = "\033[32m"
ANSI_RED = "\033[31m"
_WINDOWS_ANSI_ENABLED: bool | None = None


@dataclass(frozen=True)
class IntentPromptTemplate:
    """Raw YAML prompt template plus the parsed one-line prompt instruction."""

    raw_yaml: str
    prompt: str


@dataclass(frozen=True)
class ChatTurn:
    """One stored conversation turn: role is 'user' or 'assistant'."""

    role: str
    content: str


@dataclass
class Session:
    """In-memory multi-turn conversation.

    Turns are kept as structured (role, content) pairs and rendered as native
    Gemini contents (roles 'user'/'model') when sent, so the API sees real
    turns instead of flattened text. ``system_instruction`` carries the static
    agent instructions (role, rules, prompt template).
    """

    turns: list[ChatTurn] = field(default_factory=list)
    system_instruction: str = ""

    def add(self, role: str, content: str) -> None:
        """Append a turn (role: 'user' or 'assistant')."""
        self.turns.append(ChatTurn(role=role, content=content))

    def contents(self) -> list[dict[str, object]]:
        """Render turns as native Gemini contents (assistant -> model)."""
        return [
            {
                "role": "model" if turn.role == "assistant" else "user",
                "parts": [{"text": turn.content}],
            }
            for turn in self.turns
        ]


def load_history(path: Path) -> list[ChatTurn]:
    """Load conversation history from a JSON file (missing/corrupt -> empty)."""
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    turns: list[ChatTurn] = []
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict) or not isinstance(item.get("content"), str):
                continue
            role = item.get("role")
            turns.append(
                ChatTurn(
                    role=role if role in {"user", "assistant"} else "user",
                    content=item["content"],
                )
            )
    return turns


def save_history(path: Path, history: list[ChatTurn]) -> None:
    """Persist conversation history as a JSON file."""
    data = [{"role": turn.role, "content": turn.content} for turn in history]
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


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


def build_payload(session: Session) -> dict[str, object]:
    """Build a Gemini generateContent request payload from a session.

    Conversation turns are sent as native multi-turn contents instead of
    flattened text, and static instructions go into systemInstruction.
    """
    payload: dict[str, object] = {"contents": session.contents()}
    if session.system_instruction:
        payload["system_instruction"] = {"parts": [{"text": session.system_instruction}]}
    return payload


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


def _call_gemini(
    model: str,
    payload: dict[str, object],
    *,
    stream: bool,
    on_text: Callable[[str], None] | None = None,
    _retried: bool = False,
) -> str:
    """POST a payload to Gemini and return the generated text.

    Handles HTTP/network/timeout errors as RuntimeError. If the model rejects
    systemInstruction (HTTP 400), retries once without it.
    """
    url = build_api_url(model, stream=stream)
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            if not stream:
                data = json.loads(response.read().decode("utf-8"))
                text = extract_text(data).strip()
                if not text:
                    raise RuntimeError(
                        f"{model} returned an empty response: {json.dumps(data, indent=2)}"
                    )
                return text

            chunks: list[str] = []
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

            text = "".join(chunks).strip()
            if not text:
                raise RuntimeError(f"{model} returned an empty streaming response")
            return text
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        if (
            not _retried
            and exc.code == 400
            and "systemInstruction" in body
            and payload.get("system_instruction") is not None
        ):
            stripped = {k: v for k, v in payload.items() if k != "system_instruction"}
            return _call_gemini(
                model, stripped, stream=stream, on_text=on_text, _retried=True
            )
        kind = "streaming " if stream else ""
        raise RuntimeError(
            f"{model} {kind}API request failed with HTTP {exc.code}: {body}"
        ) from exc
    except urllib.error.URLError as exc:
        kind = "streaming " if stream else ""
        raise RuntimeError(f"Could not reach {model} {kind}API: {exc.reason}") from exc
    except TimeoutError as exc:
        kind = "streaming " if stream else ""
        suffix = "" if stream else " (no response received)"
        raise RuntimeError(
            f"{model} {kind}API request timed out after {REQUEST_TIMEOUT}s{suffix}"
        ) from exc
    except OSError as exc:
        kind = "streaming " if stream else ""
        raise RuntimeError(f"{model} {kind}API connection error: {exc}") from exc


def generate_content(model: str, session: Session) -> str:
    """Call the Gemini generateContent endpoint for a single model/session."""
    return _call_gemini(model, build_payload(session), stream=False)


def stream_generate_content(
    model: str,
    session: Session,
    on_text: Callable[[str], None] | None = None,
) -> str:
    """Stream Gemini output in real time and return the complete response text."""
    return _call_gemini(model, build_payload(session), stream=True, on_text=on_text)


def stream_to_stderr(text: str) -> None:
    """Print evaluator tokens immediately without mixing them into final stdout."""
    print_evaluator(text)


def evaluate_intent(
    user_query: str,
    intent_template: IntentPromptTemplate,
    session: Session | None = None,
) -> str:
    """Stream 31B evaluator intent analysis using the YAML prompt.

    The evaluator's static instructions go in systemInstruction; the
    conversation history (if any) is sent as native prior turns, with the
    current query as the final user turn. P3 will narrow this to a digest.
    """
    evaluator_session = Session(
        system_instruction=(
            "You are the evaluator agent in a two-agent workflow.\n\n"
            "Read this exact YAML prompt template loaded from intent_prompt.yaml:\n"
            f"```yaml\n{intent_template.raw_yaml}\n```\n\n"
            f"Parsed prompt instruction from YAML key `prompt`:\n{intent_template.prompt}\n\n"
            "Analyze the user's query intent according to the YAML prompt template. "
            "The YAML file content was provided above; do not say it is missing or implied. "
            "Use the conversation history for context when the query refers to earlier turns. "
            "Return your full analysis for the responder agent."
        )
    )
    if session is not None:
        evaluator_session.turns = list(session.turns)
    evaluator_session.add("user", user_query)

    style_enabled = start_evaluator_style()
    try:
        print_evaluator(f"\n--- Streaming evaluator ({EVALUATOR_MODEL}) ---\n")
        evaluator_response = stream_generate_content(
            EVALUATOR_MODEL,
            evaluator_session,
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


def respond(
    user_query: str,
    intent_template: IntentPromptTemplate,
    evaluator_response: str,
    session: Session | None = None,
) -> str:
    """Ask the 26B responder to produce the final end-user response.

    The responder uses the streaming endpoint too (like the evaluator) so
    output is produced incrementally instead of one blocking read, which is
    more resilient against long generations; the final answer is still printed
    only once, complete, after streaming finishes.

    Static instructions go in systemInstruction; history is native prior
    turns; the query and the evaluator's analysis are separate user turns.
    """
    responder_session = Session(
        system_instruction=(
            "You are the final responder agent. Produce exactly one final answer "
            "for the end user.\n\n"
            "Strict output rules:\n"
            "- Your entire response must be exactly one line in this format: FINAL_ANSWER: <answer>\n"
            "- Do not write anything before FINAL_ANSWER.\n"
            "- Do not mention roles, evaluator, intent recognition, routing, YAML, prompt templates, or this task.\n"
            "- Do not quote the final answer.\n"
            "- Do not provide alternatives, analysis, bullet points, labels, or explanations.\n"
            "- You may reference earlier turns from the conversation history when relevant.\n\n"
            "Intent recognition YAML prompt template loaded from intent_prompt.yaml:\n"
            f"```yaml\n{intent_template.raw_yaml}\n```\n\n"
            f"Parsed YAML prompt instruction:\n{intent_template.prompt}"
        )
    )
    if session is not None:
        responder_session.turns = list(session.turns)
    responder_session.add("user", user_query)
    responder_session.add(
        "user",
        f"Full evaluator response from {EVALUATOR_MODEL}:\n{evaluator_response}\n\n"
        "Now write only the final answer for the end user.",
    )
    return clean_final_answer(stream_generate_content(RESPONDER_MODEL, responder_session))


def ask(message: str, session: Session | None = None) -> str:
    """Run the complete multi-agent workflow and return the final response."""
    intent_template = load_intent_prompt_template()
    evaluator_response = evaluate_intent(message, intent_template, session)
    return respond(message, intent_template, evaluator_response, session)


def run_repl(session: Session, history_path: Path | None = None) -> int:
    """Interactive multi-turn session with in-memory (and optional on-disk) history."""
    print(
        "Interactive mode. Type your message, or 'exit'/'quit' to leave.",
        file=sys.stderr,
    )
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        try:
            answer = ask(user_input, session)
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            continue

        session.add("user", user_input)
        session.add("assistant", answer)
        if history_path is not None:
            save_history(history_path, session.turns)

        print(f"\nAssistant: {answer}\n")
    return 0


# ============================================================================
# Tool mode — evaluator <-> executor feedback loop
# ============================================================================

TOOLS_CLI_PATH = Path(__file__).parent / "tools" / "cli.py"


def _invoke_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Run tools/cli.py as a subprocess and capture its JSON output.

    Always passes ``--cwd`` with the current directory so the subprocess
    doesn't block waiting for interactive directory discovery.
    """
    cmd = [sys.executable, str(TOOLS_CLI_PATH), "--cwd", os.getcwd(), *args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _tool_style_header(text: str) -> None:
    """Print a colored tool-mode header to stderr."""
    if evaluator_color_enabled():
        print(f"{ANSI_BOLD_CYAN}{text}{ANSI_RESET}", file=sys.stderr, flush=True)
    else:
        print(f"=== {text} ===", file=sys.stderr, flush=True)


def _parse_commands(text: str) -> list[dict[str, str]]:
    """Extract shell commands from 26B (or 31B) output.

    Looks for fenced code blocks (bash/sh/powershell/cmd/shell) or COMMAND: lines.
    """
    import re

    commands: list[dict[str, str]] = []
    blocks = re.findall(
        r"```(?:bash|sh|shell|powershell|pwsh|cmd|bat)?\n(.*?)```",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    for block in blocks:
        for line in block.split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):
                commands.append({"command": line, "source": "codeblock"})

    cmd_lines = re.findall(r"^COMMAND:\s*(.+)$", text, re.MULTILINE)
    for c in cmd_lines:
        cmd = c.strip()
        if cmd and not any(d.get("command") == cmd for d in commands):
            commands.append({"command": cmd, "source": "prefix"})

    return commands


def _parse_gate(text: str) -> str | None:
    """Extract the quality gate decision from evaluator output."""
    import re

    m = re.search(r"\bGATE:\s*(APPROVED|RETRY|REPLAN)\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Also check for standalone keywords
    if re.search(r"\bAPPROVED\b", text, re.IGNORECASE):
        return "APPROVED"
    return None


def _normalize_command(cmd: str) -> str:
    """Stable key so the same command is never executed twice."""
    return " ".join(cmd.strip().split()).lower()


def _parse_final(text: str) -> str | None:
    """Extract the final answer from evaluator output."""
    import re

    m = re.search(r"FINAL:\s*(.*?)(?:\n|$)", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Also try FINAL_ANSWER marker
    m = re.search(r"FINAL_ANSWER:\s*(.*?)(?:\n|$)", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def _stream_model(model: str, session: Session, heading: str) -> str:
    """Stream a model response to stderr under a labeled heading."""
    style_enabled = start_evaluator_style()
    try:
        print_evaluator(f"\n--- {heading} ---\n")
        text = stream_generate_content(model, session, on_text=stream_to_stderr)
        print_evaluator("")
        return text
    finally:
        stop_evaluator_style(style_enabled)


def run_tool_workflow(request: dict) -> str:
    """Evaluator (31B) instructs responder (26B); execution feeds back to 31B.

    Each turn records command(s) actually run plus 31B/26B analysis. A command
    whose normalized key is already in history is never executed again.
    """
    objective = request["objective"]
    cwd = request.get("cwd", os.getcwd())
    limits = request.get("limits", {})
    max_steps = limits.get("max_steps", 15)
    max_replans = limits.get("max_replans", 1)
    permission_mode = request.get("permission_mode", "ask")
    timeout_strategy = limits.get("timeout_seconds", "flexible")

    _tool_style_header(f"TOOL MODE: {objective}")
    print(f"   Directory:   {cwd}", file=sys.stderr)
    print(f"   Permission:  {permission_mode}", file=sys.stderr)
    print(f"   Max steps:   {max_steps}", file=sys.stderr)
    print(f"   Max replans: {max_replans}", file=sys.stderr)
    print(f"   Evaluator:   {EVALUATOR_MODEL}", file=sys.stderr)
    print(f"   Responder:   {RESPONDER_MODEL}", file=sys.stderr)
    print(file=sys.stderr)

    original_cwd = os.getcwd()
    os.chdir(cwd)

    turn_history: list[dict] = []
    ran_commands: set[str] = set()
    step_count = 0
    replan_count = 0

    def _get_timeout(step: int) -> int:
        if isinstance(timeout_strategy, int):
            return timeout_strategy
        base = 30 if step <= 3 else 60
        return min(base + (step * 10), 300)

    os_name = "Windows (cmd/PowerShell)" if os.name == "nt" else "Unix"
    evaluator_system = (
        "You are the 31B evaluator in a two-model tool loop.\n"
        f"The 26B responder ({RESPONDER_MODEL}) writes shell commands. "
        "You never execute commands yourself.\n\n"
        f"Normalized request:\n{json.dumps(request, indent=2)}\n\n"
        f"Host OS: {os_name}. Instruct 26B to use commands that work on this OS.\n\n"
        "YOUR ROLE:\n"
        "1. Analyse the objective and write clear instructions for the 26B responder.\n"
        "2. After each turn, review turn_history (commands already run, "
        "31B/26B analysis, stdout/stderr).\n"
        "3. GATE the result, then instruct 26B again unless APPROVED.\n\n"
        "OUTPUT FORMAT:\n"
        "- First line when reviewing: GATE: APPROVED | RETRY | REPLAN\n"
        "  APPROVED: objective met. Then FINAL: <answer for the user>\n"
        "  RETRY: errors or insufficient output. Instruct 26B with a *new* command.\n"
        "  REPLAN: the plan was wrong. Write a new plan for 26B.\n"
        "- Then write INSTRUCTIONS for 26B. Do not emit shell commands yourself.\n"
        "- Never instruct harmful commands.\n"
        "- Never ask 26B to re-run a command already listed in turn_history.\n\n"
        f"Limits: max_steps={max_steps}, max_replans={max_replans}"
    )
    responder_system = (
        "You are the 26B responder in a tool-execution loop.\n"
        "The 31B evaluator gives you instructions. You output shell commands only.\n\n"
        f"Host OS: {os_name}. Use grep/find on Unix; findstr or powershell on Windows.\n"
        "Do not invent Unix tools on Windows.\n\n"
        "OUTPUT RULES:\n"
        "- Wrap each command in a fenced block: ```bash or ```powershell or ```cmd\n"
        "- One command per line. No commentary except brief # comments.\n"
        "- Never output GATE or FINAL. Never run anything; only write commands.\n"
        "- Prefer excluding .git, node_modules, venv, __pycache__.\n"
        "- Never repeat a command already listed in turn_history.\n"
    )

    while step_count < max_steps:
        step_count += 1
        timeout_sec = _get_timeout(step_count)

        evaluator_session = Session(system_instruction=evaluator_system)
        if not turn_history:
            user_msg = (
                f"Objective: {objective}\n\n"
                "Write INSTRUCTIONS for the 26B responder so it can emit "
                "OS-appropriate shell commands. Do not GATE yet."
            )
        else:
            user_msg = (
                f"Objective: {objective}\n\n"
                "Turn history (commands already run — do not repeat them):\n"
                f"{json.dumps(turn_history, indent=2)}\n\n"
                "Review each turn's analysis, command, and stdout/stderr.\n"
                "- GATE: APPROVED and FINAL: <answer> if the objective is met.\n"
                "- GATE: RETRY plus INSTRUCTIONS for a *new* 26B command.\n"
                "- GATE: REPLAN plus new INSTRUCTIONS if the plan was wrong.\n"
            )
        evaluator_session.add("user", user_msg)
        evaluator_response = _stream_model(
            EVALUATOR_MODEL,
            evaluator_session,
            f"Evaluator {EVALUATOR_MODEL} step {step_count}/{max_steps}",
        )

        gate_decision = _parse_gate(evaluator_response)
        if gate_decision == "APPROVED":
            final = _parse_final(evaluator_response)
            os.chdir(original_cwd)
            if final:
                return final.strip()
            for turn in reversed(turn_history):
                for result in turn.get("results") or []:
                    last_out = (result.get("stdout") or result.get("stderr") or "").strip()
                    if last_out:
                        return last_out
            return evaluator_response.strip()

        if gate_decision == "RETRY":
            print_evaluator(f"\n{ANSI_YELLOW}31B: RETRY — instructing 26B...{ANSI_RESET}\n")
        elif gate_decision == "REPLAN":
            replan_count += 1
            if replan_count > max_replans:
                print_evaluator(
                    f"\n{ANSI_YELLOW}Max replans ({max_replans}) reached.{ANSI_RESET}\n"
                )
                break
            print_evaluator(
                f"\n{ANSI_YELLOW}31B: REPLAN ({replan_count}/{max_replans}) "
                f"— instructing 26B...{ANSI_RESET}\n"
            )

        already = sorted(ran_commands)
        responder_session = Session(system_instruction=responder_system)
        responder_session.add(
            "user",
            f"Objective: {objective}\n\n"
            f"Instructions from {EVALUATOR_MODEL}:\n{evaluator_response}\n\n"
            + (
                "Turn history (do not repeat these commands):\n"
                f"{json.dumps(turn_history, indent=2)}\n\n"
                if turn_history
                else ""
            )
            + (
                f"Already executed (normalized): {already}\n"
                if already
                else ""
            )
            + "Emit a *new* shell command to run now.",
        )
        responder_response = _stream_model(
            RESPONDER_MODEL,
            responder_session,
            f"Responder {RESPONDER_MODEL} step {step_count}/{max_steps}",
        )

        commands = _parse_commands(responder_response)
        if not commands:
            commands = _parse_commands(evaluator_response)

        turn_results: list[dict] = []
        resp: str | None = None

        if not commands:
            turn_results.append(
                {
                    "command": "",
                    "error": "26B produced no parseable commands",
                    "stdout": "",
                    "stderr": responder_response[:2000],
                    "returncode": -1,
                    "skipped": False,
                }
            )
        else:
            for cmd_info in commands:
                cmd = cmd_info["command"].strip()
                if not cmd:
                    continue
                key = _normalize_command(cmd)
                if key in ran_commands:
                    print_evaluator(
                        f"  {ANSI_DIM_GRAY}Skipped duplicate: {cmd}{ANSI_RESET}"
                    )
                    turn_results.append(
                        {
                            "command": cmd,
                            "skipped": True,
                            "reason": "already executed this session",
                            "stdout": "",
                            "stderr": "",
                            "returncode": None,
                        }
                    )
                    continue

                if permission_mode == "ask":
                    print(f"\n  {ANSI_YELLOW}Command:{ANSI_RESET} {cmd}", file=sys.stderr)
                    try:
                        resp = input("  Execute? [Y/n/q] ").strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        resp = "q"
                    if resp in ("q", "quit"):
                        print_evaluator(
                            f"{ANSI_YELLOW}Execution cancelled by user.{ANSI_RESET}"
                        )
                        break
                    if resp in ("n", "no"):
                        print_evaluator(f"  {ANSI_DIM_GRAY}Skipped.{ANSI_RESET}")
                        turn_results.append(
                            {
                                "command": cmd,
                                "skipped": True,
                                "reason": "user declined",
                                "stdout": "",
                                "stderr": "",
                                "returncode": None,
                            }
                        )
                        continue

                print_evaluator(f"  -> Executing: {cmd}")
                try:
                    result = subprocess.run(
                        cmd,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=timeout_sec,
                    )
                    output = {
                        "command": cmd,
                        "skipped": False,
                        "returncode": result.returncode,
                        "stdout": result.stdout[:10000],
                        "stderr": result.stderr[:5000],
                    }
                    if result.returncode == 0:
                        print_evaluator(f"  OK (exit {result.returncode})")
                    else:
                        print_evaluator(f"  FAIL (exit {result.returncode})")
                        if result.stderr.strip():
                            print_evaluator(f"  stderr: {result.stderr.strip()[:200]}")
                except subprocess.TimeoutExpired:
                    output = {
                        "command": cmd,
                        "skipped": False,
                        "error": f"TIMEOUT (>={timeout_sec}s)",
                        "stdout": "",
                        "stderr": "",
                        "returncode": -1,
                    }
                    print_evaluator(f"  TIMOUT (>{timeout_sec}s)")
                except Exception as exc:
                    output = {
                        "command": cmd,
                        "skipped": False,
                        "error": str(exc),
                        "stdout": "",
                        "stderr": "",
                        "returncode": -1,
                    }
                    print_evaluator(f"  Error: {exc}")

                ran_commands.add(key)
                turn_results.append(output)

        turn_history.append(
            {
                "step": step_count,
                "gate": gate_decision,
                "evaluator_analysis": evaluator_response[:4000],
                "responder_output": responder_response[:2000],
                "results": turn_results,
            }
        )

        if resp == "q":
            break

    os.chdir(original_cwd)

    outputs: list[str] = []
    for turn in turn_history:
        for entry in turn.get("results") or []:
            if entry.get("skipped"):
                continue
            if entry.get("returncode") == 0:
                out = (entry.get("stdout") or "").strip()
                if out:
                    outputs.append(out)
            elif "error" in entry:
                outputs.append(f"[Error] {entry['error']}")

    if outputs:
        return "\n".join(outputs)

    return (
        f"[Tool mode completed — {len(turn_history)} turn(s), "
        f"{len(ran_commands)} unique command(s)]"
    )



# ============================================================================
# Main
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multi-agent Gemma chat client with conversation history and tool mode."
    )
    parser.add_argument(
        "message",
        nargs="*",
        help="one-shot message; omit to start an interactive session",
    )
    parser.add_argument(
        "--history",
        type=Path,
        help="JSON file to load/save conversation history",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="ignore any existing history file",
    )
    parser.add_argument(
        "--tool",
        action="store_true",
        help="Run in tool mode: evaluator->executor feedback loop",
    )
    args = parser.parse_args()

    # --- Tool mode ---
    if args.tool:
        message = " ".join(args.message).strip()
        if not message:
            print("Error: --tool requires an objective message.", file=sys.stderr)
            return 1

        # Invoke cli.py via subprocess to get normalized request JSON
        if not TOOLS_CLI_PATH.exists():
            print(f"Error: cli.py not found at {TOOLS_CLI_PATH}", file=sys.stderr)
            return 1

        try:
            result = _invoke_cli(message, "--dry-run")
        except subprocess.TimeoutExpired:
            print("Error: cli.py subprocess timed out.", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"Error invoking cli.py: {exc}", file=sys.stderr)
            return 1

        if result.returncode != 0:
            print(f"Error from cli.py:\n{result.stderr}", file=sys.stderr)
            return 1

        try:
            request = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            print(f"Error parsing cli.py output: {exc}\n{result.stdout}", file=sys.stderr)
            return 1

        # Run the feedback loop
        try:
            answer = run_tool_workflow(request)
        except RuntimeError as exc:
            print(f"Error in tool workflow: {exc}", file=sys.stderr)
            return 1

        print(f"\n{answer}")
        return 0

    # --- Chat mode (existing) ---
    session = Session()
    if args.history is not None and not args.reset:
        session.turns = load_history(args.history)

    message = " ".join(args.message).strip()
    if not message:
        return run_repl(session, args.history)

    try:
        answer = ask(message, session)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    session.add("user", message)
    session.add("assistant", answer)
    if args.history is not None:
        save_history(args.history, session.turns)

    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())