#!/usr/bin/env python3
"""Multi-agent Gemma chat client through the Gemini API.

Workflow:
    user query -> evaluator (gemma-4-31b-it) reads intent_prompt.yaml
    -> streams its intent analysis in real time
    -> responder (gemma-4-26b-a4b-it) streams the final answer

Tool mode (--tool):
    user query -> tools/cli.py -> normalized request JSON
    -> evaluator <-> executor feedback loop:
      - Evaluator (31B): plans & quality-gates
      - Executor (subprocess): runs commands
      - Feedback loop until quality pass or limits reached

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
    """Extract commands from evaluator output.

    Looks for bash code blocks (```bash ... ```) or COMMAND: prefixes.
    """
    import re

    commands: list[dict[str, str]] = []

    # Try ```bash code blocks first
    blocks = re.findall(r"```bash\n(.*?)```", text, re.DOTALL)
    for block in blocks:
        for line in block.split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):
                commands.append({"command": line, "source": "codeblock"})

    # Also try COMMAND: prefix lines
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


def run_tool_workflow(request: dict) -> str:
    """Execute the evaluator <-> executor feedback loop for tool execution.

    Flow:
      1. EVALUATOR (31B): analyses objective -> produces plan & commands
      2. EXECUTOR (subprocess): runs the commands
      3. EVALUATOR GATE: checks results for errors / content quality
      4. If error/insufficient -> evaluator provides fixes -> executor retries
      5. If good -> evaluator decides display -> final output

    Loop terminates when:
      - Evaluator approves the result (GATE: APPROVED + FINAL)
      - max_steps reached (safety limit)
      - max_replans reached (re-plan exhaustion)
    """
    objective = request["objective"]
    cwd = request.get("cwd", os.getcwd())
    limits = request.get("limits", {})
    max_steps = limits.get("max_steps", 15)
    max_retries_per_step = limits.get("max_retries_per_step", 2)
    max_replans = limits.get("max_replans", 1)
    permission_mode = request.get("permission_mode", "ask")
    timeout_strategy = limits.get("timeout_seconds", "flexible")

    # --- Display header ---
    _tool_style_header(f"TOOL MODE: {objective}")
    print(f"   Directory:   {cwd}", file=sys.stderr)
    print(f"   Permission:  {permission_mode}", file=sys.stderr)
    print(f"   Max steps:   {max_steps}", file=sys.stderr)
    print(f"   Max replans: {max_replans}", file=sys.stderr)
    print(file=sys.stderr)

    # Change to working directory
    original_cwd = os.getcwd()
    os.chdir(cwd)

    # State trackers
    execution_history: list[dict] = []
    step_count = 0
    replan_count = 0
    gate_decision: str | None = None

    # Helper to compute dynamic timeout
    def _get_timeout(step: int) -> int:
        if isinstance(timeout_strategy, int):
            return timeout_strategy
        # Flexible: scale with step complexity, capped at 300s
        base = 30 if step <= 3 else 60
        return min(base + (step * 10), 300)

    # Build the system instruction for the evaluator tool agent
    system_instruction = (
        "You are the evaluator agent in a tool-execution feedback loop.\n\n"
        f"Normalized request:\n{json.dumps(request, indent=2)}\n\n"
        "YOUR ROLE:\n"
        "1. Analyse the user's objective and break it into concrete steps.\n"
        "2. For each step, output the exact shell command to run.\n"
        "3. After execution results come back, review them thoroughly:\n"
        "   - CHECK_FOR_ERRORS: Are there error messages in stdout/stderr?\n"
        "   - CHECK_CONTENT: Is the output sufficient to meet the objective?\n"
        "   - CHECK_COMPLETENESS: Does the result fully satisfy the objective?\n"
        "4. Decide the next action based on your review.\n\n"
        "OUTPUT FORMAT RULES:\n"
        "- Wrap each command to execute inside ```bash\\n<command>\\n```\n"
        "- When reviewing results, start with: GATE: <decision>\n"
        "  Decision must be one of: APPROVED, RETRY, or REPLAN.\n"
        "  - APPROVED: The result is satisfactory. Output FINAL: <final answer>\n"
        "  - RETRY: The result has errors or insufficient content. Explain the fix\n"
        "    and output new commands in ```bash``` blocks.\n"
        "  - REPLAN: A full re-think is needed (plan was wrong). Output new plan.\n"
        "- Final answer must use: FINAL: <answer>\n"
        "- Be concise but specific in commands.\n"
        "- Never output commands that could cause harm.\n\n"
        f"Limits: max_steps={max_steps}, max_retries_per_step={max_retries_per_step}, "
        f"max_replans={max_replans}"
    )

    # ======================================================================
    # Main loop
    # ======================================================================
    while step_count < max_steps:
        step_count += 1
        timeout_sec = _get_timeout(step_count)

        # --- Prepare evaluator session ---
        evaluator_session = Session(system_instruction=system_instruction)

        # Build the user message for this iteration
        if not execution_history:
            # First iteration: send objective
            user_msg = (
                f"Objective: {objective}\n\n"
                "Analyse this objective and output the commands to execute. "
                "Remember to wrap each command in ```bash``` blocks."
            )
        else:
            # Subsequent iterations: send execution history for review
            user_msg = (
                f"Objective: {objective}\n\n"
                f"Execution history so far (step {step_count}):\n"
                f"{json.dumps(execution_history, indent=2)}\n\n"
                "Review the execution results above.\n"
                "- If there are errors or content is insufficient, output GATE: RETRY with fixes.\n"
                "- If the results satisfy the objective, output GATE: APPROVED and FINAL: <answer>.\n"
                "- If the plan was fundamentally wrong, output GATE: REPLAN.\n"
            )

        evaluator_session.add("user", user_msg)

        # --- Call evaluator (stream to stderr) ---
        style_enabled = start_evaluator_style()
        try:
            print_evaluator(f"\n--- Evaluator step {step_count}/{max_steps} ---\n")
            evaluator_response = stream_generate_content(
                EVALUATOR_MODEL,
                evaluator_session,
                on_text=stream_to_stderr,
            )
            print_evaluator("")  # newline
        finally:
            stop_evaluator_style(style_enabled)

        # --- Parse evaluator output ---
        commands = _parse_commands(evaluator_response)
        gate_decision = _parse_gate(evaluator_response)

        # If gate says APPROVED with FINAL, extract and return
        if gate_decision == "APPROVED":
            final = _parse_final(evaluator_response)
            if final:
                os.chdir(original_cwd)
                return final.strip()
            # If no FINAL but APPROVED, treat last execution stdout as result
            if execution_history:
                last = execution_history[-1]
                last_out = last.get("stdout", last.get("stderr", "")).strip()
                if last_out:
                    os.chdir(original_cwd)
                    return last_out
            # No content yet but approved — just return the evaluator's response
            os.chdir(original_cwd)
            return evaluator_response.strip()

        # If gate says RETRY or REPLAN
        if gate_decision == "RETRY":
            # Commands from evaluator will be re-executed in next loop iteration
            print_evaluator(f"\n{ANSI_YELLOW}Retrying with fixes...{ANSI_RESET}\n")
            continue

        if gate_decision == "REPLAN":
            replan_count += 1
            if replan_count > max_replans:
                print_evaluator(
                    f"\n{ANSI_YELLOW}Max replans ({max_replans}) reached.{ANSI_RESET}\n"
                )
                break
            print_evaluator(
                f"\n{ANSI_YELLOW}Re-planning ({replan_count}/{max_replans})...{ANSI_RESET}\n"
            )
            continue

        # --- If no gate decision yet but we have commands, execute them ---
        if not commands:
            continue

        # --- Execute commands ---
        resp: str | None = None
        for cmd_info in commands:
            cmd = cmd_info["command"].strip()
            if not cmd:
                continue

            # Permission check
            if permission_mode == "ask":
                print(f"\n  {ANSI_YELLOW}Command:{ANSI_RESET} {cmd}", file=sys.stderr)
                try:
                    resp = input("  Execute? [Y/n/q] ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    resp = "q"
                if resp in ("q", "quit"):
                    print_evaluator(f"{ANSI_YELLOW}Execution cancelled by user.{ANSI_RESET}")
                    break
                if resp in ("n", "no"):
                    print_evaluator(f"  {ANSI_DIM_GRAY}Skipped.{ANSI_RESET}")
                    continue

            print_evaluator(f"  -> Executing: {cmd}")

            # Execute with retry logic
            for attempt in range(1 + max_retries_per_step):
                try:
                    result = subprocess.run(
                        cmd,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=timeout_sec,
                    )
                    output = {
                        "step": step_count,
                        "attempt": attempt + 1,
                        "command": cmd,
                        "returncode": result.returncode,
                        "stdout": result.stdout[:10000],   # trim for model context
                        "stderr": result.stderr[:5000],
                    }

                    # Print result summary
                    if result.returncode == 0:
                        print_evaluator(f"  OK (exit {result.returncode})")
                    else:
                        print_evaluator(f"  FAIL (exit {result.returncode}) attempt {attempt+1}")
                        if result.stderr.strip():
                            print_evaluator(f"  stderr: {result.stderr.strip()[:200]}")

                except subprocess.TimeoutExpired:
                    output = {
                        "step": step_count,
                        "attempt": attempt + 1,
                        "command": cmd,
                        "error": f"TIMEOUT (>={timeout_sec}s)",
                        "stdout": "",
                        "stderr": "",
                    }
                    print_evaluator(f"  TIMOUT (>{timeout_sec}s)")

                except Exception as exc:
                    output = {
                        "step": step_count,
                        "attempt": attempt + 1,
                        "command": cmd,
                        "error": str(exc),
                        "stdout": "",
                        "stderr": "",
                    }
                    print_evaluator(f"  Error: {exc}")

                execution_history.append(output)

                # If success or last attempt, move to next command
                if output.get("returncode", -1) == 0 or attempt >= max_retries_per_step:
                    break

        # If user cancelled, exit loop
        if resp == "q":
            break

    # ======================================================================
    # Loop exhausted — build best-effort result
    # ======================================================================
    os.chdir(original_cwd)

    # Collect all stdout from successful commands
    outputs: list[str] = []
    for entry in execution_history:
        if entry.get("returncode") == 0:
            out = entry.get("stdout", "").strip()
            if out:
                outputs.append(out)
        elif "error" in entry:
            outputs.append(f"[Error] {entry['error']}")

    if outputs:
        return "\n".join(outputs)

    # Fallback: return summary
    return (
        f"[Tool mode completed — {len(execution_history)} step(s) executed, "
        f"{len(execution_history)} result(s) collected]"
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