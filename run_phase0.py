"""Phase 0 runner: run real smolagents 1.26.0 on the stored Gemini key.

Key handling mirrors mini-agent/run_gemini.py: the surrogate is fetched
from the Secure Vault via dynamic_credentials and passed only in memory
as api_key to OpenAIServerModel. It is never written to any file or log,
and it never appears in stdout.

Usage (from this directory, with the project venv):
    .venv/bin/python run_phase0.py leopard
    .venv/bin/python run_phase0.py calc_file
    .venv/bin/python run_phase0.py tokyo
    .venv/bin/python run_phase0.py primes
    PHASE0_MODEL=gemini-2.5-flash .venv/bin/python run_phase0.py leopard

Each run appends its trajectory to trajectories/<task>.txt and prints a
one-line summary (model, seconds, final answer, tokens).
"""
import io
import os
import sys
import time
import traceback
from contextlib import redirect_stdout

sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
from dynamic_credentials import dynamic_credential_entry

HERE = os.path.dirname(os.path.abspath(__file__))
TRAJ = os.path.join(HERE, "trajectories")
os.makedirs(TRAJ, exist_ok=True)

entry = dynamic_credential_entry("custom.gemini")
SURROGATE = str(entry["surrogate"]).strip()
if not SURROGATE.startswith("hsurr:"):
    raise SystemExit("stored Gemini credential did not return a surrogate")

MODEL_ID = os.environ.get("PHASE0_MODEL", "gemini-3.5-flash")
API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"

from smolagents import CodeAgent, DuckDuckGoSearchTool, OpenAIServerModel, tool


@tool
def read_file(path: str) -> str:
    """Read a text file and return its contents.

    Args:
        path: Path to the file, relative to the working directory.
    """
    with open(path, encoding="utf-8") as f:
        return f.read()


@tool
def write_file(path: str, content: str) -> str:
    """Write text to a file, creating it if needed.

    Args:
        path: Path to the file, relative to the working directory.
        content: Text content to write into the file.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Wrote {len(content)} characters to {path}"


def make_model():
    return OpenAIServerModel(model_id=MODEL_ID, api_base=API_BASE, api_key=SURROGATE)


TASKS = {
    "leopard": (
        "How many seconds would it take for a leopard at full speed to run through Pont des Arts?",
        [DuckDuckGoSearchTool()],
    ),
    "calc_file": (
        "Read the file data/numbers.txt in the current working directory. "
        "It contains one integer per line. Sum all the numbers, multiply the sum by 2, "
        "and report both the sum and the final result.",
        [],
    ),
    "tokyo": (
        "What is the current population of Tokyo, and in which year was that figure estimated? "
        "Give the answer with the source name.",
        [DuckDuckGoSearchTool()],
    ),
    "primes": (
        "Create a file called data/primes.txt in the current working directory "
        "containing the first 5 prime numbers, one per line, then read it back to confirm "
        "the contents and report them.",
        [],
    ),
    # Corrected reruns: the plain CodeAgent executor forbids open()/os, so file
    # access has to go through @tool tools (the smolagents-native way).
    "calc_file2": (
        "Read the file data/numbers.txt in the current working directory. "
        "It contains one integer per line. Sum all the numbers, multiply the sum by 2, "
        "and report both the sum and the final result.",
        [read_file],
    ),
    "primes2": (
        "Create a file called data/primes.txt in the current working directory "
        "containing the first 5 prime numbers, one per line, then read it back to confirm "
        "the contents and report them.",
        [write_file, read_file],
    ),
}


def dump_steps(agent) -> str:
    """Serialize the agent's memory steps into readable text."""
    lines = []
    for i, step in enumerate(getattr(agent, "memory", None).steps if getattr(agent, "memory", None) else []):
        lines.append(f"\n===== Step {i} ({type(step).__name__}) =====")
        for attr in ("model_output", "tool_calls", "observations", "error", "action_output"):
            val = getattr(step, attr, None)
            if val:
                lines.append(f"[{attr}]\n{val}")
        if getattr(step, "token_usage", None):
            lines.append(f"[token_usage] {step.token_usage}")
    return "\n".join(lines)


def run_task(name: str) -> int:
    task, tools = TASKS[name]
    out_path = os.path.join(TRAJ, f"{name}.txt")
    buf = io.StringIO()
    t0 = time.time()
    status, answer, err = "ok", "", ""
    try:
        agent = CodeAgent(tools=tools, model=make_model(), max_steps=8)
        with redirect_stdout(buf):
            answer = agent.run(task)
        steps_text = dump_steps(agent)
        monitor = getattr(agent, "monitor", None)
        tokens = ""
        if monitor is not None:
            tokens = f"total_input_tokens={getattr(monitor, 'total_input_token_count', '?')} " \
                     f"total_output_tokens={getattr(monitor, 'total_output_token_count', '?')}"
    except Exception:
        status = "FAILED"
        err = traceback.format_exc()
        steps_text = ""
        tokens = ""
    dt = time.time() - t0
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"task: {name}\nmodel: {MODEL_ID}\napi_base: {API_BASE}\n")
        f.write(f"prompt: {task}\nstatus: {status}\nseconds: {dt:.1f}\n")
        f.write(f"tokens: {tokens}\nfinal_answer: {answer}\n")
        if err:
            f.write(f"\n----- ERROR -----\n{err}\n")
        f.write("\n----- stdout (agent log) -----\n")
        f.write(buf.getvalue())
        f.write("\n----- memory steps -----\n")
        f.write(steps_text)
        f.write("\n")
    print(f"[{name}] status={status} model={MODEL_ID} seconds={dt:.1f} tokens=({tokens})")
    print(f"[{name}] final_answer: {str(answer)[:500]}")
    if err:
        print(f"[{name}] ERROR:\n{err[-2000:]}")
    print(f"[{name}] trajectory: trajectories/{name}.txt")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in TASKS:
        print("usage: run_phase0.py <" + "|".join(TASKS) + ">")
        sys.exit(2)
    sys.exit(run_task(sys.argv[1]))
