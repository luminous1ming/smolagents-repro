#!/usr/bin/env python3
"""Phase 3 联网检查：真模型走一遍 ToolCallingAgent 的完整循环。

Key handling mirrors run_live_check.py: the surrogate is fetched from the
Secure Vault via dynamic_credentials and passed only in memory as the
api_key. It is never written to any file or log, and it never appears
in stdout. The JSON record below contains no credential.

Quota budget: this check costs exactly 2 generate calls — the task is
"算一下 128*46" (step 1: calc, step 2: final_answer). Do NOT run
longer tasks here; loop logic is already proven on FakeModel in
tests_agents.py (45/45).

Usage (from this directory, with the project venv):
    .venv/bin/python run_live_check_agents.py

Writes live_checks/phase3-agents-check.json with latency, tokens and
the step trace, and prints a one-line summary.
"""
import json
import os
import sys
import time

sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
from dynamic_credentials import dynamic_credential_entry

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from agents import ToolCallingAgent
from default_tools import default_tools
from models import OpenAICompatibleModel

entry = dynamic_credential_entry("custom.gemini")
SURROGATE = str(entry["surrogate"]).strip()
if not SURROGATE.startswith("hsurr:"):
    raise SystemExit("stored Gemini credential did not return a surrogate")

MODEL_ID = os.environ.get("PHASE3_MODEL", "gemini-2.5-flash")
API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
TASK = "算一下 128*46"


def main():
    model = OpenAICompatibleModel(model_id=MODEL_ID, api_base=API_BASE,
                                  api_key=SURROGATE)
    agent = ToolCallingAgent(tools=default_tools(base_dir=HERE), model=model)
    t0 = time.perf_counter()
    result = agent.run(TASK, return_full_result=True)
    latency_s = round(time.perf_counter() - t0, 1)

    steps = [{"step_number": s.get("step_number"),
              "tool_calls": [tc["function"]["name"] for tc in s.get("tool_calls", [])],
              "observations": s.get("observations"),
              "error": s.get("error")}
             for s in result.steps if "step_number" in s]
    record = {"model_id": MODEL_ID, "api_base": API_BASE, "task": TASK,
              "output": result.output, "state": result.state,
              "token_usage": result.token_usage.dict() if result.token_usage else None,
              "latency_s": latency_s, "steps": steps,
              "log_path": agent.log_path}
    out_dir = os.path.join(HERE, "live_checks")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "phase3-agents-check.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    ok = result.state == "success" and "5888" in str(result.output)
    print(f"model={MODEL_ID} task={TASK!r} output={result.output!r} "
          f"state={result.state} steps={[s['tool_calls'] for s in steps]} "
          f"latency={latency_s}s tokens={result.token_usage} "
          f"-> {'PASS' if ok else 'FAIL'}")
    print(f"record: {out_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
