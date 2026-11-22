#!/usr/bin/env python3
"""Phase 4 联网检查：真模型走一遍 CodeAgent 的完整循环。

Key handling mirrors run_live_check_agents.py: the surrogate is fetched
from the Secure Vault via dynamic_credentials and passed only in memory
as the api_key. It is never written to any file or log, and it never
appears in stdout. The JSON record below contains no credential.

Quota budget: at most 2 generate calls — max_steps=2, and the task is
chosen to be finishable in one code block ("直接算出 128*46，一步内用
final_answer 返回答案，不要分步打印"). Do NOT run longer tasks here;
loop logic is already proven on FakeModel in tests_code_agent.py (54/54).

Note (2026-11-21, first attempt): the first task was "计算 128*46，再把
结果乘以 2" with max_steps=2 — the model followed prompt rule 4 (dependent
tool calls: print first, use next block), so it spent step 1 on 128*46 and
step 2 on 5888*2, never calling final_answer; the forced fallback answer
was still correct (11776) but state=max_steps_error. That run also exposed
and fixed the real bug below. This script now uses a one-step task.

This check also answers the Phase 2 open item: the CodeAgent system
prompt renders final_answer as `def final_answer(answer: any) -> any:`.
If the model produces a code block and calls final_answer(), it tolerates
the `-> any` rendering in a code prompt (the Phase 2 failure was only in
the JSON tools parameter, not in prompt text).

Usage (from this directory, with the project venv):
    .venv/bin/python run_live_check_code_agent.py

Writes live_checks/phase4-code-agent-check.json with latency, tokens and
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

from agents import CodeAgent
from default_tools import default_tools
from models import OpenAICompatibleModel

entry = dynamic_credential_entry("custom.gemini")
SURROGATE = str(entry["surrogate"]).strip()
if not SURROGATE.startswith("hsurr:"):
    raise SystemExit("stored Gemini credential did not return a surrogate")

MODEL_ID = os.environ.get("PHASE4_MODEL", "gemini-2.5-flash")
API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
TASK = "直接算出 128*46，一步内用 final_answer 返回答案，不要分步打印"


def main():
    model = OpenAICompatibleModel(model_id=MODEL_ID, api_base=API_BASE,
                                  api_key=SURROGATE)
    agent = CodeAgent(tools=default_tools(base_dir=HERE), model=model,
                      max_steps=2)
    t0 = time.perf_counter()
    result = agent.run(TASK, return_full_result=True)
    latency_s = round(time.perf_counter() - t0, 1)

    steps = [{"step_number": s.get("step_number"),
              "code_action": s.get("code_action"),
              "observations": s.get("observations"),
              "error": s.get("error")}
             for s in result.steps if "step_number" in s]
    called_final_answer = any(
        "final_answer(" in (s.get("code_action") or "") for s in steps)
    record = {"model_id": MODEL_ID, "api_base": API_BASE, "task": TASK,
              "output": result.output, "state": result.state,
              "token_usage": result.token_usage.dict() if result.token_usage else None,
              "latency_s": latency_s, "steps": steps,
              "code_prompt_has_arrow_any":
                  "def final_answer(answer: any) -> any:" in agent.system_prompt,
              "model_called_final_answer_in_code": called_final_answer,
              "log_path": agent.log_path}
    out_dir = os.path.join(HERE, "live_checks")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "phase4-code-agent-check.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    ok = result.state == "success" and "5888" in str(result.output)
    print(f"model={MODEL_ID} task={TASK!r} output={result.output!r} "
          f"state={result.state} steps={len(steps)} "
          f"latency={latency_s}s tokens={result.token_usage} "
          f"arrow_any_in_prompt={record['code_prompt_has_arrow_any']} "
          f"model_called_final_answer={called_final_answer} "
          f"-> {'PASS' if ok else 'FAIL'}")
    print(f"record: {out_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
