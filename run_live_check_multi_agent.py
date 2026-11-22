#!/usr/bin/env python3
"""Phase 5 联网检查：真模型走一遍 managed-agent 接线（CodeAgent lead + 1 个 researcher）。

Key handling mirrors run_live_check.py: the surrogate is fetched from the
Secure Vault via dynamic_credentials and passed only in memory as the
api_key. It is never written to any file or log, and it never appears
in stdout. The JSON record below contains no credential.

Quota budget: this check is designed to cost exactly 2 generate calls —
lead 的 CodeAgent 一步写出 `final_answer(researcher(task=...))`（instructions
里明确允许同块串联，见下），researcher 一步 final_answer。
如果模型不听话、把依赖调用拆成两步，就会是 3 次调用：脚本如实记录
实际次数，不会为了凑数多跑；3 次也在额度内（Phase 4 当天在 2.5-flash
上已用 8 次，20 次/天）。
Do NOT use gemini-3.5-flash here: its quota for today (2026-11-21) is
already exhausted. Loop logic is already proven on FakeModel in
tests_multi_agent.py (43/43).

Usage (from this directory, with the project venv):
    .venv/bin/python run_live_check_multi_agent.py

Writes live_checks/phase5-multi-agent-check.json with latency, tokens and
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

from agents import CodeAgent, ToolCallingAgent
from default_tools import default_tools
from models import OpenAICompatibleModel

entry = dynamic_credential_entry("custom.gemini")
SURROGATE = str(entry["surrogate"]).strip()
if not SURROGATE.startswith("hsurr:"):
    raise SystemExit("stored Gemini credential did not return a surrogate")

MODEL_ID = os.environ.get("PHASE5_MODEL", "gemini-2.5-flash")
API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
TASK = "让研究员算 128*46，把它的答案原样作为最终答案返回。"


def _action_steps(agent):
    """一个 agent 的动作步数 ≈ 它的 generate 调用次数（规划关着，没别的调用）。"""
    return [s for s in agent.memory.get_full_steps() if "step_number" in s]


def main():
    model = OpenAICompatibleModel(model_id=MODEL_ID, api_base=API_BASE,
                                  api_key=SURROGATE)
    researcher = ToolCallingAgent(
        tools=default_tools(base_dir=HERE), model=model,
        name="researcher_1",
        description="Answers arithmetic questions directly with final_answer.")
    lead = CodeAgent(
        tools=default_tools(base_dir=HERE), model=model,
        name="lead", description="Lead agent",
        managed_agents=[researcher],
        instructions=("You may call the researcher tool and pass its result "
                      "directly to final_answer in the same code block, e.g. "
                      "final_answer(researcher(task=\"...\")) — no need to "
                      "print it first and use it in the next block."))
    t0 = time.perf_counter()
    result = lead.run(TASK, return_full_result=True)
    latency_s = round(time.perf_counter() - t0, 1)

    lead_steps = _action_steps(lead)
    researcher_steps = _action_steps(researcher)
    generate_calls = len(lead_steps) + len(researcher_steps)
    record = {
        "model_id": MODEL_ID, "api_base": API_BASE, "task": TASK,
        "output": result.output, "state": result.state,
        "token_usage": result.token_usage.dict() if result.token_usage else None,
        "latency_s": latency_s,
        "generate_calls": {"lead": len(lead_steps),
                           "researcher": len(researcher_steps),
                           "total": generate_calls},
        "lead_steps": [{"step_number": s.get("step_number"),
                        "code_action": (s.get("code_action") or "")[:300],
                        "observations": (s.get("observations") or "")[:500]}
                       for s in lead_steps],
        "researcher_steps": [{"step_number": s.get("step_number"),
                              "tool_calls": [tc["function"]["name"]
                                           for tc in s.get("tool_calls", [])],
                              "observations": (s.get("observations") or "")[:300]}
                             for s in researcher_steps],
    }
    out_dir = os.path.join(HERE, "live_checks")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "phase5-multi-agent-check.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    ok = result.state == "success" and "5888" in str(result.output)
    print(f"model={MODEL_ID} output={str(result.output)[:120]!r} state={result.state} "
          f"generate_calls={generate_calls} (lead {len(lead_steps)} + "
          f"researcher {len(researcher_steps)}) latency={latency_s}s "
          f"tokens={record['token_usage']} -> {'PASS' if ok else 'FAIL'}")
    print(f"record: {out_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
