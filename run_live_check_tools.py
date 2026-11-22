#!/usr/bin/env python3
"""Phase 2 联网检查：真模型拿到注册表的 OpenAI tools 参数，能正确调 calc。

只调一次模型（额度预算：3.5-flash 今天已用完，默认用 gemini-2.5-flash；
用 PHASE2_MODEL 环境变量可换模型）。

Key handling mirrors run_live_check.py: the surrogate is fetched from the
Secure Vault via dynamic_credentials and passed only in memory as the
api_key. It is never written to any file or log, and it never appears
in stdout. The JSON record below contains no credential.

Usage (from this directory, with the project venv):
    .venv/bin/python run_live_check_tools.py
    PHASE2_MODEL=gemini-3.5-flash .venv/bin/python run_live_check_tools.py

Writes live_checks/phase2-tools-check.json with latency, tokens, the
model's tool_call, and the registry's execution result, and prints a
one-line summary.
"""
import json
import os
import sys
import time

sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
from dynamic_credentials import dynamic_credential_entry

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from models import ChatMessage, OpenAICompatibleModel
from default_tools import default_tools

entry = dynamic_credential_entry("custom.gemini")
SURROGATE = str(entry["surrogate"]).strip()
if not SURROGATE.startswith("hsurr:"):
    raise SystemExit("stored Gemini credential did not return a surrogate")

MODEL_ID = os.environ.get("PHASE2_MODEL", "gemini-2.5-flash")
API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"

MESSAGES = [ChatMessage(role="system",
                        content="You are a helpful assistant. Use the provided tools when needed."),
            ChatMessage(role="user",
                        content="Use the calc tool to compute 128*46.")]


def main():
    registry = default_tools()
    live = OpenAICompatibleModel(model_id=MODEL_ID, api_base=API_BASE,
                                 api_key=SURROGATE)
    t0 = time.perf_counter()
    msg = live.generate(MESSAGES, tools_to_call_from=registry.to_openai_tools())
    latency = round(time.perf_counter() - t0, 1)

    calls = msg.tool_calls or []
    record = {"model_id": MODEL_ID, "api_base": API_BASE,
              "latency_s": latency,
              "token_usage": msg.token_usage.dict() if msg.token_usage else None,
              "content": msg.content,
              "tool_calls": [c.dict() for c in calls],
              "executed": None}

    ok = False
    if len(calls) == 1 and calls[0].name == "calc":
        try:
            args = json.loads(calls[0].arguments)
        except json.JSONDecodeError:
            args = None
        if isinstance(args, dict) and "128*46" in str(args.get("expression", "")):
            result = registry.call(calls[0].name, args)
            record["executed"] = {"arguments": args, "result": result}
            ok = result == "5888"

    out_dir = os.path.join(HERE, "live_checks")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "phase2-tools-check.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    print(f"model={MODEL_ID} tool_calls={len(calls)} "
          f"first={[ (c.name, c.arguments) for c in calls ]} "
          f"executed={record['executed']} latency={latency}s "
          f"tokens={record['token_usage']} -> {'PASS' if ok else 'FAIL'}")
    print(f"record: {out_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
