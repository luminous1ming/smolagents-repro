#!/usr/bin/env python3
"""Phase 1 联网检查：同一段输入，FakeModel 和真模型返回的消息形状一致。

Key handling mirrors run_phase0.py: the surrogate is fetched from the
Secure Vault via dynamic_credentials and passed only in memory as the
api_key. It is never written to any file or log, and it never appears
in stdout. The JSON record below contains no credential.

Usage (from this directory, with the project venv):
    .venv/bin/python run_live_check.py

Writes live_checks/phase1-model-check.json with latency, tokens and
the message shapes from both models, and prints a one-line summary.
"""
import json
import os
import sys
import time

sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
from dynamic_credentials import dynamic_credential_entry

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from models import ChatMessage, FakeModel, OpenAICompatibleModel, message_shape

entry = dynamic_credential_entry("custom.gemini")
SURROGATE = str(entry["surrogate"]).strip()
if not SURROGATE.startswith("hsurr:"):
    raise SystemExit("stored Gemini credential did not return a surrogate")

MODEL_ID = os.environ.get("PHASE1_MODEL", "gemini-3.5-flash")
API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"

MESSAGES = [ChatMessage(role="system",
                        content="You are a helpful assistant. Answer concisely."),
            ChatMessage(role="user",
                        content="What is 128*46? Answer with just the number.")]

# 第二个检查：stop 序列在真模型上也生效（客户端截断兜底）。
STOP_MESSAGES = [ChatMessage(role="user",
                             content="Write the numbers 1, 2, 3, each on its own line.")]
STOP_SEQUENCES = ["2"]


def _record(model, messages, stop_sequences=None):
    t0 = time.perf_counter()
    msg = model.generate(messages, stop_sequences=stop_sequences)
    return {"shape": list(message_shape(msg)),
            "content": msg.content,
            "token_usage": msg.token_usage.dict() if msg.token_usage else None,
            "latency_s": round(time.perf_counter() - t0, 1)}


def main():
    fake = FakeModel(["5888"])
    live = OpenAICompatibleModel(model_id=MODEL_ID, api_base=API_BASE,
                                 api_key=SURROGATE)
    record = {"model_id": MODEL_ID, "api_base": API_BASE, "checks": {}}

    fake_main = _record(fake, MESSAGES)
    live_main = _record(live, MESSAGES)
    record["checks"]["main"] = {"fake": fake_main, "live": live_main}
    record["checks"]["main"]["shapes_match"] = fake_main["shape"] == live_main["shape"]
    record["checks"]["main"]["live_answer_contains_5888"] = "5888" in (live_main["content"] or "")

    fake_stop = _record(fake, STOP_MESSAGES, stop_sequences=STOP_SEQUENCES)
    live_stop = _record(live, STOP_MESSAGES, stop_sequences=STOP_SEQUENCES)
    record["checks"]["stop"] = {"fake": fake_stop, "live": live_stop}
    record["checks"]["stop"]["shapes_match"] = fake_stop["shape"] == live_stop["shape"]
    record["checks"]["stop"]["live_truncated_before_2"] = "2" not in (live_stop["content"] or "")

    out_dir = os.path.join(HERE, "live_checks")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "phase1-model-check.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    main_ok = record["checks"]["main"]["shapes_match"] \
        and record["checks"]["main"]["live_answer_contains_5888"]
    stop_ok = record["checks"]["stop"]["shapes_match"] \
        and record["checks"]["stop"]["live_truncated_before_2"]
    print(f"model={MODEL_ID} main: shapes_match={record['checks']['main']['shapes_match']} "
          f"live_content={live_main['content']!r} "
          f"live_latency={live_main['latency_s']}s "
          f"live_tokens={live_main['token_usage']} "
          f"-> {'PASS' if main_ok else 'FAIL'}")
    print(f"model={MODEL_ID} stop: shapes_match={record['checks']['stop']['shapes_match']} "
          f"live_content={live_stop['content']!r} "
          f"live_latency={live_stop['latency_s']}s "
          f"-> {'PASS' if stop_ok else 'FAIL'}")
    print(f"record: {out_path}")
    return 0 if (main_ok and stop_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
