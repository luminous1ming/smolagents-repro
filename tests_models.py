#!/usr/bin/env python3
"""Phase 1 自测：模型抽象层（不联网；联网检查在 run_live_check.py）。

Run from the repro directory:
    .venv/bin/python tests_models.py

测四件事：
  1. FakeModel：按顺序回写死的回复，用量按词数算、确定性；
  2. stop 截断：FakeModel 和真模型走同一份 apply_stop_sequences；
  3. OpenAICompatibleModel：请求体拼得对、返回解析得出 ChatMessage、
     工具调用也解析得出；重试只对 5xx/超时/连接错误，4xx 不重试；
  4. 验收形状：同一段输入，FakeModel 和（mock 的）真模型返回的消息形状一致。
"""
import io
import json
import sys
import urllib.error
from unittest import mock

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import models
from models import (ChatMessage, FakeModel, Model, OpenAICompatibleModel,
                    TokenUsage, apply_stop_sequences, message_shape)

PASSED = 0
FAILED = 0


def check(name, ok):
    global PASSED, FAILED
    if ok:
        PASSED += 1
        print(f"PASS {name}")
    else:
        FAILED += 1
        print(f"FAIL {name}")


def _msgs():
    return [ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="What is 128*46?")]


# ---------- 1. FakeModel ----------

fake = FakeModel(["5888", "second"])
m1 = fake.generate(_msgs())
check("FakeModel first reply", m1.role == "assistant" and m1.content == "5888")
m2 = fake.generate(_msgs())
check("FakeModel second reply in order", m2.content == "second")
m3 = fake.generate(_msgs())
check("FakeModel repeats from the start", m3.content == "5888" and fake.calls == 3)

# 用量确定性：同一段输入，两次算出同一个数；词数口径可手算
u1 = FakeModel(["5888"]).generate(_msgs()).token_usage
u2 = FakeModel(["5888"]).generate(_msgs()).token_usage
check("FakeModel usage deterministic",
      u1 == u2 and u1.input_tokens == 8 and u1.output_tokens == 1
      and u1.total_tokens == 9)

fake_once = FakeModel(["only"], repeat=False)
fake_once.generate(_msgs())
fake_once.generate(_msgs())
check("FakeModel repeat=False sticks to the last reply",
      fake_once.generate(_msgs()).content == "only")

# ---------- 2. stop 截断 ----------

check("apply_stop_sequences cuts at first stop",
      apply_stop_sequences("aaa STOP bbb", ["STOP"]) == "aaa ")
check("apply_stop_sequences no stop returns as-is",
      apply_stop_sequences("abc", ["zzz"]) == "abc")
check("apply_stop_sequences None-safe",
      apply_stop_sequences(None, ["x"]) is None)
check("FakeModel applies stop_sequences",
      FakeModel(["one two three"]).generate(_msgs(), stop_sequences=["two"]).content == "one ")

# ---------- 3. OpenAICompatibleModel：请求体与解析（mock 掉网络） ----------


class _FakeResponse:
    """urllib response 的替身：with 语句 + read()。"""

    def __init__(self, payload):
        self._payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._payload


def _api_payload(content="5888", tool_calls=None, prompt_tokens=10, completion_tokens=3):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"choices": [{"message": msg}],
            "usage": {"prompt_tokens": prompt_tokens,
                      "completion_tokens": completion_tokens}}


captured = {}


def _capture_urlopen(req, timeout=None):
    captured["url"] = req.full_url
    captured["body"] = json.loads(req.data.decode())
    captured["auth"] = req.headers.get("Authorization", "")
    return _FakeResponse(_api_payload())


model = OpenAICompatibleModel(model_id="gemini-3.5-flash",
                              api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
                              api_key="KEY-NOT-REAL")
with mock.patch.object(models.urllib.request, "urlopen", _capture_urlopen):
    out = model.generate(_msgs(), stop_sequences=["Observation:"])
check("request hits the chat completions endpoint",
      captured["url"].endswith("/chat/completions"))
check("request body has model/messages/temperature",
      captured["body"]["model"] == "gemini-3.5-flash"
      and captured["body"]["messages"][1] == {"role": "user", "content": "What is 128*46?"}
      and captured["body"]["temperature"] == 0)
check("request body carries stop sequences",
      captured["body"]["stop"] == ["Observation:"])
check("request carries the bearer key",
      captured["auth"] == "Bearer KEY-NOT-REAL")
check("response parses to ChatMessage with usage",
      out.role == "assistant" and out.content == "5888"
      and out.token_usage == TokenUsage(input_tokens=10, output_tokens=3))

tool_call_payload = [{"id": "call_1", "type": "function",
                      "function": {"name": "calc", "arguments": '{"expression": "128*46"}'}}]
with mock.patch.object(models.urllib.request, "urlopen",
                       lambda req, timeout=None: _FakeResponse(
                           _api_payload(content=None, tool_calls=tool_call_payload))):
    out_tc = model.generate(_msgs())
check("response tool_calls parse",
      out_tc.content is None and len(out_tc.tool_calls or []) == 1
      and out_tc.tool_calls[0].name == "calc"
      and out_tc.tool_calls[0].arguments == '{"expression": "128*46"}')

# tools_to_call_from 原样进 body（Phase 3 传 schema 就走这条路）
with mock.patch.object(models.urllib.request, "urlopen", _capture_urlopen):
    model.generate(_msgs(), tools_to_call_from=[
        {"type": "function", "function": {"name": "calc", "parameters": {}}}])
check("tools_to_call_from go into the request body",
      captured["body"]["tools"][0]["function"]["name"] == "calc")

# ---------- 4. 重试（A4：只重试 5xx/超时/连接错误） ----------


def _flaky_urlopen_factory(failures, exc_factory):
    calls = {"n": 0}

    def _flaky(req, timeout=None):
        calls["n"] += 1
        if calls["n"] <= failures:
            raise exc_factory()
        return _FakeResponse(_api_payload())

    _flaky.calls = calls
    return _flaky


def _http_500():
    return urllib.error.HTTPError("http://x", 500, "Server Error", {}, io.BytesIO(b""))


def _http_400():
    return urllib.error.HTTPError("http://x", 400, "Bad Request", {}, io.BytesIO(b""))


flaky = _flaky_urlopen_factory(2, _http_500)
fast_model = OpenAICompatibleModel(model_id="m", api_base="https://x/", api_key="k",
                                   retry_backoff=[0, 0, 0])
with mock.patch.object(models.urllib.request, "urlopen", flaky):
    out_retry = fast_model.generate(_msgs())
check("retry: 2x HTTP 500 then success",
      out_retry.content == "5888" and flaky.calls["n"] == 3)

flaky_400 = _flaky_urlopen_factory(5, _http_400)
try:
    with mock.patch.object(models.urllib.request, "urlopen", flaky_400):
        fast_model.generate(_msgs())
    check("no retry on HTTP 400", False)
except urllib.error.HTTPError as e:
    check("no retry on HTTP 400", e.code == 400 and flaky_400.calls["n"] == 1)

# 429 也是 4xx：按 A4 规则不重试，直接抛给调用方（调用方决定等一会儿再跑；
# 是否把 429 纳入重试是留给以后定的改进，见 notes/phase1-model.md）。
flaky_429 = _flaky_urlopen_factory(
    5, lambda: urllib.error.HTTPError("http://x", 429, "Too Many Requests", {}, io.BytesIO(b"")))
try:
    with mock.patch.object(models.urllib.request, "urlopen", flaky_429):
        fast_model.generate(_msgs())
    check("no retry on HTTP 429 (A4: 4xx not retried)", False)
except urllib.error.HTTPError as e:
    check("no retry on HTTP 429 (A4: 4xx not retried)",
          e.code == 429 and flaky_429.calls["n"] == 1)

flaky_url = _flaky_urlopen_factory(4, lambda: urllib.error.URLError("conn reset"))
try:
    with mock.patch.object(models.urllib.request, "urlopen", flaky_url):
        fast_model.generate(_msgs())
    check("URLError retries then raises after 3 retries", False)
except urllib.error.URLError:
    check("URLError retries then raises after 3 retries", flaky_url.calls["n"] == 4)

# ---------- 5. 验收：同一段输入，假模型和真模型形状一致 ----------


def _shape_of(m, msgs):
    return message_shape(m.generate(msgs))


fake_shape = _shape_of(FakeModel(["5888"]), _msgs())
with mock.patch.object(models.urllib.request, "urlopen", _capture_urlopen):
    live_shape = _shape_of(model, _msgs())
check("acceptance: FakeModel and live model return the same message shape",
      fake_shape == live_shape == ("assistant", "str", True, 0))

# ---------- 6. ChatMessage 往返 ----------

def _raises_not_implemented():
    try:
        Model().generate(_msgs())
        return False
    except NotImplementedError:
        return True


rt = ChatMessage(role="assistant", content="hi",
                 token_usage=TokenUsage(input_tokens=2, output_tokens=1))
check("ChatMessage dict round-trips role/content",
      rt.dict()["role"] == "assistant" and rt.dict()["content"] == "hi")
check("ChatMessage from_dict keeps role/content",
      ChatMessage.from_dict({"role": "user", "content": "yo"}).role == "user")
check("Model base generate raises NotImplementedError",
      _raises_not_implemented())
check("Model is callable via __call__",
      FakeModel(["x"])(_msgs()).content == "x")

print(f"\n{PASSED}/{PASSED + FAILED} passed.")
sys.exit(0 if FAILED == 0 else 1)
