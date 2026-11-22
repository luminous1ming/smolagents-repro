#!/usr/bin/env python3
"""Phase 1：模型抽象层 —— 复现 smolagents 的 models.py 核心。

对照的上游设计（smolagents 1.26.0，见 notes/architecture.md §七）：
  - `Model` 基类：`generate(messages, stop_sequences, tools_to_call_from) -> ChatMessage`
  - `ChatMessage`（role/content/tool_calls/token_usage）、`TokenUsage`
  - `OpenAIServerModel`：包一层 OpenAI 兼容接口，任何兼容端点都能接

自己的改动（照着 mini-agent 的习惯来）：
  - 只用标准库 urllib，不依赖 openai 包（和 agent.py 的 call_llm 同一路）。
  - 上游 `Model.generate` 没有重试/退避——这里把 mini-agent 的 A4 重试
    （超时 / 连接错误 / 5xx 重试 3 次，退避 2s→5s→10s，4xx 不重试）
    直接做到模型封装层里，之后所有 agent 都自动受益。
  - stop 序列不只靠服务端：返回后也在客户端截一次，和上游做法一致，
    防止某些兼容端点忽略 stop 参数。

这个文件不调任何 agent 循环，只解决一件事：给一段消息列表，
换回一段形状固定的 `ChatMessage`。Phase 3 的 MultiStepAgent 会直接用它。
"""
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

# ---------- 消息类型 ----------

# 角色直接用字符串，和 mini-agent 的消息 dict 保持一致：
# "system" / "user" / "assistant" / "tool"。


@dataclass
class TokenUsage:
    """一次调用的 token 用量（和上游同名同字段）。"""

    input_tokens: int
    output_tokens: int
    total_tokens: int = field(init=False)

    def __post_init__(self):
        self.total_tokens = self.input_tokens + self.output_tokens

    def dict(self) -> dict:
        return {"input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.total_tokens}


@dataclass
class ToolCall:
    """一次工具调用（OpenAI tool_calls 里的一项，Phase 3 才用上）。"""

    id: str
    name: str
    arguments: str  # JSON 字符串

    def dict(self) -> dict:
        return {"id": self.id, "type": "function",
                "function": {"name": self.name, "arguments": self.arguments}}


@dataclass
class ChatMessage:
    """模型一次返回的完整消息。上游同名类的精简版：只要循环需要的字段。"""

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    token_usage: TokenUsage | None = None
    tool_call_id: str | None = None  # role == "tool" 时，回应的是哪次调用

    def dict(self) -> dict:
        """转成能直接发给 OpenAI 兼容接口的消息 dict。"""
        out: dict = {"role": self.role, "content": self.content}
        if self.tool_calls:
            out["tool_calls"] = [tc.dict() for tc in self.tool_calls]
        if self.tool_call_id:
            out["tool_call_id"] = self.tool_call_id
        return out

    @classmethod
    def from_dict(cls, data: dict, token_usage: TokenUsage | None = None) -> "ChatMessage":
        """从接口返回的 message dict 解析回来。"""
        tool_calls = None
        if data.get("tool_calls"):
            tool_calls = [ToolCall(id=tc.get("id", ""),
                                   name=(tc.get("function") or {}).get("name", ""),
                                   arguments=(tc.get("function") or {}).get("arguments", ""))
                          for tc in data["tool_calls"]]
        return cls(role=data.get("role", "assistant"),
                   content=data.get("content"),
                   tool_calls=tool_calls,
                   token_usage=token_usage)


def message_shape(msg: ChatMessage) -> tuple:
    """消息的“形状”——验收标准就用它：FakeModel 和真模型对同一段输入，
    返回的消息形状必须一致（role、content 是不是字符串、有没有用量、
    有几处工具调用）。内容本身可以不一样，形状不能不一样。"""
    return (msg.role,
            type(msg.content).__name__ if msg.content is not None else "None",
            msg.token_usage is not None,
            len(msg.tool_calls or []))


def apply_stop_sequences(content: str | None, stop_sequences: list[str] | None) -> str | None:
    """客户端截断：内容在第一个出现的 stop 序列处截断（和上游同名函数同款）。"""
    if content is None or not stop_sequences:
        return content
    for stop in stop_sequences:
        content = content.split(stop)[0]
    return content


# ---------- 重试（从 mini-agent 的 A4 原样搬过来） ----------

# 只对“抖动类”错误重试——超时、连接错误、5xx；4xx 参数错误不重试。
MAX_RETRIES = 3
RETRY_BACKOFF = [2, 5, 10]  # 秒


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code >= 500
    if isinstance(exc, urllib.error.URLError):
        return True
    return isinstance(exc, (TimeoutError, OSError))


# ---------- 模型基类 ----------


class Model:
    """所有模型的接口。子类只实现 `generate()`，别的都共用。

    签名照着上游 `Model.generate`：
        generate(messages, stop_sequences=None, tools_to_call_from=None, **kwargs)
        -> ChatMessage
    其中 tools_to_call_from 是 OpenAI 工具 schema 的 dict 列表
    （Phase 2 的 @tool 会产出这种 schema，Phase 3 才会真正传进来）。
    """

    def __init__(self, model_id: str | None = None, **kwargs):
        self.model_id = model_id
        self.kwargs = kwargs

    def generate(self, messages: list[ChatMessage],
                 stop_sequences: list[str] | None = None,
                 tools_to_call_from: list[dict] | None = None,
                 **kwargs) -> ChatMessage:
        raise NotImplementedError("子类必须实现 generate()")

    def __call__(self, *args, **kwargs) -> ChatMessage:
        return self.generate(*args, **kwargs)


# ---------- OpenAI 兼容模型（走 Gemini 免费层） ----------


class OpenAICompatibleModel(Model):
    """调任何 OpenAI 兼容的 chat completions 端点。

    Phase 1 接的是 Gemini 免费层：
        api_base = "https://generativelanguage.googleapis.com/v1beta/openai/"
        model_id = "gemini-3.5-flash"（Phase 0 已验证能跑）
    temperature 默认 0，和 mini-agent 一样，求可复现。
    """

    def __init__(self, model_id: str, api_base: str, api_key: str,
                 timeout: int = 120, temperature: float = 0,
                 max_retries: int = MAX_RETRIES,
                 retry_backoff: list[int] | None = None, **kwargs):
        super().__init__(model_id=model_id, **kwargs)
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff or list(RETRY_BACKOFF)

    def _post(self, body: dict) -> dict:
        """发一次请求，带 A4 重试。重试耗尽后把最后一次异常抛给调用方。"""
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self.api_base}/chat/completions", data=data,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"})
        last_exc: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode("utf-8"))
            except Exception as e:  # noqa: BLE001 - 重试判断在 _retryable 里
                last_exc = e
                if not _retryable(e) or attempt == self.max_retries:
                    raise
                time.sleep(self.retry_backoff[attempt])
        raise last_exc  # 正常走不到这里

    def generate(self, messages: list[ChatMessage],
                 stop_sequences: list[str] | None = None,
                 tools_to_call_from: list[dict] | None = None,
                 **kwargs) -> ChatMessage:
        body: dict = {"model": self.model_id,
                      "messages": [m.dict() for m in messages],
                      "temperature": self.temperature}
        if stop_sequences:
            body["stop"] = stop_sequences
        if tools_to_call_from:
            body["tools"] = tools_to_call_from
        body.update(kwargs)
        payload = self._post(body)
        message = payload["choices"][0]["message"]
        usage = payload.get("usage") or {}
        token_usage = TokenUsage(input_tokens=int(usage.get("prompt_tokens", 0)),
                                 output_tokens=int(usage.get("completion_tokens", 0)))
        content = apply_stop_sequences(message.get("content"), stop_sequences)
        return ChatMessage.from_dict({**message, "content": content},
                                     token_usage=token_usage)


# ---------- 测试用的假模型（不联网） ----------


def _fake_usage(messages: list[ChatMessage], reply: str) -> TokenUsage:
    """假用量：按词数算，确定性——同一段输入永远得到同一个数，
    这样测试能断言，Phase 7 评测也能拿它当基线。"""
    input_tokens = sum(len((m.content or "").split()) for m in messages)
    return TokenUsage(input_tokens=input_tokens,
                      output_tokens=len(reply.split()))


class FakeModel(Model):
    """按顺序返回写死的回复，不联网。用法：

        FakeModel(["5888"])                          # 每次都回 "5888"
        FakeModel([ChatMessage(role="assistant", content="...")])  # 直接给消息
        FakeModel(replies=[...], repeat=False)        # 用完后一直回最后一条

    测试和 Phase 7 评测都用它：先在假模型上把循环跑通，再换真模型。
    """

    def __init__(self, replies: list[str | ChatMessage] | str,
                 model_id: str = "fake", repeat: bool = True):
        super().__init__(model_id=model_id)
        if isinstance(replies, (str, ChatMessage)):
            replies = [replies]
        self.replies = list(replies)
        self.repeat = repeat
        self.calls = 0  # 调过几次 generate，可断言

    def generate(self, messages: list[ChatMessage],
                 stop_sequences: list[str] | None = None,
                 tools_to_call_from: list[dict] | None = None,
                 **kwargs) -> ChatMessage:
        idx = self.calls if self.repeat else min(self.calls, len(self.replies) - 1)
        self.calls += 1
        reply = self.replies[idx % len(self.replies)]
        if isinstance(reply, ChatMessage):
            content = apply_stop_sequences(reply.content, stop_sequences)
            return ChatMessage(role=reply.role, content=content,
                               tool_calls=reply.tool_calls,
                               token_usage=_fake_usage(messages, reply.content or ""))
        content = apply_stop_sequences(reply, stop_sequences)
        return ChatMessage(role="assistant", content=content,
                           token_usage=_fake_usage(messages, reply))
