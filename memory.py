#!/usr/bin/env python3
"""Phase 3：步骤记忆 —— 复现 smolagents 的 memory.py 核心。

对照的上游设计（smolagents 1.26.0，见 notes/architecture.md §三）：
  - 四种步骤都是 dataclass，各自实现 `to_messages()`；
  - `AgentMemory` 存 SystemPromptStep + steps 列表；
  - 历史 = 内存的投影：循环不拼消息，每步开始时把全部步骤重新
    序列化成消息列表（见 agents.py 的 write_memory_to_messages）。

自己的取舍（和 Phase 1/2 同一路）：
  - 只用标准库 dataclass，不引 smolagents；
  - 上游的 ChatMessage.content 是 [{"type": "text", "text": ...}] 列表，
    这里用 Phase 1 的 ChatMessage（content 就是字符串），所以
    to_messages() 直接产出字符串内容的 system/user/assistant/tool 消息；
  - 上游 ActionStep 的 tool_call 是一条 TOOL_CALL 角色消息、
    observation 是一条 TOOL_RESPONSE 角色消息；这里按 OpenAI 的
    形状来：tool_calls 直接挂在 assistant 消息上，observation 是
    带 tool_call_id 的 tool 角色消息——和 mini-agent 已有的做法一致，
    真模型（Gemini OpenAI 兼容端点）也吃得下；
  - error 存字符串（上游存 AgentError 对象）；解析/执行失败的文本
    直接进 JSONL 日志和纠正 Observation，不用再序列化异常对象；
  - CallbackRegistry 照着上游留一个最小版：按步骤类型注册回调，
    循环在每步结束时触发（token 统计等可以挂在这里）。
"""
import inspect
import time
from dataclasses import dataclass, field

from models import ChatMessage, TokenUsage, ToolCall


# ---------- 计时（上游 monitoring.Timing 的最小版） ----------


@dataclass
class StepTiming:
    """一步的起止时间（秒，time.time()）。"""

    start_time: float = field(default_factory=time.time)
    end_time: float | None = None

    def dict(self) -> dict:
        return {"start_time": self.start_time, "end_time": self.end_time}


@dataclass
class MemoryStep:
    """一步记忆的基类：能序列化成 dict（进日志），能投影成消息（进模型）。"""

    def dict(self) -> dict:
        raise NotImplementedError("子类必须实现 dict()")

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        raise NotImplementedError("子类必须实现 to_messages()")


@dataclass
class SystemPromptStep(MemoryStep):
    """系统提示词（上游同名）。summary 模式下不进消息。"""

    system_prompt: str

    def dict(self) -> dict:
        return {"system_prompt": self.system_prompt}

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        if summary_mode:
            return []
        return [ChatMessage(role="system", content=self.system_prompt)]


@dataclass
class TaskStep(MemoryStep):
    """用户下的任务（上游同名）。"""

    task: str

    def dict(self) -> dict:
        return {"task": self.task}

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        return [ChatMessage(role="user", content=f"New task:\n{self.task}")]


@dataclass
class ActionStep(MemoryStep):
    """一次“想—调工具—看结果”的动作步（上游同名，字段按需精简）。

    error 是字符串：模型解析失败、工具执行失败都会记在这里，
    下一步 to_messages() 会把它变成“Error: … Now let's retry”的
    Observation 喂回模型（mini-agent A2 纠正 Observation 的同款做法）。
    """

    step_number: int
    timing: StepTiming
    model_input_messages: list[ChatMessage] | None = None
    tool_calls: list[ToolCall] | None = None
    error: str | None = None
    model_output: str | None = None
    model_output_message: ChatMessage | None = None
    observations: str | None = None
    action_output: object = None
    token_usage: TokenUsage | None = None
    is_final_answer: bool = False
    code_action: str | None = None  # CodeAgent 抽出来的代码块（Phase 4；
    # ToolCallingAgent 不用。记下来才能复读整轮的代码轨迹，上游同款字段）

    def dict(self) -> dict:
        return {
            "step_number": self.step_number,
            "timing": self.timing.dict(),
            "model_input_messages": [m.dict() for m in self.model_input_messages]
            if self.model_input_messages else None,
            "tool_calls": [tc.dict() for tc in self.tool_calls] if self.tool_calls else [],
            "error": self.error,
            "model_output": self.model_output,
            "observations": self.observations,
            "action_output": self.action_output,
            "token_usage": self.token_usage.dict() if self.token_usage else None,
            "is_final_answer": self.is_final_answer,
            "code_action": self.code_action,
        }

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        """投影成消息：assistant（模型原文 + tool_calls）→ tool（Observation）。

        summary 模式跳过模型原文（上游同款），但工具调用和 Observation
        保留——更新计划时只看“做了什么、结果怎样”，不看模型原话。

        代码步（code_action 不为空，即 CodeAgent 的步）走另一条渲染：
        上游的 ActionStep.to_messages 也是纯文本——模型原文是普通
        assistant 文本，工具调用只是一行 "Calling tools: …" 文本，
        Observation 是 TOOL_RESPONSE 文本；它从不把 tool_calls 挂到
        assistant 消息上。原因：CodeAgent 调模型时不传 tools 参数，
        历史里一旦出现 OpenAI 形状的 tool_calls，Gemini 的 OpenAI
        兼容端点会 400（实测：第二步请求直接 INVALID_ARGUMENT）。
        所以代码步的历史全是纯文本：assistant（Thought + 代码块原文）
        → user（"Observation: …"），纠正消息也是 user 角色。
        ToolCallingAgent 的步（code_action 为空）保持原来的
        OpenAI 形状——它每步都传 tools 参数，端点吃得下。
        """
        messages: list[ChatMessage] = []
        is_code_step = self.code_action is not None
        if self.model_output is not None and not summary_mode:
            if is_code_step:
                # 代码已经在模型原文里（Thought + 代码块），不再另挂 tool_calls。
                messages.append(ChatMessage(role="assistant", content=self.model_output))
            else:
                messages.append(ChatMessage(role="assistant", content=self.model_output,
                                            tool_calls=self.tool_calls))
        elif self.tool_calls and not summary_mode and not is_code_step:
            # 模型没写正文、只调了工具：assistant 消息只带 tool_calls。
            messages.append(ChatMessage(role="assistant", content=None,
                                        tool_calls=self.tool_calls))
        if self.observations is not None:
            if is_code_step:
                # user 角色发 Observation（mini-agent 原来的做法），
                # 不用 tool 角色——tool 消息必须挂在一个真实 tool_call 上，
                # 而代码步的 python_interpreter 不是 API 层面的调用。
                messages.append(ChatMessage(role="user",
                                            content=f"Observation:\n{self.observations}"))
            else:
                call_id = self.tool_calls[0].id if self.tool_calls else None
                messages.append(ChatMessage(role="tool", content=f"Observation:\n{self.observations}",
                                            tool_call_id=call_id))
        if self.error is not None:
            # 上游同款纠正提示：告诉模型错在哪、别重复、多次失败就换思路。
            error_text = ("Error:\n" + self.error
                          + "\nNow let's retry: take care not to repeat previous errors! "
                          "If you have retried several times, try a completely different approach.\n")
            if is_code_step:
                messages.append(ChatMessage(role="user", content=error_text))
            else:
                call_id = self.tool_calls[0].id if self.tool_calls else None
                if call_id:
                    messages.append(ChatMessage(role="tool", content=error_text,
                                                tool_call_id=call_id))
                else:
                    # 没有 tool_call 可挂时，用 user 角色发纠正（tool 角色没 id
                    # 会被 OpenAI 兼容端点拒绝，见 Phase 2 笔记的 any 教训）。
                    messages.append(ChatMessage(role="user", content=error_text))
        return messages


@dataclass
class PlanningStep(MemoryStep):
    """一次规划步（上游同名）：模型产出的计划 + 触发它的输入消息。"""

    model_input_messages: list[ChatMessage]
    model_output_message: ChatMessage
    plan: str
    timing: StepTiming
    token_usage: TokenUsage | None = None

    def dict(self) -> dict:
        return {
            "model_input_messages": [m.dict() for m in self.model_input_messages],
            "model_output_message": self.model_output_message.dict(),
            "plan": self.plan,
            "timing": self.timing.dict(),
            "token_usage": self.token_usage.dict() if self.token_usage else None,
        }

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        if summary_mode:
            return []
        # 第二条 user 消息刻意换 role，防止模型把计划续写下去（上游注释原话）。
        return [ChatMessage(role="assistant", content=self.plan.strip()),
                ChatMessage(role="user", content="Now proceed and carry out this plan.")]


@dataclass
class FinalAnswerStep(MemoryStep):
    """最终答案（上游同名）。循环已结束，不进消息、不进 memory.steps。"""

    output: object

    def dict(self) -> dict:
        return {"output": self.output}

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        return []


# ---------- 记忆容器 ----------


class AgentMemory:
    """存系统提示词和全部步骤（上游同名）。

    steps 里只有 TaskStep / ActionStep / PlanningStep；
    FinalAnswerStep 只作为 run() 的返回值，不存进来（上游同款）。
    """

    def __init__(self, system_prompt: str):
        self.system_prompt = SystemPromptStep(system_prompt=system_prompt)
        self.steps: list[MemoryStep] = []

    def reset(self):
        """清掉所有步骤，系统提示词保留。"""
        self.steps = []

    def get_succinct_steps(self) -> list[dict]:
        """精简版步骤：去掉 model_input_messages（上游同名）。"""
        return [{k: v for k, v in step.dict().items() if k != "model_input_messages"}
                for step in self.steps]

    def get_full_steps(self) -> list[dict]:
        """全量步骤（含 model_input_messages，上游同名）。"""
        return [step.dict() for step in self.steps]


# ---------- 步骤回调（上游 CallbackRegistry 的最小版） ----------


class CallbackRegistry:
    """按步骤类型注册回调；循环在每步结束时触发。

    回调只收一个参数（memory_step）或两个参数（memory_step, agent），
    和上游的兼容规则一致。
    """

    def __init__(self):
        self._callbacks: dict[type, list] = {}

    def register(self, step_cls: type, callback) -> None:
        self._callbacks.setdefault(step_cls, []).append(callback)

    def callback(self, memory_step, **kwargs) -> None:
        for cls in memory_step.__class__.__mro__:
            for cb in self._callbacks.get(cls, []):
                if len(inspect.signature(cb).parameters) == 1:
                    cb(memory_step)
                else:
                    cb(memory_step, **kwargs)
