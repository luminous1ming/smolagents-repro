#!/usr/bin/env python3
"""Phase 3：MultiStepAgent + ToolCallingAgent —— 复现 smolagents 的 agents.py 核心。

对照的上游设计（smolagents 1.26.0，见 notes/architecture.md §二/§四）：
  - `MultiStepAgent`：run() → 循环（规划步 + 动作步），max_steps，
    write_memory_to_messages() 是历史的唯一来源，错误分两类
    （AgentGenerationError 实现侧问题直接抛；其他 AgentError 记到
    ActionStep.error，循环继续，错误以 Observation 形式喂回模型），
    步数用完走 _handle_max_steps_reached（再调一次模型强行给答案）。
  - `ToolCallingAgent`：一步之内模型输出 → 动作的翻译方式是 JSON
    tool_calls；优先用模型返回的 tool_calls，没有则文本解析兜底；
    final_answer 按名字认（且一轮只能单独调它）。

自己的取舍（和 Phase 1/2 同一路）：
  - 只用标准库，不引 smolagents；不上流式（上游 _run_stream 是生成器，
    这里 _step_stream 直接返回 ActionOutput——Phase 1 已说明流式
    generate_stream 留到后面，循环本身不需要生成器）。
  - max_steps 默认 8（mini-agent 的默认值；上游默认 20，任务量大时
    调用方可以传更大的值）。
  - 工具执行失败（未知工具、参数校验失败、工具内部抛异常）不抛错，
    直接变成该次调用的 Observation——mini-agent 的工具约定就是
    “错误也要变成 Observation 告诉模型”，和上游“抛 AgentError 再由
    循环记 error”的效果一样（循环继续），但少一层异常转换；
    只有模型生成失败和模型输出解析失败才走 AgentError。
  - JSONL 步日志沿用 mini-agent 的 runs/ 格式（见 _log_event /
    log_step）：{"ts", "step", "tool", "arg", "observation",
    "latency_ms", "raw_reply"}，规划步记 tool="planning"，最后的
    最终答案记一步 step="final"、tool="final_answer"。
  - 429 仍按 Phase 1 的 A4 规则：模型层直接抛，不重试（见
    notes/phase1-model.md 的待定事项）。

Phase 5（2026-11-21）新增 managed_agents：子智能体即工具（上游
  _setup_managed_agents + __call__ 的同款）——子智能体包成
  ManagedAgentTool（输入 task + additional_args），ToolCallingAgent
  和 CodeAgent 的系统提示词、工具列表、执行器里都带上它们；
  research.py 的 orchestrator-worker 流程就建在这上面。
"""
import ast
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime

from default_tools import FinalAnswerTool
from docker_executor import DockerExecutor
from executor import (BASE_BUILTIN_MODULES, InterpreterError,
                      LocalPythonExecutor, fix_final_answer_code)
from memory import (ActionStep, AgentMemory, CallbackRegistry, FinalAnswerStep,
                    MemoryStep, PlanningStep, StepTiming, TaskStep)
from models import ChatMessage, Model, TokenUsage, ToolCall
from tools import Tool, ToolRegistry

# ---------- 错误类型（上游 utils.py 的异常层级，按需精简） ----------


class AgentError(Exception):
    """智能体错误基类：模型侧的问题（解析、执行、步数用完），循环记下来继续跑。"""


class AgentGenerationError(AgentError):
    """模型调用本身失败（实现侧问题）：直接抛，不重试、不记 error。"""


class AgentParsingError(AgentError):
    """模型输出解析失败：没有 tool_calls，也解析不出 Action——记 error，
    下一步以纠正 Observation 喂回模型（mini-agent A2 的同款做法）."""


class AgentMaxStepsError(AgentError):
    """步数用完了还没最终答案（记在最后一步的 error 里）。"""


class AgentExecutionError(AgentError):
    """代码执行失败（CodeAgent）：记 error，下一步以纠正 Observation
    喂回模型（上游同名错误，Phase 4 新增）。"""


# ---------- 一步的产出 / 一轮的完整结果 ----------


@dataclass
class ActionOutput:
    """一步的产出（上游同名）：output 是最终答案（不是最终答案时为 None）。"""

    output: object = None
    is_final_answer: bool = False


@dataclass
class RunResult:
    """run(return_full_result=True) 时的完整结果（上游同名，字段精简）。

    steps 是 memory.get_full_steps()（TaskStep + ActionStep + PlanningStep），
    state 是 "success" 或 "max_steps_error"，token_usage 是各步用量之和。
    """

    output: object = None
    token_usage: TokenUsage | None = None
    steps: list | None = None
    state: str = "success"


# ---------- 规划提示词（上游 prompts/toolcalling_agent.yaml 的 planning 段，精简版） ----------

# 上游把提示词放 prompts/*.yaml；这里先内联成常量（Phase 4 再看要不要抽出去）。
PLANNING_INITIAL_TEMPLATE = """You are a world expert at analyzing a situation to derive facts, and plan accordingly towards solving a task.
Below I will present you a task. You will need to 1. build a survey of facts known or needed to solve the task, then 2. make a plan of action to solve the task.

## 1. Facts survey
### 1.1. Facts given in the task
List here the specific facts given in the task that could help you (there might be nothing here).
### 1.2. Facts to look up
List here any facts that we may need to look up, and where to find each of them.
### 1.3. Facts to derive
List here anything that we want to derive by logical reasoning, for instance computation.

## 2. Plan
Develop a step-by-step high-level plan taking into account the above inputs. Only write the high-level plan, DO NOT DETAIL INDIVIDUAL TOOL CALLS. After writing the final step of the plan, write the '<end_plan>' tag and stop there.

You can leverage these tools:
{tools}

---
Now begin! Here is your task:
```
{task}
```
"""

PLANNING_UPDATE_PRE = """You are a world expert at analyzing a situation, and plan accordingly towards solving a task.
You have been given the following task:
```
{task}
```

Below you will find a history of attempts made to solve this task.
You will first have to produce a survey of known and unknown facts, then propose a step-by-step high-level plan to solve the task.
If the previous tries so far have met some success, your updated plan can build on these results.
If you are stalled, you can make a completely new plan starting from scratch.
"""

PLANNING_UPDATE_POST_TEMPLATE = """Now write your updated facts below, taking into account the above history:
## 1. Updated facts survey
### 1.1. Facts given in the task
### 1.2. Facts that we have learned
### 1.3. Facts still to look up
### 1.4. Facts still to derive

Then write a step-by-step high-level plan to solve the task above.
## 2. Plan
Beware that you have {remaining_steps} steps remaining.
Do not skip steps, do not add any superfluous steps. Only write the high-level plan, DO NOT DETAIL INDIVIDUAL TOOL CALLS.
After writing the final step of the plan, write the '<end_plan>' tag and stop there.

You can leverage these tools:
{tools}

Now write your new plan below.
"""

FINAL_ANSWER_PRE = "An agent tried to answer a user query but it got stuck and failed to do so. You are tasked with providing an answer instead. Here is the agent's memory:"
FINAL_ANSWER_POST_TEMPLATE = "Based on the above, please provide an answer to the following user task:\n{task}"


# ---------- 子智能体提示词模板（上游 managed_agent 模板的同款思路，自己写的话） ----------

# 派活时包在任务外面：告诉子智能体自己是谁、谁派的活、最后交什么。
MANAGED_AGENT_TASK_TEMPLATE = """You are a managed agent named '{name}'. Your manager gave you this task:

{task}

Solve it with your own tools, then return a report: a short summary of what you found or did, followed by the key facts. Your manager will only see the report, not your intermediate steps."""

# 交报告时包在最终答案外面：父智能体只看到这份报告。
MANAGED_AGENT_REPORT_TEMPLATE = """### Report from {name}

{final_answer}"""


class ManagedAgentTool(Tool):
    """把一个子智能体包成工具（上游把 managed agent 当工具的同款做法）。

    输入固定两项：task（要它干的活）+ additional_args（可空的额外输入）。
    forward 直接调子智能体的 __call__，拿回来的是它包好的报告字符串。
    name / description / inputs 是每个实例自己的——__init__ 里先挂到
    实例上，再调 super().__init__() 做校验（ReadFileTool 的同款写法）。
    """

    def __init__(self, agent: "MultiStepAgent"):
        self.agent = agent
        self.name = agent.name
        self.description = agent.description
        self.inputs = {
            "task": {"type": "string",
                     "description": "Long detailed description of the task."},
            "additional_args": {"type": "object",
                                "description": "Dictionary of extra inputs to pass to the managed agent.",
                                "nullable": True},
        }
        self.output_type = "string"
        super().__init__()

    def forward(self, task: str, additional_args: dict | None = None) -> str:
        return self.agent(task, additional_args=additional_args)


# ---------- MultiStepAgent 基类 ----------


class MultiStepAgent:
    """多步智能体基类（上游同名）。

    子类只实现两件事：initialize_system_prompt()（系统提示词）和
    _step_stream(action_step)（一步之内“模型输出 → 动作”的翻译）。
    循环、记忆、规划、步数上限、日志都在这里。

    Args:
        tools: ToolRegistry 或 Tool 列表（final_answer 没有就自动补上，
            和上游 _setup_tools 的 setdefault 一致）。
        model: Phase 1 的 Model（generate 是唯一入口）。
        max_steps: 最多动作步数，默认 8（mini-agent 的默认值；上游是 20）。
        planning_interval: 每隔几步做一次规划（None = 不规划）；
            上游规则：第 1 步做，之后每隔 planning_interval 步做一次。
        instructions: 追加到系统提示词末尾的自定义说明。
        log_dir: JSONL 日志目录，默认本文件旁边的 runs/（mini-agent 同款）。
        return_full_result: run() 默认只返回最终答案；True 则返回 RunResult。
        step_callbacks: {步骤类: [回调]}，每步结束时触发（见 memory.CallbackRegistry）。
        name / description: 自己的名字和介绍——被当成子智能体登记时，
            父智能体靠这两个认它（上游要求 managed agent 必须有这两项）。
        managed_agents: 子智能体列表（别的 MultiStepAgent 实例），
            登记后可以像工具一样被调用（见 _setup_managed_agents）。
    """

    def __init__(self, tools, model: Model, max_steps: int = 8,
                 planning_interval: int | None = None,
                 instructions: str | None = None,
                 log_dir: str | None = None,
                 return_full_result: bool = False,
                 step_callbacks: dict | None = None,
                 name: str | None = None,
                 description: str | None = None,
                 managed_agents: list | None = None):
        if isinstance(tools, ToolRegistry):
            self.tools = tools
        else:
            self.tools = ToolRegistry(list(tools))
        if "final_answer" not in self.tools:
            self.tools.register(FinalAnswerTool())
        # 名字/介绍：被当成子智能体时，父智能体靠这两个认它。
        self.name = name or self.__class__.__name__
        self.description = description or ""
        # 子智能体必须在建记忆（系统提示词）之前登记好——系统提示词里
        # 要列出它们，循环里也要能调到它们。
        self._setup_managed_agents(managed_agents)
        self.model = model
        self.max_steps = max_steps
        self.planning_interval = planning_interval
        self.instructions = instructions
        self.return_full_result = return_full_result
        self.state: dict = {}
        self.task: str | None = None
        self.step_number = 0
        self.memory = AgentMemory(self.system_prompt)
        self.step_callbacks = CallbackRegistry()
        for step_cls, callbacks in (step_callbacks or {}).items():
            for cb in callbacks if isinstance(callbacks, list) else [callbacks]:
                self.step_callbacks.register(step_cls, cb)
        self._log_dir = log_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "runs")
        self._log_path: str | None = None

    # ---------- 子智能体（managed_agents） ----------

    def _setup_managed_agents(self, managed_agents: list | None) -> None:
        """登记子智能体（上游 _setup_managed_agents 的同款）。

        子智能体 = 另一个 MultiStepAgent；登记时要求它有 name +
        description（缺了就报错，模型没法认）；名字不能和工具重名、
        不能和自己的名字重名、子智能体之间也不能重名。
        登记完，每个子智能体包成一个 ManagedAgentTool，放进
        _managed_registry——父智能体调工具时先查自己的工具箱，
        再查这个注册表（见 ToolCallingAgent._execute_tool_call）。
        """
        self.managed_agents: dict[str, MultiStepAgent] = {}
        self._managed_tools: list[Tool] = []
        self._managed_registry = ToolRegistry()
        for agent in managed_agents or []:
            if not getattr(agent, "name", None) \
                    or not getattr(agent, "description", None):
                raise ValueError("子智能体必须有 name 和 description（模型靠它们认人）")
            if agent.name in self.tools:
                raise ValueError(f"子智能体 '{agent.name}' 和已有工具重名了")
            if agent.name == self.name:
                raise ValueError(f"子智能体 '{agent.name}' 和自己的名字重名了")
            if agent.name in self.managed_agents:
                raise ValueError(f"子智能体 '{agent.name}' 登记了两次")
            self.managed_agents[agent.name] = agent
            tool = ManagedAgentTool(agent)
            self._managed_tools.append(tool)
            self._managed_registry.register(tool)

    def __call__(self, task: str, additional_args: dict | None = None) -> str:
        """作为子智能体被父智能体调用（上游 MultiStepAgent.__call__ 的同款）。

        包一层任务提示词（告诉它自己是谁、谁派的活、最后要交一份报告），
        跑一轮，再把最终答案包成报告返回。父智能体的历史里只会看到这份
        报告字符串，看不到子智能体内部的步骤——上下文隔离就靠这层包。
        additional_args 进 self.state（CodeAgent 的执行器会把它当变量
        送进去），跑完恢复原样。
        """
        full_task = MANAGED_AGENT_TASK_TEMPLATE.format(name=self.name, task=task)
        saved_state = dict(self.state)
        try:
            if additional_args:
                self.state.update(additional_args)
            result = self.run(full_task)
        finally:
            self.state = saved_state
        final_answer = result.output if isinstance(result, RunResult) else result
        return MANAGED_AGENT_REPORT_TEMPLATE.format(name=self.name,
                                                    final_answer=final_answer)

    # ---------- 给子类用的工具列表小帮手 ----------

    def _tools_text_for_prompt(self) -> str:
        """系统提示词里的工具介绍：自己的工具 + 子智能体工具。"""
        text = "\n".join("- " + self.tools[name].to_tool_calling_prompt()
                         for name in self.tools.names())
        if self._managed_tools:
            text += "\n" + "\n".join("- " + t.to_tool_calling_prompt()
                                     for t in self._managed_tools)
        return text

    def _tools_code_for_prompt(self) -> str:
        """CodeAgent 提示词里的工具定义：自己的工具 + 子智能体工具。"""
        text = self.tools.to_code_prompt()
        if self._managed_tools:
            text += "\n\n" + "\n\n".join(t.to_code_prompt()
                                         for t in self._managed_tools)
        return text

    def _all_tools_openai(self) -> list[dict]:
        """OpenAI tools 参数：自己的工具 + 子智能体工具。"""
        return self.tools.to_openai_tools() + \
            [t.to_openai_function() for t in self._managed_tools]

    @property
    def system_prompt(self) -> str:
        return self.initialize_system_prompt()

    def initialize_system_prompt(self) -> str:
        raise NotImplementedError("子类必须实现 initialize_system_prompt()")

    # ---------- JSONL 步日志（mini-agent 的 runs/ 格式） ----------

    @property
    def log_path(self) -> str | None:
        """本次 run() 的 JSONL 日志文件（run 之前是 None）。"""
        return self._log_path

    def _new_log_path(self) -> str:
        # 文件名带微秒：同一秒内连跑两轮（测试里常见）也不会撞到同一个文件
        # （mini-agent 只精确到秒，这里多加 6 位微秒）。
        os.makedirs(self._log_dir, exist_ok=True)
        return os.path.join(self._log_dir,
                            datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".jsonl")

    def _log_event(self, record: dict) -> None:
        """往当前这轮的 JSONL 日志追加一行；没有日志文件时静默跳过。"""
        if not self._log_path:
            return
        record = {"ts": datetime.now().isoformat(timespec="seconds"), **record}
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _log_step(self, step, tool, arg, observation: str,
                  latency_ms: int, raw_reply: str) -> None:
        self._log_event({"step": step, "tool": tool, "arg": arg,
                         "observation": (observation or "")[:500],
                         "latency_ms": latency_ms,
                         "raw_reply": (raw_reply or "")[:2000]})

    # ---------- 记忆 → 消息 ----------

    def write_memory_to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        """把全部步骤重新序列化成消息列表（上游同名）。

        历史 = 内存的投影，不是增量追加的聊天记录——每步开始时都从
        memory.steps 重新拼一遍（见 notes/architecture.md §二）。
        """
        messages = self.memory.system_prompt.to_messages(summary_mode=summary_mode)
        for step in self.memory.steps:
            messages.extend(step.to_messages(summary_mode=summary_mode))
        return messages

    # ---------- 主循环 ----------

    def run(self, task: str, max_steps: int | None = None,
            return_full_result: bool | None = None):
        """跑一轮任务，返回最终答案（或 return_full_result=True 时返回 RunResult）。

        每次 run() 都重置记忆、开一个新 JSONL 日志文件（上游的 reset=True 同款）。
        """
        max_steps = max_steps or self.max_steps
        self.task = task
        self.step_number = 1
        self._log_path = self._new_log_path()
        self.memory.reset()
        self.memory.steps.append(TaskStep(task=task))

        final_answer = None
        returned_final_answer = False
        while not returned_final_answer and self.step_number <= max_steps:
            # 到点就先做一次规划步（上游 _run_stream 的同款条件）。
            if self.planning_interval is not None and (
                    self.step_number == 1
                    or (self.step_number - 1) % self.planning_interval == 0):
                planning_step = self._generate_planning_step(
                    task, is_first_step=len(self.memory.steps) == 1,
                    step=self.step_number)
                self._finalize_step(planning_step)
                self.memory.steps.append(planning_step)

            action_step = ActionStep(step_number=self.step_number,
                                     timing=StepTiming())
            try:
                output = self._step_stream(action_step)
                if output.is_final_answer:
                    final_answer = output.output
                    returned_final_answer = True
                    action_step.is_final_answer = True
            except AgentGenerationError:
                # 实现侧问题（模型调用本身失败）：直接抛，循环结束。
                raise
            except AgentError as e:
                # 模型侧问题（解析失败等）：记到 step.error，循环继续；
                # 下一步的 to_messages() 会把它变成纠正 Observation。
                action_step.error = str(e)
            finally:
                self._finalize_step(action_step)
                self.memory.steps.append(action_step)
                self.step_number += 1

        if not returned_final_answer and self.step_number == max_steps + 1:
            final_answer = self._handle_max_steps_reached(task)

        self._log_step("final", "final_answer", "",
                       str(final_answer), 0, str(final_answer))
        self._finalize_step(FinalAnswerStep(output=final_answer))

        if return_full_result if return_full_result is not None else self.return_full_result:
            total_in = total_out = 0
            usage_ok = True
            for step in self.memory.steps:
                if isinstance(step, (ActionStep, PlanningStep)):
                    if step.token_usage is None:
                        usage_ok = False
                        break
                    total_in += step.token_usage.input_tokens
                    total_out += step.token_usage.output_tokens
            token_usage = TokenUsage(input_tokens=total_in,
                                     output_tokens=total_out) if usage_ok else None
            state = "max_steps_error" if any(
                isinstance(s, ActionStep) and s.error and "max steps" in s.error.lower()
                for s in self.memory.steps) else "success"
            return RunResult(output=final_answer, token_usage=token_usage,
                             steps=self.memory.get_full_steps(), state=state)
        return final_answer

    def _finalize_step(self, memory_step: MemoryStep) -> None:
        """一步结束：补上结束时间，触发回调（上游同名）。"""
        if isinstance(memory_step, (ActionStep, PlanningStep)):
            memory_step.timing.end_time = time.time()
            if isinstance(memory_step, ActionStep):
                tool = memory_step.tool_calls[0].name if memory_step.tool_calls else None
                arg = memory_step.tool_calls[0].arguments if memory_step.tool_calls else ""
                latency_ms = int((memory_step.timing.end_time
                                  - memory_step.timing.start_time) * 1000)
                self._log_step(memory_step.step_number, tool, arg,
                               memory_step.observations or memory_step.error or "",
                               latency_ms, memory_step.model_output or "")
            else:
                self._log_step("planning", "planning", "", memory_step.plan, 0, "")
        self.step_callbacks.callback(memory_step, agent=self)

    # ---------- 规划 ----------

    def _generate_planning_step(self, task: str, is_first_step: bool,
                                step: int) -> PlanningStep:
        """做一次规划：调一次模型产出计划（上游 _generate_planning_step 的精简版）。

        首次用 initial 模板（任务 + 工具列表）；更新时用 summary 模式的
        历史（去掉系统提示词和旧计划，只看“做了什么、结果怎样”）。
        """
        start_time = time.time()
        tools_text = self._tools_text_for_prompt()
        if is_first_step:
            input_messages = [ChatMessage(
                role="user",
                content=PLANNING_INITIAL_TEMPLATE.format(task=task, tools=tools_text))]
        else:
            input_messages = [
                ChatMessage(role="system",
                            content=PLANNING_UPDATE_PRE.format(task=task)),
                *self.write_memory_to_messages(summary_mode=True),
                ChatMessage(role="user",
                            content=PLANNING_UPDATE_POST_TEMPLATE.format(
                                task=task, tools=tools_text,
                                remaining_steps=self.max_steps - step)),
            ]
        try:
            plan_message = self.model.generate(input_messages, stop_sequences=["<end_plan>"])
        except Exception as e:  # noqa: BLE001 - 规划失败也要记成步骤，不能悄悄跳过
            raise AgentGenerationError(f"Error while generating plan:\n{e}") from e
        plan_text = plan_message.content or ""
        if is_first_step:
            plan = ("Here are the facts I know and the plan of action that I will "
                    f"follow to solve the task:\n```\n{plan_text}\n```")
        else:
            plan = ("I still need to solve the task I was given:\n```\n{self.task}\n```\n\n"
                    "Here are the facts I know and my new/updated plan of action to "
                    f"solve the task:\n```\n{plan_text}\n```")
        return PlanningStep(model_input_messages=input_messages,
                            model_output_message=plan_message,
                            plan=plan,
                            timing=StepTiming(start_time=start_time, end_time=time.time()),
                            token_usage=plan_message.token_usage)

    # ---------- 步数用完 ----------

    def _handle_max_steps_reached(self, task: str):
        """步数用完了还没最终答案：再调一次模型强行给答案（上游同名）。"""
        start_time = time.time()
        final_answer = self.provide_final_answer(task)
        error_step = ActionStep(
            step_number=self.step_number,
            timing=StepTiming(start_time=start_time, end_time=time.time()),
            error="Reached max steps.",
            action_output=final_answer,
        )
        self._finalize_step(error_step)
        self.memory.steps.append(error_step)
        return final_answer

    def provide_final_answer(self, task: str) -> str:
        """根据已有记忆直接让模型给一个答案（上游同名，精简版）。

        上游是 system pre + 记忆（去掉系统提示词）+ user post；
        模型再失败就返回错误文本，不抛异常。
        """
        messages = [ChatMessage(role="system", content=FINAL_ANSWER_PRE)]
        messages += self.write_memory_to_messages()[1:]
        messages.append(ChatMessage(
            role="user",
            content=FINAL_ANSWER_POST_TEMPLATE.format(task=task)))
        try:
            return self.model.generate(messages).content or ""
        except Exception as e:  # noqa: BLE001 - 兜底答案也要返回，不能抛
            return f"Error in generating final LLM output: {e}"

    # ---------- 子类接口 ----------

    def _step_stream(self, action_step: ActionStep) -> ActionOutput:
        """一步之内“模型输出 → 动作”的翻译（上游同名，子类实现）。"""
        raise NotImplementedError("子类必须实现 _step_stream()")


# ---------- ToolCallingAgent ----------

TOOLCALLING_SYSTEM_PROMPT_TEMPLATE = """You are an expert assistant who can solve any task using tool calls. You will be given a task to solve as best you can.
To do so, you have been given access to some tools.

The tool call you write is an action: after the tool is executed, you will get the result of the tool call as an "observation".
This Action/Observation can repeat N times, you should take several steps when needed.

You can use the result of the previous action as input for the next action.
The observation will always be a string: it can represent a file, like "image_1.jpg".
Then you can use it as input for the next action.

To provide the final answer to the task, use an action blob with "name": "final_answer" tool. It is the only way to complete the task, else you will be stuck on a loop. So your final output should look like this:
Action:
{
  "name": "final_answer",
  "arguments": {"answer": "insert your final answer here"}
}

You only have access to these tools:
{tools}

Here are the rules you should always follow to solve your task:
1. ALWAYS provide a tool call, else you will fail.
2. Always use the right arguments for the tools. Never use variable names as the action arguments, use the value instead.
3. Call a tool only when needed: do not call the search agent if you do not need information, try to solve the task yourself. If no tool call is needed, use final_answer tool to return your answer.
4. Never re-do a tool call that you previously did with the exact same parameters.

Now Begin!"""


class ToolCallingAgent(MultiStepAgent):
    """JSON 工具调用智能体（上游同名）。

    每步：调模型（工具以 OpenAI tools 参数传入）→ 优先用返回的
    tool_calls；没有则文本解析兜底（先找 {"name", "arguments"} 的
    JSON 块，再试 mini-agent 的 Action: name(args) 文本格式）→
    逐个执行工具，Observation 拼起来记到 ActionStep；
    调了 final_answer（且一轮只有它）就结束。
    """

    def initialize_system_prompt(self) -> str:
        tools_text = self._tools_text_for_prompt()
        # 注意：模板里有 final_answer 的 JSON 示例（含大括号），不能用
        # str.format（会把示例里的花括号当占位符），用 replace 只换 {tools}。
        prompt = TOOLCALLING_SYSTEM_PROMPT_TEMPLATE.replace("{tools}", tools_text)
        if self.instructions:
            prompt += f"\n\n{self.instructions}"
        return prompt

    def _step_stream(self, action_step: ActionStep) -> ActionOutput:
        input_messages = self.write_memory_to_messages()
        action_step.model_input_messages = input_messages
        try:
            chat_message = self.model.generate(
                input_messages,
                stop_sequences=["Observation:", "Calling tools:"],
                tools_to_call_from=self._all_tools_openai())
        except Exception as e:  # noqa: BLE001 - 生成失败是实现侧问题，直接抛
            raise AgentGenerationError(f"Error while generating output:\n{e}") from e
        action_step.model_output_message = chat_message
        action_step.model_output = chat_message.content
        action_step.token_usage = chat_message.token_usage

        tool_calls = chat_message.tool_calls
        if not tool_calls:
            # 没有结构化 tool_calls：文本解析兜底（上游 parse_tool_calls 同款思路）。
            tool_calls = self._parse_tool_calls_from_text(chat_message.content or "")

        final_calls = [tc for tc in tool_calls if tc.name == "final_answer"]
        if final_calls and len(tool_calls) > 1:
            # 上游会抛 AgentExecutionError；这里记成 Observation 让模型自己改，
            # 效果一样（循环继续），少一层异常转换。
            action_step.tool_calls = tool_calls
            action_step.observations = (
                "Error: If you want to return an answer, please do not perform "
                "any other tool calls than the final answer tool call!")
            return ActionOutput(output=None, is_final_answer=False)

        observations: list[str] = []
        for tc in tool_calls:
            observations.append(self._execute_tool_call(tc))
        action_step.tool_calls = tool_calls
        action_step.observations = "\n".join(observations)

        if final_calls:
            answer = self._final_answer_value(final_calls[0])
            action_step.action_output = answer
            return ActionOutput(output=answer, is_final_answer=True)
        action_step.action_output = action_step.observations
        return ActionOutput(output=None, is_final_answer=False)

    def _final_answer_value(self, tool_call: ToolCall):
        """从 final_answer 的调用里取出答案值（arguments 可能是 JSON 字符串或 dict）。"""
        args = tool_call.arguments
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return args
        if isinstance(args, dict):
            return args.get("answer", "")
        return args

    def _execute_tool_call(self, tool_call: ToolCall) -> str:
        """执行一次工具调用，返回 Observation 字符串。

        先查自己的工具箱，再查子智能体注册表（上游也是两处都查）；
        未知工具、参数校验失败、工具内部抛异常——全部变成 Observation
        文本（mini-agent 的工具约定），不抛错、不结束循环。
        """
        name = tool_call.name
        registry = None
        if name in self.tools:
            registry = self.tools
        elif name in self._managed_registry:
            registry = self._managed_registry
        if registry is None:
            all_names = self.tools.names() + self._managed_registry.names()
            return (f"Error executing tool '{name}': Unknown tool {name}, "
                    f"should be one of: {', '.join(all_names)}.")
        try:
            return registry.call(name, tool_call.arguments)
        except Exception as e:  # noqa: BLE001 - 工具错误要变成 Observation
            return f"Error executing tool '{name}': {type(e).__name__}: {e}"

    # ---------- 文本解析兜底 ----------

    def _parse_tool_calls_from_text(self, content: str) -> list[ToolCall]:
        """模型没给结构化 tool_calls 时，从正文里解析（上游 parse_tool_calls 同款思路）。

        先找带 "name" 的 JSON 对象块；再试 mini-agent 的
        Action: name(args) 文本格式；都找不到就抛 AgentParsingError，
        循环会把它记成 step.error、下一步以纠正 Observation 喂回模型。
        """
        content = content.strip()
        if not content:
            raise AgentParsingError(
                "模型没有返回任何内容，也没有调用工具。请用工具调用回答，"
                "或调 final_answer 工具给出最终答案。")
        obj = _extract_json_object(content)
        if obj is not None and isinstance(obj, dict) and "name" in obj:
            args = obj.get("arguments", {})
            return [ToolCall(id="call_text_1", name=str(obj["name"]),
                             arguments=args if isinstance(args, str)
                             else json.dumps(args, ensure_ascii=False))]
        m = re.search(r"Action:\s*(\w+)\((.*)\)\s*$", content, re.S | re.M)
        if m:
            name, arg = m.group(1), m.group(2).strip()
            tool = self.tools.get(name)
            if tool is not None and len(tool.inputs) == 1:
                # 单输入工具：括号里的原始参数直接当唯一参数——
                # registry.call 只接受 dict 或 JSON 字符串，所以这里包成 dict；
                # 本来就是 JSON dict 的就原样用。
                param = next(iter(tool.inputs))
                try:
                    parsed = json.loads(arg)
                except json.JSONDecodeError:
                    parsed = arg
                if isinstance(parsed, dict):
                    args = json.dumps(parsed, ensure_ascii=False)
                else:
                    args = json.dumps({param: parsed}, ensure_ascii=False)
                return [ToolCall(id="call_text_1", name=name, arguments=args)]
            return [ToolCall(id="call_text_1", name=name, arguments=arg)]
        raise AgentParsingError(
            "模型没有返回工具调用，也解析不出 Action。请用工具调用回答，"
            "或调 final_answer 工具给出最终答案。\n模型输出：\n" + content[:500])


def _extract_json_object(text: str):
    """从文本里抠出第一个含 "name" 的 JSON 对象（大括号配对，字符串内的大括号不算）。

    找不到或解析失败返回 None。
    """
    idx = text.find('"name"')
    if idx == -1:
        return None
    start = text.rfind("{", 0, idx)
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None

# ---------- CodeAgent ----------

# 代码块的起止标签（上游 code_block_tags="markdown" 的同款：模型写
# ```python ... ```；抽取失败时再兜底 ```py 和裸代码）。
CODE_BLOCK_OPEN = "```python"
CODE_BLOCK_CLOSE = "```"

# 系统提示词模板（照着上游 prompts/code_agent.yaml 的结构，自己重新写的话）：
# Thought / Code / Observation 循环、代码块标签、print 留信息、
# final_answer 收尾、工具写成 Python def、规则列表。
# 注意：模板里只有 {tools} 和 {authorized_imports} 两个占位符，
# 用 replace 换（不用 format，避免示例代码里的花括号被误读）。
CODE_AGENT_SYSTEM_PROMPT_TEMPLATE = """You are an expert assistant who can solve any task using code blobs. You will be given a task to solve as best you can.
To do so, you have been given access to a list of tools: these tools are basically Python functions which you can call with code.
To solve the task, you must plan forward to proceed in a series of steps, in a cycle of Thought, Code, and Observation sequences.

At each step, in the 'Thought:' sequence, you should first explain your reasoning towards solving the task and the tools that you want to use.
Then in the Code sequence you should write the code in simple Python. The code sequence must be opened with '```python', and closed with '```'.
During each intermediate step, you can use 'print()' to save whatever important information you will then need.
These print outputs will then appear in the 'Observation:' field, which will be available as input for the next step.
In the end you have to return a final answer using the `final_answer` tool.

Here are a few examples using notional tools:
---
Task: "What is 128 * 46?"
Thought: I will compute this with Python code and return the answer with the final_answer tool.
```python
result = 128 * 46
final_answer(result)
```
---
Task: "Add up all the numbers in the file 'nums.txt' and double the total."
Thought: I will read the file with the read_file tool, sum the numbers with a loop, and return the doubled total.
```python
text = read_file(path="nums.txt")
total = 0
for line in text.split():
    total += int(line)
final_answer(total * 2)
```
---
Above examples were using notional tools that might not exist for you. On top of performing computations in the Python code snippets that you create, you only have access to these tools, behaving like regular python functions:
```python
{tools}
```

Here are the rules you should always follow to solve your task:
1. Always provide a 'Thought:' sequence, and a '```python' sequence ending with '```', else you will fail.
2. Use only variables that you have defined!
3. Always use the right arguments for the tools. DO NOT pass the arguments as a dict, but use the arguments directly as in 'answer = read_file(path="nums.txt")'.
4. Do not chain tool calls that depend on each other's output in the same code block: their return format is plain text, so print the result first and use it in the next block.
5. Call a tool only when needed, and never re-do a tool call that you previously did with the exact same parameters.
6. Don't name any new variable with the same name as a tool: for instance don't name a variable 'final_answer'.
7. You can use imports in your code, but only from the following list of modules: {authorized_imports}
8. The state persists between code executions: so if in one step you've created variables or imported modules, these will all persist.
9. Files can only be read through the read_file tool. Never use open() or import os: they are forbidden and will fail.
10. Don't give up! You're in charge of solving the task, not providing directions to solve it.

Now Begin!"""


def extract_code_blocks(text: str, tags=(CODE_BLOCK_OPEN, CODE_BLOCK_CLOSE)) -> str | None:
    """抽出给定标签围起来的所有代码块，拼成一段（上游 extract_code_from_text 同款）。

    没找到返回 None（不是抛错），调用方再试别的标签或兜底。
    """
    pattern = rf"{re.escape(tags[0])}(.*?){re.escape(tags[1])}"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return "\n\n".join(m.strip() for m in matches)
    return None


def parse_code_blobs(text: str, tags=(CODE_BLOCK_OPEN, CODE_BLOCK_CLOSE)) -> str:
    """从模型输出里抽代码（上游 parse_code_blobs 同款思路）。

    1) 主标签 ```python 围栏；2) ```py 围栏也认；3) 整段能 ast.parse
    就当裸代码用。都失败抛 AgentParsingError——循环会把它记成
    step.error，下一步以纠正 Observation 喂回模型。
    """
    code = extract_code_blocks(text, tags)
    if code is None:
        code = extract_code_blocks(text, ("```py", CODE_BLOCK_CLOSE))
    if code is not None:
        return code
    try:
        ast.parse(text)
        return text
    except SyntaxError:
        pass
    if "final" in text and "answer" in text:
        raise AgentParsingError(
            "Your code snippet is invalid, because no code block was found in it.\n"
            f"Here is your output:\n{text[:500]}\n"
            "It seems like you're trying to return the final answer, "
            "you can do it as follows:\n"
            f"{CODE_BLOCK_OPEN}\nfinal_answer(\"YOUR FINAL ANSWER HERE\")\n"
            f"{CODE_BLOCK_CLOSE}")
    raise AgentParsingError(
        "No code block found in model output. Please write your code in a "
        f"'{CODE_BLOCK_OPEN}' block ending with '{CODE_BLOCK_CLOSE}', "
        "or call final_answer() to return the final answer.\n"
        f"Model output:\n{text[:500]}")


class CodeAgent(MultiStepAgent):
    """代码智能体（上游同名）。

    每步：调模型 → 从回复里抽 ```python 代码块 → 交给 AST 执行器跑，
    print 输出 + 最后一条表达式的值拼成 Observation；代码里调了
    final_answer() 就结束。一步能做多件事（变量、循环、组合调用），
    所以同样任务的步数比 ToolCallingAgent 少——Phase 4 的验收标准。

    Args:
        additional_authorized_imports: 在 11 个默认白名单模块之外，
            再允许模型 import 的模块（上游同名参数）。
        executor: 自带执行器可传进来（测试、Phase 6 的 Docker 执行器都
            用这个口子）；不传就建一个本地执行器。
        max_operations / timeout_seconds: 执行器的操作数上限和墙钟超时
            （上游的 MAX_OPERATIONS / MAX_EXECUTION_TIME_SECONDS）。
    """

    def __init__(self, tools, model: Model, max_steps: int = 8,
                 planning_interval: int | None = None,
                 instructions: str | None = None,
                 log_dir: str | None = None,
                 return_full_result: bool = False,
                 step_callbacks: dict | None = None,
                 name: str | None = None,
                 description: str | None = None,
                 managed_agents: list | None = None,
                 additional_authorized_imports: list[str] | None = None,
                 executor: LocalPythonExecutor | DockerExecutor | None = None,
                 max_operations: int = 100000,
                 timeout_seconds: int | None = 30):
        self.additional_authorized_imports = list(additional_authorized_imports or [])
        self.authorized_imports = sorted(
            set(BASE_BUILTIN_MODULES) | set(self.additional_authorized_imports))
        self.python_executor = executor or LocalPythonExecutor(
            additional_authorized_imports=self.additional_authorized_imports,
            max_operations=max_operations, timeout_seconds=timeout_seconds)
        super().__init__(tools, model, max_steps, planning_interval,
                         instructions, log_dir, return_full_result, step_callbacks,
                         name, description, managed_agents)
        self.python_executor.send_tools(self.tools)
        # 子智能体也要进执行器的命名空间，模型才能在代码里直接调它们
        # （上游 send_tools({**tools, **managed_agents}) 的同款）。
        self.python_executor.send_tools(self._managed_registry)

    def initialize_system_prompt(self) -> str:
        tools_text = self._tools_code_for_prompt()
        prompt = CODE_AGENT_SYSTEM_PROMPT_TEMPLATE.replace("{tools}", tools_text)
        prompt = prompt.replace("{authorized_imports}", str(self.authorized_imports))
        if self.instructions:
            prompt += f"\n\n{self.instructions}"
        return prompt

    def run(self, task: str, max_steps: int | None = None,
            return_full_result: bool | None = None):
        # 每轮开始前清空执行器状态（变量不跨任务），再把 agent.state 的
        # 变量送进去（上游 run() 里 send_variables 的同款）。
        self.python_executor.reset()
        self.python_executor.send_variables(dict(self.state))
        return super().run(task, max_steps=max_steps,
                           return_full_result=return_full_result)

    def _step_stream(self, action_step: ActionStep) -> ActionOutput:
        input_messages = self.write_memory_to_messages()
        action_step.model_input_messages = input_messages
        try:
            chat_message = self.model.generate(
                input_messages,
                stop_sequences=["Observation:", "Calling tools:"])
        except Exception as e:  # noqa: BLE001 - 生成失败是实现侧问题，直接抛
            raise AgentGenerationError(f"Error while generating output:\n{e}") from e
        output_text = chat_message.content or ""
        # 上游同款：如果输出没以结束标签结尾，补上——能提醒后续调用
        # 也以这个标签结尾，省得模型一直写下去。
        if output_text and not output_text.strip().endswith(CODE_BLOCK_CLOSE):
            output_text += "\n" + CODE_BLOCK_CLOSE
        action_step.model_output_message = chat_message
        action_step.model_output = output_text
        action_step.token_usage = chat_message.token_usage

        # 解析：抽代码块 → 修 final_answer 赋值。解析失败抛
        # AgentParsingError，循环记 error、下一步纠正（Phase 3 同款）。
        code_action = parse_code_blobs(output_text)
        code_action = fix_final_answer_code(code_action)
        action_step.code_action = code_action
        action_step.tool_calls = [ToolCall(id=f"call_{action_step.step_number}",
                                           name="python_interpreter",
                                           arguments=code_action)]

        # 执行：交给 AST 执行器。执行失败先把已打印的日志记成
        # Observation（上游同款），再以 AgentExecutionError 交给循环。
        try:
            code_output = self.python_executor(code_action)
        except InterpreterError as e:
            logs = str(self.python_executor.state.get("_print_outputs", ""))
            if logs:
                action_step.observations = "Execution logs:\n" + logs
            raise AgentExecutionError(str(e)) from e
        observation = "Execution logs:\n" + code_output.logs
        observation += "\nLast output from code snippet:\n" + str(code_output.output)
        action_step.observations = observation
        action_step.action_output = code_output.output
        if code_output.is_final_answer:
            return ActionOutput(output=code_output.output, is_final_answer=True)
        return ActionOutput(output=None, is_final_answer=False)
