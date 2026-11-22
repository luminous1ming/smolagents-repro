#!/usr/bin/env python3
"""Phase 3 自测：MultiStepAgent + ToolCallingAgent（不联网）。

Run from the repro directory:
    .venv/bin/python tests_agents.py

测六件事：
  1. 记忆：四种步骤的 to_messages() 形状（system/task/action/planning/
     final_answer），summary 模式的裁剪，error 的纠正 Observation，
     AgentMemory 的 reset 和精简/全量步骤；
  2. 与 mini-agent 的对等验收：同一个演示任务
     “帮我算一下 128*46，再读一下 README.md 的前 5 行”，
     FakeModel 脚本走新 ToolCallingAgent，结果和 mini-agent 一样正确
     （5888 + README 前 5 行），且 JSONL 日志记下每一步；
  3. 规划：planning_interval=1/2/None 时规划步出现的时机；
  4. 步数上限：用完 max_steps 后 _handle_max_steps_reached 强行给答案，
     并记 error；
  5. 错误处理：模型输出解析失败（纠正 Observation，不崩）、
     文本解析兜底（JSON 块 / Action: 文本格式）、未知工具和参数
     校验失败都变成 Observation、final_answer 不能和其他调用同批；
  6. 循环本身：模型确实拿到 OpenAI tools 参数、生成失败直接抛、
     run() 会重置记忆、步骤回调会触发、max_steps 可按次覆盖。
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import agents
from agents import (ActionOutput, AgentGenerationError, MultiStepAgent,
                    RunResult, ToolCallingAgent)
from default_tools import calc as _calc_tool
from default_tools import default_tools
from memory import (ActionStep, AgentMemory, FinalAnswerStep, PlanningStep,
                    StepTiming, SystemPromptStep, TaskStep)
from models import ChatMessage, FakeModel, Model, ToolCall
from tools import ToolRegistry

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


def raises(name, exc_type, fn):
    try:
        fn()
    except exc_type:
        check(name, True)
    except Exception:
        check(name, False)
    else:
        check(name, False)


# ---------- 测试夹具 ----------

README_LINES = ["line1", "line2", "line3", "line4", "line5", "line6"]


def make_readme(base_dir: str) -> str:
    """在 base_dir 里写一个 6 行的 README.md（read_file 的测试靶子）。"""
    path = os.path.join(base_dir, "README.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(README_LINES) + "\n")
    return path


def tool_msg(name: str, call_id: str, arguments, content: str | None = None) -> ChatMessage:
    """一段 FakeModel 回复：一次工具调用（arguments 写 JSON 字符串）。

    默认带一句简短正文（真模型调工具时通常也有正文），这样
    FakeModel 的用量里 output_tokens 不为 0。
    """
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    return ChatMessage(role="assistant", content=content or f"调用 {name} 工具",
                       tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)])


def parity_replies(final_text: str = "128*46=5888，README 前 5 行：line1..line5"):
    """mini-agent 演示任务的三步脚本：calc → read_file → final_answer。"""
    return [tool_msg("calc", "c1", {"expression": "128*46"}),
            tool_msg("read_file", "c2", {"path": "README.md"}),
            tool_msg("final_answer", "c3", {"answer": final_text})]


def parity_task() -> str:
    return "帮我算一下 128*46，再读一下 README.md 的前 5 行"


def run_parity(base_dir: str, **agent_kwargs):
    """跑一遍对等任务，返回 (agent, 最终答案)。"""
    agent = ToolCallingAgent(tools=default_tools(base_dir=base_dir),
                             model=FakeModel(parity_replies()), **agent_kwargs)
    return agent, agent.run(parity_task())


# ---------- 1. 记忆 ----------

_task = TaskStep(task="算一下 128*46")
check("TaskStep.to_messages 是 1 条 user 消息",
      [m.dict() for m in _task.to_messages()] ==
      [{"role": "user", "content": "New task:\n算一下 128*46"}])

_sys = SystemPromptStep(system_prompt="你是助手")
check("SystemPromptStep.to_messages 是 system 消息",
      [m.dict() for m in _sys.to_messages()] ==
      [{"role": "system", "content": "你是助手"}])
check("SystemPromptStep summary 模式不进消息",
      _sys.to_messages(summary_mode=True) == [])

_plan = PlanningStep(
    model_input_messages=[ChatMessage(role="user", content="plan?")],
    model_output_message=ChatMessage(role="assistant", content="第一步算，第二步读"),
    plan="第一步算，第二步读", timing=StepTiming())
_plan_msgs = _plan.to_messages()
check("PlanningStep.to_messages 是 assistant 计划 + user 推进",
      len(_plan_msgs) == 2 and _plan_msgs[0].role == "assistant"
      and "第一步算" in _plan_msgs[0].content
      and _plan_msgs[1].role == "user"
      and "Now proceed" in _plan_msgs[1].content)
check("PlanningStep summary 模式不进消息",
      _plan.to_messages(summary_mode=True) == [])

_step = ActionStep(
    step_number=1, timing=StepTiming(),
    model_output="先算一下",
    tool_calls=[ToolCall(id="c1", name="calc", arguments='{"expression": "128*46"}')],
    observations="5888")
_step_msgs = _step.to_messages()
check("ActionStep.to_messages：assistant 带 tool_calls，tool 带 Observation",
      len(_step_msgs) == 2
      and _step_msgs[0].role == "assistant" and _step_msgs[0].content == "先算一下"
      and _step_msgs[0].tool_calls[0].name == "calc"
      and _step_msgs[1].role == "tool" and _step_msgs[1].tool_call_id == "c1"
      and _step_msgs[1].content == "Observation:\n5888")
check("ActionStep summary 模式跳过模型原文、保留 Observation",
      [m.role for m in _step.to_messages(summary_mode=True)] == ["tool"])

_err = ActionStep(step_number=1, timing=StepTiming(),
                  model_output="我不知道", error="解析失败：没有工具调用")
_err_msgs = _err.to_messages()
check("ActionStep error 变成纠正 Observation（含 retry 提示）",
      len(_err_msgs) == 2 and "Now let's retry" in _err_msgs[1].content
      and "解析失败" in _err_msgs[1].content)
_err2 = ActionStep(step_number=1, timing=StepTiming(), error="解析失败")
check("ActionStep error 没 tool_call 时走 user 角色（tool 没 id 会被端点拒）",
      [m.role for m in _err2.to_messages()] == ["user"])

check("FinalAnswerStep.to_messages 为空（循环已结束，不进消息）",
      FinalAnswerStep(output="5888").to_messages() == [])

_mem = AgentMemory("sys")
_mem.steps.append(TaskStep(task="t"))
_mem.steps.append(ActionStep(step_number=1, timing=StepTiming(),
                             model_input_messages=[ChatMessage(role="user", content="x")]))
check("AgentMemory.get_succinct_steps 去掉 model_input_messages",
      len(_mem.get_succinct_steps()) == 2
      and "model_input_messages" not in _mem.get_succinct_steps()[1]
      and "model_input_messages" in _mem.get_full_steps()[1])
_mem.reset()
check("AgentMemory.reset 清掉步骤、保留系统提示词",
      _mem.steps == [] and _mem.system_prompt.system_prompt == "sys")


# ---------- 2. 与 mini-agent 的对等验收 ----------

with tempfile.TemporaryDirectory() as tmp:
    make_readme(tmp)
    agent, out = run_parity(tmp)
    check("对等：演示任务最终答案正确（5888 + README 前 5 行）",
          "5888" in out and "line1" in out and "line5" in out)
    _actions = [s for s in agent.memory.steps if isinstance(s, ActionStep)]
    check("对等：3 个 ActionStep，工具顺序 calc → read_file → final_answer",
          len(_actions) == 3
          and [[t.name for t in s.tool_calls] for s in _actions]
          == [["calc"], ["read_file"], ["final_answer"]])
    check("对等：每步 Observation 正确（5888 / README 内容 / 最终答案）",
          _actions[0].observations == "5888"
          and "line1" in _actions[1].observations
          and "5888" in _actions[2].observations)
    check("对等：run() 返回最终答案字符串",
          isinstance(out, str) and out.startswith("128*46=5888"))

    agent2, out2 = run_parity(tmp)
    # FakeModel 默认 repeat=True：第二次跑会从头再放一遍同样的 3 条回复。
    result = agent2.run(parity_task(), return_full_result=True)
    check("对等：return_full_result=True 返回 RunResult（state success）",
          isinstance(result, RunResult) and result.state == "success"
          and "5888" in str(result.output))
    check("对等：RunResult 汇总了 token 用量、4 条步骤（Task + 3 Action）",
          result.token_usage is not None
          and result.token_usage.input_tokens > 0
          and result.token_usage.output_tokens > 0
          and len(result.steps) == 4)

    # JSONL 日志：读 agent2 这次跑的日志文件。
    log_lines = [json.loads(line) for line in open(agent2.log_path, encoding="utf-8")]
    _action_logs = [ln for ln in log_lines if isinstance(ln["step"], int)]
    check("JSONL 日志：每一步都有 ts/step/tool/arg/observation/latency_ms/raw_reply",
          all(set(ln) >= {"ts", "step", "tool", "arg", "observation",
                          "latency_ms", "raw_reply"} for ln in log_lines))
    check("JSONL 日志：动作步的工具顺序是 calc → read_file → final_answer",
          [ln["tool"] for ln in _action_logs] == ["calc", "read_file", "final_answer"])
    check("JSONL 日志：最后一条是 step=final 的最终答案",
          log_lines[-1]["step"] == "final"
          and log_lines[-1]["tool"] == "final_answer"
          and "5888" in log_lines[-1]["observation"])
    check("JSONL 日志：第 1 步 Observation 是 5888",
          _action_logs[0]["observation"] == "5888")

    # write_memory_to_messages：历史是内存的投影。
    msgs = agent2.write_memory_to_messages()
    check("write_memory_to_messages：首条 system，次条 New task",
          msgs[0].role == "system" and msgs[1].role == "user"
          and msgs[1].content.startswith("New task:"))
    check("write_memory_to_messages：assistant 消息带 tool_calls，tool 消息带 id 和 Observation",
          any(m.role == "assistant" and m.tool_calls
              and m.tool_calls[0].name == "calc" for m in msgs)
          and any(m.role == "tool" and m.tool_call_id == "c1"
                  and "5888" in (m.content or "") for m in msgs))


# ---------- 3. 规划 ----------

def _planning_agent(interval, replies, task=None):
    with tempfile.TemporaryDirectory() as tmp:
        make_readme(tmp)
        agent = ToolCallingAgent(tools=default_tools(base_dir=tmp),
                                 model=FakeModel(replies),
                                 planning_interval=interval)
        agent.run(task or parity_task())
        return agent


_plan1 = ChatMessage(role="assistant", content="计划：先算 128*46，再读 README，最后给答案")
_plan2 = ChatMessage(role="assistant", content="更新计划：算完了，读 README 再给答案")
_plan3 = ChatMessage(role="assistant", content="更新计划：读完了，给最终答案")
_replies = parity_replies()
# interval=1：3 个动作步，每步前都规划 → 3 个 PlanningStep，
# 6 次 generate 调用按顺序消费：plan1、calc、plan2、read、plan3、final。
_ag = _planning_agent(1, [_plan1, _replies[0], _plan2, _replies[1], _plan3, _replies[2]])
_plans = [s for s in _ag.memory.steps if isinstance(s, PlanningStep)]
check("规划 interval=1：3 个动作步 → 3 个 PlanningStep",
      len(_plans) == 3 and "先算" in _plans[0].plan
      and "更新计划" in _plans[1].plan and "更新计划" in _plans[2].plan)
_actions = [s for s in _ag.memory.steps if isinstance(s, ActionStep)]
check("规划步的推进消息进了第 2 步的输入（Now proceed）",
      any("Now proceed and carry out this plan." in (m.content or "")
          for m in _actions[1].model_input_messages))

# interval=2：2 个动作步（calc → final），只有第 1 步前规划。
_ag = _planning_agent(2, [_plan1, _replies[0], _replies[2]], task="算一下 128*46")
check("规划 interval=2：2 个动作步 → 只有第 1 步有 PlanningStep",
      len([s for s in _ag.memory.steps if isinstance(s, PlanningStep)]) == 1)

_ag = _planning_agent(None, parity_replies())
check("规划 interval=None：没有 PlanningStep",
      len([s for s in _ag.memory.steps if isinstance(s, PlanningStep)]) == 0)


# ---------- 4. 步数上限 ----------

with tempfile.TemporaryDirectory() as tmp:
    make_readme(tmp)
    _calc_only = tool_msg("calc", "c1", {"expression": "128*46"})
    _agent = ToolCallingAgent(
        tools=default_tools(base_dir=tmp),
        model=FakeModel([_calc_only, _calc_only,
                         ChatMessage(role="assistant", content="步数用完了，答案是 5888")]),
        max_steps=2)
    _out = _agent.run(parity_task(), return_full_result=True)
    check("步数上限：2 步没给答案，_handle_max_steps_reached 强行给答案",
          _out.output == "步数用完了，答案是 5888")
    check("步数上限：RunResult state 是 max_steps_error",
          _out.state == "max_steps_error")
    check("步数上限：记忆里有一条 error 含 'max steps' 的 ActionStep",
          any(isinstance(s, ActionStep) and s.error
              and "max steps" in s.error.lower() for s in _agent.memory.steps))


# ---------- 5. 错误处理 ----------

with tempfile.TemporaryDirectory() as tmp:
    make_readme(tmp)
    # 5a. 模型输出解析失败：纠正 Observation，不崩，循环继续。
    _agent = ToolCallingAgent(
        tools=default_tools(base_dir=tmp),
        model=FakeModel([ChatMessage(role="assistant", content="我不知道该干嘛"),
                         parity_replies()[2]]))
    _out = _agent.run("算一下 128*46")
    _steps = [s for s in _agent.memory.steps if isinstance(s, ActionStep)]
    check("解析失败：第 1 步记 error，循环继续，第 2 步给出最终答案",
          len(_steps) == 2 and _steps[0].error
          and "没有返回工具调用" in _steps[0].error
          and "5888" in _out)
    check("解析失败：第 2 步的输入里有纠正 Observation（Now let's retry）",
          any("Now let's retry" in (m.content or "")
              for m in _steps[1].model_input_messages))

    # 5b. 文本解析兜底：JSON 块。
    _agent = ToolCallingAgent(
        tools=default_tools(base_dir=tmp),
        model=FakeModel([
            ChatMessage(role="assistant", content=(
                '我先算一下：{"name": "calc", "arguments": '
                '{"expression": "128*46"}}')),
            parity_replies()[2]]))
    _out = _agent.run("算一下 128*46")
    _steps = [s for s in _agent.memory.steps if isinstance(s, ActionStep)]
    check("文本兜底 JSON 块：解析出 calc 调用，Observation 是 5888",
          _steps[0].observations == "5888" and "5888" in _out)

    # 5c. 文本解析兜底：Action: 文本格式（mini-agent 的老格式）。
    _agent = ToolCallingAgent(
        tools=default_tools(base_dir=tmp),
        model=FakeModel([
            ChatMessage(role="assistant", content="Thought: 先算一下\nAction: calc(128*46)"),
            parity_replies()[2]]))
    _out = _agent.run("算一下 128*46")
    _steps = [s for s in _agent.memory.steps if isinstance(s, ActionStep)]
    check("文本兜底 Action 格式：解析出 calc 调用，Observation 是 5888",
          _steps[0].observations == "5888" and "5888" in _out)

    # 5d. 未知工具 → Observation，不崩。
    _agent = ToolCallingAgent(
        tools=default_tools(base_dir=tmp),
        model=FakeModel([tool_msg("nope", "c1", {"x": "1"}),
                         parity_replies()[2]]))
    _out = _agent.run("算一下 128*46")
    _steps = [s for s in _agent.memory.steps if isinstance(s, ActionStep)]
    check("未知工具：Observation 是错误说明，循环继续给出最终答案",
          "Error executing tool 'nope'" in _steps[0].observations
          and "5888" in _out)

    # 5e. 参数校验失败 → Observation，不崩。
    _agent = ToolCallingAgent(
        tools=default_tools(base_dir=tmp),
        model=FakeModel([tool_msg("calc", "c1", {"expression": 123}),
                         parity_replies()[2]]))
    _out = _agent.run("算一下 128*46")
    _steps = [s for s in _agent.memory.steps if isinstance(s, ActionStep)]
    check("参数校验失败：Observation 是错误说明，循环继续",
          "Error executing tool 'calc'" in _steps[0].observations
          and "5888" in _out)

    # 5f. 一步多调：两个工具调用都执行，Observation 拼起来。
    _agent = ToolCallingAgent(
        tools=default_tools(base_dir=tmp),
        model=FakeModel([
            ChatMessage(role="assistant", tool_calls=[
                ToolCall(id="c1", name="calc", arguments='{"expression": "128*46"}'),
                ToolCall(id="c2", name="read_file", arguments='{"path": "README.md"}')]),
            parity_replies()[2]]))
    _out = _agent.run(parity_task())
    _steps = [s for s in _agent.memory.steps if isinstance(s, ActionStep)]
    check("一步多调：两个工具都执行，Observation 拼在一起",
          len(_steps) == 2 and len(_steps[0].tool_calls) == 2
          and "5888" in _steps[0].observations
          and "line1" in _steps[0].observations)

    # 5g. final_answer 不能和其他调用同批。
    _agent = ToolCallingAgent(
        tools=default_tools(base_dir=tmp),
        model=FakeModel([
            ChatMessage(role="assistant", tool_calls=[
                ToolCall(id="c1", name="calc", arguments='{"expression": "128*46"}'),
                ToolCall(id="c2", name="final_answer", arguments='{"answer": "5888"}')]),
            parity_replies()[2]]))
    _out = _agent.run("算一下 128*46")
    _steps = [s for s in _agent.memory.steps if isinstance(s, ActionStep)]
    check("final_answer 同批有其他调用：不结束，Observation 报错，下一步才结束",
          len(_steps) == 2
          and "final answer tool call" in _steps[0].observations
          and "5888" in _out)


# ---------- 6. 循环本身 ----------

# 6a. final_answer 没注册时自动补上（上游 setdefault 同款）。
_reg = ToolRegistry()
_reg.register(_calc_tool)
_agent = ToolCallingAgent(tools=_reg, model=FakeModel("x"))
check("final_answer 没注册时自动补上",
      "final_answer" in _agent.tools.names())


class CapturingModel(FakeModel):
    """记下第一次 generate 收到的 tools_to_call_from 的 FakeModel。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen_tools = None

    def generate(self, messages, stop_sequences=None, tools_to_call_from=None, **kwargs):
        if self.seen_tools is None:
            self.seen_tools = tools_to_call_from
        return super().generate(messages, stop_sequences, tools_to_call_from, **kwargs)


with tempfile.TemporaryDirectory() as tmp:
    make_readme(tmp)
    _cap = CapturingModel(parity_replies())
    _agent = ToolCallingAgent(tools=default_tools(base_dir=tmp), model=_cap)
    _agent.run(parity_task())
    _names = {t["function"]["name"] for t in (_cap.seen_tools or [])}
    check("模型收到的 tools 参数是注册表的 OpenAI schema（5 个工具）",
          _names == {"calc", "read_file", "write_file", "web_search",
                     "final_answer"})


class BoomingModel(Model):
    """generate 直接抛异常的模型（模拟模型层挂掉）。"""

    def generate(self, messages, stop_sequences=None, tools_to_call_from=None, **kwargs):
        raise RuntimeError("boom")


with tempfile.TemporaryDirectory() as tmp:
    _agent = ToolCallingAgent(tools=default_tools(base_dir=tmp),
                              model=BoomingModel())
    raises("生成失败：AgentGenerationError 直接抛（不吞、不记 error 继续跑）",
           AgentGenerationError, lambda: _agent.run("算一下 128*46"))

    # run() 会重置记忆：跑两次，第二次的记忆里只有自己的 TaskStep。
    _agent = ToolCallingAgent(tools=default_tools(base_dir=tmp),
                              model=FakeModel(parity_replies()))
    _agent.run(parity_task())
    _agent.run("再算一次 1+1")
    check("run() 重置记忆：第二次跑完记忆里只有 1 个 TaskStep",
          len([s for s in _agent.memory.steps if isinstance(s, TaskStep)]) == 1)

    # 步骤回调：每步结束时触发。
    _seen = []
    _agent = ToolCallingAgent(
        tools=default_tools(base_dir=tmp),
        model=FakeModel(parity_replies()),
        step_callbacks={ActionStep: [lambda step: _seen.append(step.step_number)]})
    _agent.run(parity_task())
    check("步骤回调：3 个 ActionStep 各触发一次",
          _seen == [1, 2, 3])

    # max_steps 可按次覆盖。
    _agent = ToolCallingAgent(
        tools=default_tools(base_dir=tmp),
        model=FakeModel([parity_replies()[0],
                         ChatMessage(role="assistant", content="只给 1 步，答案是 5888")]),
        max_steps=8)
    _out = _agent.run(parity_task(), max_steps=1)
    check("max_steps 按次覆盖为 1：1 步后走强行给答案",
          _out == "只给 1 步，答案是 5888")


print(f"\n{PASSED}/{PASSED + FAILED} passed.")
sys.exit(0 if FAILED == 0 else 1)
