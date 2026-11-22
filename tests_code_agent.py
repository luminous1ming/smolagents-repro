#!/usr/bin/env python3
"""Phase 4 自测：CodeAgent + 本地 Python 执行器（不联网）。

Run from the repro directory:
    .venv/bin/python tests_code_agent.py

测七件事：
  1. 代码块抽取：```python 围栏、多块拼接、```py 兜底、裸代码兜底、
     没代码抛 AgentParsingError（提到 final answer 时给提示）；
  2. fix_final_answer_code：既调用又赋值时重命名赋值，只赋值不动；
  3. 执行器基础：算术 + print 收集、最后一条表达式的值、有状态命名空间、
     工具进命名空间、final_answer 终止；
  4. 执行器安全策略（Phase 0 的教训）：白名单导入（11 个模块）、
     open()/import os 禁止、dunder 访问拦截（__builtins__、
     ().__class__.__base__、eval("open")、getattr 拼字符串）；
  5. 执行器上限：操作数上限（小上限跑循环必触发）、墙钟超时；
  6. CodeAgent 循环：一步完成任务、执行失败后纠正 Observation 自修复、
     每轮重置执行器状态、系统提示词里工具渲染成 def（含 final_answer
     的 `-> any` 写法，Phase 2 的待定项——提示词侧先锁定）；
  7. 验收标准：同样 3 个多工具任务，CodeAgent 的步数比 Phase 3 的
     ToolCallingAgent 少（FakeModel 脚本，步数是实测数）。
"""
import os
import sys
import tempfile

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from agents import (AgentExecutionError, AgentParsingError, CodeAgent,
                    ToolCallingAgent, extract_code_blocks, parse_code_blobs)
from default_tools import default_tools
from executor import (BASE_BUILTIN_MODULES, InterpreterError,
                      LocalPythonExecutor, fix_final_answer_code)
from memory import ActionStep
from models import ChatMessage, FakeModel, ToolCall
from tools import tool

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


def _code_reply(code: str) -> str:
    """FakeModel 给 CodeAgent 的回复：Thought + 一个代码块。"""
    return f"Thought: 按计划执行。\n```python\n{code}\n```"


def _tc_reply(name: str, arguments: str) -> ChatMessage:
    """FakeModel 给 ToolCallingAgent 的回复：一条结构化 tool_call。"""
    return ChatMessage(role="assistant", content="",
                       tool_calls=[ToolCall(id="call_1", name=name,
                                            arguments=arguments)])


def _action_step_count(result) -> int:
    """RunResult 里 ActionStep 的个数（= 实际跑的动作步数）。"""
    return len([s for s in result.steps if "step_number" in s])


# ---------- 1. 代码块抽取 ----------

check("extract fenced python block",
      extract_code_blocks("Thought: x\n```python\nx = 1\n```\n") == "x = 1")
check("extract multiple blocks joined",
      extract_code_blocks("```python\na = 1\n```\n```python\nb = 2\n```")
      == "a = 1\n\nb = 2")
check("extract none -> None",
      extract_code_blocks("no code here") is None)
check("parse py fence fallback",
      parse_code_blobs("```py\nx = 2\n```") == "x = 2")
check("parse bare code fallback",
      parse_code_blobs("x = 1 + 2") == "x = 1 + 2")
raises("parse no code -> AgentParsingError",
       AgentParsingError, lambda: parse_code_blobs("hello world"))
try:
    parse_code_blobs("I want the final answer now, no code")
    check("parse final-answer hint", False)
except AgentParsingError as e:
    check("parse final-answer hint", "final_answer" in str(e))

# ---------- 2. fix_final_answer_code ----------

check("fix renames assignment when called",
      fix_final_answer_code('final_answer = 5\nfinal_answer("x")')
      == 'final_answer_variable = 5\nfinal_answer("x")')
check("fix renames variable use when called",
      fix_final_answer_code('final_answer = 5\nx = final_answer\nfinal_answer("y")')
      == 'final_answer_variable = 5\nx = final_answer_variable\nfinal_answer("y")')
check("fix leaves assignment alone when not called",
      fix_final_answer_code('final_answer = 5') == 'final_answer = 5')
check("fix leaves plain call alone",
      fix_final_answer_code('final_answer("x")') == 'final_answer("x")')

# ---------- 3. 执行器基础 ----------

ex = LocalPythonExecutor()
out = ex("x = 2 + 3\nprint('x is', x)\nx * 10")
check("executor arithmetic + print + last expr",
      out.output == 50 and "x is 5" in out.logs and not out.is_final_answer)

ex("a = 7")
check("executor stateful namespace across calls",
      ex("a * 6").output == 42)

ex("import math")
check("executor import persists across calls",
      ex("math.sqrt(16)").output == 4.0)

check("executor 11-module allow-list",
      sorted(BASE_BUILTIN_MODULES) == sorted(["collections", "datetime", "itertools",
                                              "math", "queue", "random", "re",
                                              "stat", "statistics", "time",
                                              "unicodedata"]))

# ---------- 4. 执行器安全策略 ----------

raises("executor import os forbidden",
       InterpreterError, lambda: ex("import os"))
raises("executor from os import forbidden",
       InterpreterError, lambda: ex("from os import path"))
raises("executor import sys forbidden",
       InterpreterError, lambda: ex("import sys"))
raises("executor import subprocess forbidden",
       InterpreterError, lambda: ex("import subprocess"))
check("executor import collections allowed",
      ex("import collections\nlen(collections.Counter([1, 1, 2]))").output == 2)
raises("executor open() forbidden",
       InterpreterError, lambda: ex('open("x")'))
raises("executor eval() forbidden",
       InterpreterError, lambda: ex('eval("1+1")'))
raises("executor __builtins__ lookup forbidden",
       InterpreterError, lambda: ex('__builtins__["open"]'))
raises("executor dunder chain forbidden",
       InterpreterError, lambda: ex("().__class__.__base__.__subclasses__()"))
raises("executor eval('open') forbidden",
       InterpreterError, lambda: ex('eval("open")'))
raises("executor dunder attr forbidden",
       InterpreterError, lambda: ex("x = 1\nx.__class__"))
raises("executor getattr dunder-string forbidden",
       InterpreterError, lambda: ex('getattr((), "__cl" + "ass__")'))
raises("executor exec() forbidden",
       InterpreterError, lambda: ex('exec("x = 1")'))

with tempfile.TemporaryDirectory() as tmp:
    ex_tools = LocalPythonExecutor()
    ex_tools.send_tools(default_tools(base_dir=tmp))
    check("executor tool callable in namespace",
          ex_tools('calc(expression="6*7")').output == "42")
    out = ex_tools('final_answer("done")')
    check("executor final_answer terminates",
          out.is_final_answer and out.output == "done")
    raises("executor cannot assign to tool name",
           InterpreterError, lambda: ex_tools("calc = 5"))
    raises("executor cannot delete tool name",
           InterpreterError, lambda: ex_tools("del calc"))

# ---------- 5. 执行器上限 ----------

ex_cap = LocalPythonExecutor(max_operations=20)
raises("executor op cap trips on loop",
       InterpreterError,
       lambda: ex_cap("total = 0\nfor i in range(100):\n    total += i"))
try:
    ex_cap("total = 0\nfor i in range(100):\n    total += i")
    check("executor op cap message", False)
except InterpreterError as e:
    check("executor op cap message", "max number of operations" in str(e))

ex_timeout = LocalPythonExecutor(timeout_seconds=1)
try:
    ex_timeout("import time\ntime.sleep(3)")
    check("executor wall-clock timeout", False)
except InterpreterError as e:
    check("executor wall-clock timeout", "timed out" in str(e))

# ---------- 6. CodeAgent 循环 ----------

with tempfile.TemporaryDirectory() as tmp:
    with open(os.path.join(tmp, "nums.txt"), "w", encoding="utf-8") as f:
        f.write("10\n20\n35\n")

    # 6a. 一步完成：calc + final_answer 在一个代码块里。
    agent = CodeAgent(tools=default_tools(base_dir=tmp),
                      model=FakeModel([_code_reply(
                          'result = calc(expression="128*46")\nfinal_answer(result)')]))
    result = agent.run("算一下 128*46", return_full_result=True)
    check("code agent one-step calc",
          str(result.output) == "5888" and _action_step_count(result) == 1)
    step0 = result.steps[1]  # steps[0] 是 TaskStep
    check("code agent records code_action + python_interpreter call",
          "calc(" in (step0.get("code_action") or "")
          and step0.get("tool_calls", [{}])[0].get("function", {}).get("name")
          == "python_interpreter")
    check("code agent observation format",
          "Execution logs:" in (step0.get("observations") or "")
          and "Last output from code snippet:" in (step0.get("observations") or ""))

    # 6b. 自修复：第 1 步 import os 被禁，第 2 步改用工具，循环继续。
    agent2 = CodeAgent(tools=default_tools(base_dir=tmp), model=FakeModel([
        _code_reply("import os\nprint(os.getcwd())"),
        _code_reply('result = calc(expression="128*46")\nfinal_answer(result)'),
    ]))
    result2 = agent2.run("算一下 128*46", return_full_result=True)
    check("code agent self-repair after forbidden import",
          str(result2.output) == "5888" and _action_step_count(result2) == 2)
    err_step = result2.steps[1]
    check("code agent step 1 error recorded",
          "Import of os is not allowed" in (err_step.get("error") or ""))
    step2_inputs = " ".join(
        (m.get("content") or "") for m in result2.steps[2].get("model_input_messages") or [])
    check("code agent correction observation fed back",
          "Now let's retry" in step2_inputs
          and "Import of os is not allowed" in step2_inputs)

    # 6c. 每轮重置执行器状态：上一轮的变量下一轮看不到。
    agent3 = CodeAgent(tools=default_tools(base_dir=tmp), model=FakeModel([
        _code_reply("v = 123\nfinal_answer(v)"),
        _code_reply("final_answer(v)"),
        "forced answer",
    ], repeat=False))
    r3a = agent3.run("定义 v 并返回它", return_full_result=True)
    check("code agent state persists within a run",
          str(r3a.output) == "123")
    r3b = agent3.run("直接返回上一轮的 v", max_steps=1, return_full_result=True)
    check("code agent state reset between runs",
          "not defined" in (r3b.steps[1].get("error") or ""))

    # 6d. 系统提示词：工具渲染成 def、final_answer 是 `-> any`、
    # 白名单列出来、open() 禁止写明。
    prompt = agent.system_prompt
    check("code prompt renders tools as defs",
          "def calc(expression: string) -> string:" in prompt
          and "def read_file(path: string) -> string:" in prompt)
    check("code prompt final_answer renders -> any (Phase 2 open item)",
          "def final_answer(answer: any) -> any:" in prompt)
    check("code prompt lists authorized imports",
          str(sorted(BASE_BUILTIN_MODULES)) in prompt)
    check("code prompt forbids open()/import os",
          "Never use open() or import os" in prompt)

    # 6e. 代码步的历史必须是纯文本（上游同款）：assistant 消息不挂
    # tool_calls，Observation 和纠正消息都走 user 角色——CodeAgent
    # 调模型时不传 tools 参数，历史里一旦出现 OpenAI 形状的
    # tool_calls，Gemini 的 OpenAI 兼容端点会在第二步报 400
    # （实测：live check 两次都在第二步 INVALID_ARGUMENT）。
    from memory import StepTiming as _StepTiming
    code_step = ActionStep(
        step_number=1, timing=_StepTiming(),
        model_output="Thought: x\n```python\nx = 1\n```",
        tool_calls=[ToolCall(id="call_1", name="python_interpreter",
                             arguments="x = 1")],
        observations="Execution logs:\n\nLast output from code snippet:\n1",
        error="Import of os is not allowed",
        code_action="x = 1")
    code_msgs = code_step.to_messages()
    check("code step history is plain text (no tool_calls on assistant)",
          code_msgs[0].role == "assistant" and code_msgs[0].tool_calls is None)
    check("code step observation is a user message",
          code_msgs[1].role == "user"
          and code_msgs[1].content.startswith("Observation:"))
    check("code step error is a user message",
          code_msgs[2].role == "user"
          and "Now let's retry" in (code_msgs[2].content or ""))
    # 对照：ToolCallingAgent 的步（code_action 为空）保持 OpenAI 形状。
    tc_step = ActionStep(
        step_number=1, timing=_StepTiming(),
        model_output="",
        tool_calls=[ToolCall(id="call_1", name="calc",
                             arguments='{"expression": "1+1"}')],
        observations="2")
    tc_msgs = tc_step.to_messages()
    check("tool-calling step keeps OpenAI tool_calls shape",
          tc_msgs[0].tool_calls is not None
          and tc_msgs[1].role == "tool"
          and tc_msgs[1].tool_call_id == "call_1")

    # ---------- 7. 步数对比：CodeAgent vs ToolCallingAgent ----------
    from tools import tool as _tool_decorator

    @_tool_decorator
    def lookup_population(city: str) -> str:
        """返回某城市的人口（百万），代替联网搜索做离线对比。

        Args:
            city: 城市名，例如 "Tokyo"。

        Returns:
            人口（百万）的字符串。
        """
        return {"Tokyo": "14.18", "Paris": "2.1"}[city]

    def _tools_with_lookup():
        registry = default_tools(base_dir=tmp)
        registry.register(lookup_population)
        return registry

    def _run_both(task, code_replies, tc_replies):
        """同一任务走两个 agent，返回 (code_steps, tc_steps, code_out, tc_out)。"""
        ca = CodeAgent(tools=_tools_with_lookup(), model=FakeModel(code_replies))
        rc = ca.run(task, return_full_result=True)
        ta = ToolCallingAgent(tools=_tools_with_lookup(), model=FakeModel(tc_replies))
        rt = ta.run(task, return_full_result=True)
        return (_action_step_count(rc), _action_step_count(rt),
                str(rc.output), str(rt.output))

    # 任务 A：calc + 读文件 + 求和（CodeAgent 1 步 vs ToolCallingAgent 4 步）。
    a_steps = _run_both(
        "计算 128*46，再读 nums.txt，把两个结果加起来",
        [_code_reply(
            'a = calc(expression="128*46")\n'
            'text = read_file(path="nums.txt")\n'
            'total = int(a) + sum(int(x) for x in text.split())\n'
            'final_answer(total)')],
        [_tc_reply("calc", '{"expression": "128*46"}'),
         _tc_reply("read_file", '{"path": "nums.txt"}'),
         _tc_reply("calc", '{"expression": "5888+65"}'),
         _tc_reply("final_answer", '{"answer": "5953"}')])
    check("compare task A: code 1 step vs tool-calling 4 steps",
          a_steps == (1, 4, "5953", "5953"))

    # 任务 B：读文件 + 循环平方求和（CodeAgent 1 步 vs ToolCallingAgent 3 步）。
    b_steps = _run_both(
        "读 nums.txt，把每行数字平方后求和",
        [_code_reply(
            'text = read_file(path="nums.txt")\n'
            'total = 0\n'
            'for line in text.split():\n'
            '    total += int(line) ** 2\n'
            'final_answer(total)')],
        [_tc_reply("read_file", '{"path": "nums.txt"}'),
         _tc_reply("calc", '{"expression": "100+400+1225"}'),
         _tc_reply("final_answer", '{"answer": "1725"}')])
    check("compare task B: code 1 step vs tool-calling 3 steps",
          b_steps == (1, 3, "1725", "1725"))

    # 任务 C：查数 + 计算（CodeAgent 1 步 vs ToolCallingAgent 3 步）。
    c_steps = _run_both(
        "东京人口（百万）乘以 10 是多少？",
        [_code_reply(
            'pop = lookup_population(city="Tokyo")\n'
            'print(pop)\n'
            'final_answer(float(pop) * 10)')],
        [_tc_reply("lookup_population", '{"city": "Tokyo"}'),
         _tc_reply("calc", '{"expression": "14.18*10"}'),
         _tc_reply("final_answer", '{"answer": "141.8"}')])
    check("compare task C: code 1 step vs tool-calling 3 steps",
          c_steps == (1, 3, "141.8", "141.8"))

    print(f"\n步数对比汇总：任务 A {a_steps[0]} vs {a_steps[1]} 步；"
          f"任务 B {b_steps[0]} vs {b_steps[1]} 步；"
          f"任务 C {c_steps[0]} vs {c_steps[1]} 步"
          "（CodeAgent vs ToolCallingAgent）")

print(f"\n{PASSED}/{PASSED + FAILED} passed.")
