#!/usr/bin/env python3
"""Phase 5 自测：managed_agents + orchestrator-worker 研究流程（不联网）。

Run from the repro directory:
    .venv/bin/python tests_multi_agent.py

测六件事：
  1. managed_agents 基础：子智能体包成工具、调一次拿回报告；
     缺 name/description、重名（工具/自己/子智能体之间）都报错；
  2. ToolCallingAgent + managed：lead 调 researcher 工具拿报告；
     上下文隔离——researcher 内部的工具调用不进 lead 的历史；
  3. CodeAgent + managed：lead 在代码里调 researcher，执行器里
     直接当 Python 函数用；
  4. orchestrator-worker 端到端（FakeModel）：规划 → 3 个并行
     researcher → 综合 → 引用核查；每条引用都能追溯到 researcher
     交过的来源；子智能体上下文互相隔离；综合只用子智能体输出；
  5. CitationAgent 单元：来源抽取、去重、无效引用删除、不编来源；
  6. plan 解析：编号/项目符号/兜底。
"""
import json
import os
import re
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from agents import CodeAgent, ManagedAgentTool, MultiStepAgent, ToolCallingAgent
from default_tools import FinalAnswerTool, calc as _calc_tool
from default_tools import default_tools
from models import ChatMessage, FakeModel, Model, ToolCall
from research import (CitationAgent, ResearchOrchestrator, Source,
                      parse_subquestions)
from tools import ToolRegistry, tool

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
    except Exception as e:  # noqa: BLE001 - 别的异常就是测挂了
        check(f"{name}（抛错了 {type(e).__name__}，不是 {exc_type.__name__}）", False)
    else:
        check(f"{name}（没抛错）", False)


# ---------- 脚本回复小帮手（tests_agents.py 的同款） ----------


def _tool_reply(name, arguments, content=None, call_id="c1"):
    return ChatMessage(role="assistant", content=content or f"调用 {name} 工具",
                       tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)])


def _final_answer_reply(answer, call_id="c1"):
    return _tool_reply("final_answer",
                       json.dumps({"answer": answer}, ensure_ascii=False),
                       call_id=call_id)


def _memory_text(agent) -> str:
    """一个 agent 全部记忆拼成的文本（断言隔离时用）：
    正文 + 工具调用的参数——工具参数不在 content 里，只拼正文会漏掉。"""
    bits = []
    for m in agent.write_memory_to_messages():
        bits.append(m.content or "")
        for tc in m.tool_calls or []:
            bits.append(tc.name)
            bits.append(tc.arguments if isinstance(tc.arguments, str)
                        else json.dumps(tc.arguments, ensure_ascii=False))
    return "\n".join(bits)


# ---------- 假搜索工具（researcher 在测试里用的桩，不联网） ----------


@tool
def fake_search(query: str) -> str:
    """假搜索：返回写死的文本，不联网。

    Args:
        query: 搜索关键词。

    Returns:
        写死的搜索结果文本。
    """
    return f"关于“{query}”的资料：示例文本。"


def _stub_tools():
    """researcher 在测试里的工具箱：假搜索 + calc + final_answer。"""
    registry = ToolRegistry()
    registry.register(fake_search)
    registry.register(_calc_tool)
    registry.register(FinalAnswerTool())
    return registry


# ---------- §1 managed_agents 基础 ----------

_r1 = ToolCallingAgent(tools=default_tools(), model=FakeModel([
    _final_answer_reply("calc result is 5888")]),
    name="researcher_1", description="算数的研究员")
_r1_tool = ManagedAgentTool(_r1)
check("1a 包成工具：name/description/inputs 跟着子智能体走",
      _r1_tool.name == "researcher_1"
      and "算数" in _r1_tool.description
      and set(_r1_tool.inputs) == {"task", "additional_args"}
      and _r1_tool.output_type == "string")
_reg = ToolRegistry([_r1_tool])
_report = _reg.call("researcher_1", {"task": "算一下"})
check("1b 调一次工具：拿回子智能体的报告（带名字和答案）",
      "Report from researcher_1" in _report and "calc result is 5888" in _report)
check("1c 报告里没有子智能体内部的步骤细节",
      "tool_calls" not in _report and "ActionStep" not in _report)

_no_desc = ToolCallingAgent(tools=default_tools(), model=FakeModel(["x"]),
                            name="sub_no_desc")
raises("1d 子智能体缺 description：登记时报错", ValueError, lambda: ToolCallingAgent(
    tools=default_tools(), model=FakeModel(["x"]),
    managed_agents=[_no_desc]))
_calc_named = ToolCallingAgent(tools=default_tools(), model=FakeModel(["x"]),
                               name="calc", description="和工具重名")
raises("1e 子智能体和工具重名：登记时报错", ValueError, lambda: ToolCallingAgent(
    tools=default_tools(), model=FakeModel(["x"]),
    managed_agents=[_calc_named]))
_self_named = ToolCallingAgent(tools=default_tools(), model=FakeModel(["x"]),
                               name="lead", description="自己")
_dup = ToolCallingAgent(tools=default_tools(), model=FakeModel(["x"]),
                        name="lead", description="和自己重名")
raises("1f 子智能体和自己重名：登记时报错", ValueError, lambda: ToolCallingAgent(
    tools=default_tools(), model=FakeModel(["x"]), name="lead",
    description="自己", managed_agents=[_dup]))
_dup_a = ToolCallingAgent(tools=default_tools(), model=FakeModel(["x"]),
                          name="dup", description="第一个")
_dup_b = ToolCallingAgent(tools=default_tools(), model=FakeModel(["x"]),
                          name="dup", description="第二个")
raises("1g 两个子智能体重名：登记时报错", ValueError, lambda: ToolCallingAgent(
    tools=default_tools(), model=FakeModel(["x"]),
    managed_agents=[_dup_a, _dup_b]))

# ---------- §2 ToolCallingAgent + managed ----------


def _make_calc_researcher():
    """算 987*65 的 researcher：内部先调 calc，再 final_answer。

    注意：算式故意不用 128*46——calc 工具的描述里就拿 128*46 当例子，
    lead 的系统提示词里本来就有它；用别的算式，§2c 的隔离断言才有意义。
    """
    return ToolCallingAgent(
        tools=default_tools(), model=FakeModel([
            _tool_reply("calc", '{"expression": "987*65"}'),
            _final_answer_reply("calc result is 64155")]),
        name="researcher_1", description="算数的研究员")


_r2 = _make_calc_researcher()
_lead2 = ToolCallingAgent(
    tools=default_tools(), model=FakeModel([
        _tool_reply("researcher_1", '{"task": "请计算题目里的算式"}'),
        _final_answer_reply("简报：研究员算出来了")]),
    name="lead", description="主智能体", managed_agents=[_r2])
_out2 = _lead2.run("研究一下这个算式", return_full_result=True)
_lead2_obs = "\n".join(s.get("observations") or ""
                       for s in _out2.steps if "observations" in s)
check("2a lead 调 researcher 工具：最终答案是脚本里的简报",
      _out2.output == "简报：研究员算出来了")
check("2b lead 的 Observation 里有 researcher 的报告（含名字和 64155）",
      "Report from researcher_1" in _lead2_obs and "64155" in _lead2_obs)
check("2c 上下文隔离：lead 的历史里没有 researcher 内部的 calc 参数",
      "987*65" not in _memory_text(_lead2))
check("2d 上下文隔离：researcher 自己的历史里有 calc 调用",
      "987*65" in _memory_text(_r2))
check("2e 记忆是两个对象：researcher 的步骤没进 lead 的记忆",
      _r2.memory is not _lead2.memory
      and any("请计算题目里的算式" in (s.get("task") or "")
              for s in _r2.memory.get_full_steps()))
check("2f 系统提示词里列出了 researcher（模型能看到它）",
      "researcher_1" in _lead2.system_prompt
      and "算数的研究员" in _lead2.system_prompt)
check("2g 模型收到的 tools 参数里有 researcher",
      any(t["function"]["name"] == "researcher_1"
          for t in _lead2._all_tools_openai()))

# ---------- §3 CodeAgent + managed ----------

_r3 = _make_calc_researcher()
_lead3 = CodeAgent(
    tools=default_tools(), model=FakeModel([
        "Thought: 先让研究员算一下。\n```python\n"
        "report = researcher_1(task=\"请计算题目里的算式\")\nprint(report)\n```",
        "Thought: 报告拿到了，返回它。\n```python\nfinal_answer(report)\n```",
    ]),
    name="lead", description="主智能体", managed_agents=[_r3])
_out3 = _lead3.run("研究一下这个算式")
check("3a CodeAgent 在代码里调 researcher：最终答案是 researcher 的报告",
      "Report from researcher_1" in str(_out3) and "64155" in str(_out3))
check("3b CodeAgent 提示词里 researcher 是 Python 函数定义",
      "def researcher_1(task: string" in _lead3.system_prompt)
check("3c 执行器里有 researcher 这个名字（代码里能直接调）",
      "researcher_1" in _lead3.python_executor._tools)

# ---------- §4 orchestrator-worker 端到端 ----------

_FINDINGS = [
    ("发现一：苹果公司 2024 年营收 3910 亿美元。\nSources:\n"
     "- 苹果 2024 财报 — https://example.com/apple2024\n"
     "- 苹果公司概况 — https://example.com/apple"),
    ("发现二：苹果公司 2024 年 iPhone 收入 2010 亿美元。\nSources:\n"
     "- 苹果 2024 财报 — https://example.com/apple2024\n"
     "- iPhone 销售数据 — https://example.com/iphone"),
    ("发现三：苹果公司员工约 16.4 万人。\nSources:\n"
     "- 苹果员工数 — https://example.com/headcount"),
]
_PLAN = "1. 子问题一\n2. 子问题二\n3. 子问题三"
_DRAFT = ("- 苹果公司 2024 年营收 3910 亿美元 [S1]。\n"
          "- iPhone 收入 2010 亿美元 [S3]。\n"
          "- 员工约 16.4 万人 [S4]。")


def _researcher_model(findings: str):
    """一个 researcher 的脚本模型：先假搜索，再交 findings 当最终答案。"""
    return FakeModel([
        _tool_reply("fake_search", '{"query": "test"}'),
        _final_answer_reply(findings),
    ])


def _model_factory(models):
    """按顺序发模型：每建一个 researcher 发一个（各人有自己的脚本）。"""
    it = iter(models)
    return lambda: next(it)


def _make_orch(lead_replies, researcher_models):
    return ResearchOrchestrator(
        model=FakeModel(lead_replies),
        researcher_model_factory=_model_factory(researcher_models),
        tools_factory=_stub_tools)


_orch = _make_orch([_PLAN, _DRAFT],
                   [_researcher_model(f) for f in _FINDINGS])
_brief = _orch.research("苹果公司 2024 年怎么样？", n_subquestions=3)
check("4a 规划拆出 3 个子问题",
      _brief.sub_questions == ["子问题一", "子问题二", "子问题三"])
check("4b 3 个 researcher 各交一份结果，名字和问题一一对应",
      len(_brief.results) == 3
      and [r.agent_name for r in _brief.results]
      == ["researcher_1", "researcher_2", "researcher_3"]
      and [r.question for r in _brief.results] == _brief.sub_questions)
check("4c 来源按 URL 去重：两个 researcher 交了同一条，只剩 4 条",
      len(_brief.sources) == 4
      and len({s.url for s in _brief.sources}) == 4)
check("4d 简报正文引用了 S1/S3/S4，没被引用的 S2 不进正文",
      "[S1]" in _brief.brief and "[S3]" in _brief.brief
      and "[S4]" in _brief.brief and "[S2]" not in _brief.brief)
check("4e 每条引用都可追溯：正文的 [S#] 都在来源列表里",
      all(1 <= int(n) <= len(_brief.sources)
          for n in re.findall(r"\[S(\d+)\]",
                              _brief.brief.split("## Sources")[0]))
      and "## Sources" in _brief.brief)
check("4f Sources 节只列正文引用到的来源（S1/S3/S4，不含 S2 的标题）",
      "苹果公司概况" not in _brief.brief.split("## Sources")[1]
      and "苹果 2024 财报" in _brief.brief.split("## Sources")[1])
check("4g 干净流程：没有无效引用被删",
      _brief.dropped_citations == [])
check("4h 上下文隔离：各 researcher 的记忆是独立对象",
      _orch.researchers[0].memory is not _orch.researchers[1].memory
      and _orch.researchers[1].memory is not _orch.researchers[2].memory)
check("4i 上下文隔离：每个 researcher 的任务里只有自己的子问题",
      all(_brief.sub_questions[i] in _memory_text(r)
          and all(_brief.sub_questions[j] not in _memory_text(r)
                  for j in range(3) if j != i)
          for i, r in enumerate(_orch.researchers)))
check("4j lead 拥有 3 个 managed researcher",
      len(_orch.lead.managed_agents) == 3
      and set(_orch.lead.managed_agents)
      == {"researcher_1", "researcher_2", "researcher_3"})

# 编出来的引用进不了正文：综合稿里 [S99] + 自带 Sources 节写了假链接。
_BOGUS_DRAFT = ("- 苹果公司 2024 年营收 3910 亿美元 [S1]。\n"
                "- 编出来的说法 [S99]。\n\n"
                "## Sources\n- 编出来的来源 — https://invented.example/secret")
_orch_bogus = _make_orch([_PLAN, _BOGUS_DRAFT],
                         [_researcher_model(f) for f in _FINDINGS])
_brief_bogus = _orch_bogus.research("苹果公司 2024 年怎么样？",
                                    n_subquestions=3)
check("4k 综合只用子智能体输出：编出来的链接进不了最终简报",
      "invented.example" not in _brief_bogus.brief)
check("4l 无效引用 [S99] 被删掉并记下来",
      "[S99]" not in _brief_bogus.brief
      and _brief_bogus.dropped_citations == ["S99"])
check("4m 有效引用 [S1] 不受影响，仍在正文和 Sources 节里",
      "[S1]" in _brief_bogus.brief
      and "https://example.com/apple2024" in _brief_bogus.brief)


class _FailingModel(Model):
    """一调就抛错的模型：测 researcher 失败隔离。"""

    def generate(self, messages, stop_sequences=None,
                 tools_to_call_from=None, **kwargs):
        raise RuntimeError("boom")


_orch_fail = ResearchOrchestrator(
    model=FakeModel([_PLAN, _DRAFT]),
    researcher_model_factory=_model_factory([
        _researcher_model(_FINDINGS[0]), _FailingModel(),
        _researcher_model(_FINDINGS[2])]),
    tools_factory=_stub_tools)
_brief_fail = _orch_fail.research("苹果公司 2024 年怎么样？",
                                  n_subquestions=3)
check("4n researcher 失败不炸整轮：其他两路照常出简报",
      len(_brief_fail.results) == 3
      and "failed" in _brief_fail.results[1].answer
      and "[S1]" in _brief_fail.brief)
check("4o 失败的 researcher 不交来源",
      _brief_fail.results[1].sources == [])

# 并行数上限：规划出 8 个子问题，max_researchers=5 只派 5 个。
_PLAN8 = "\n".join(f"{i + 1}. 子问题{i + 1}" for i in range(8))
_orch_cap = ResearchOrchestrator(
    model=FakeModel([_PLAN8, "- 要点 [S1]。"]),
    researcher_model_factory=lambda: _researcher_model(_FINDINGS[0]),
    tools_factory=_stub_tools, max_researchers=5)
_brief_cap = _orch_cap.research("大主题", n_subquestions=8)
check("4p 并行数上限：8 个子问题只派 5 个 researcher",
      len(_brief_cap.results) == 5
      and len(_brief_cap.sub_questions) == 5)

# ---------- §5 CitationAgent 单元 ----------

_src_text = ("摘要：苹果营收 3910 亿。\nSources:\n"
             "- 苹果 2024 财报 — https://example.com/a\n"
             "- 苹果公司概况 — https://example.com/b\n"
             "- 苹果 2024 财报 — https://example.com/a\n"
             "https://example.com/c")
_extracted = CitationAgent.extract_sources(_src_text)
check("5a 抽来源：只认 Sources: 节，按 URL 去重",
      [s.url for s in _extracted]
      == ["https://example.com/a", "https://example.com/b",
          "https://example.com/c"]
      and _extracted[0].title == "苹果 2024 财报")
_no_section = CitationAgent.extract_sources("见 https://example.com/x 和 "
                                            "https://example.com/y。")
check("5b 没有 Sources: 节：全文扫 URL 兜底",
      [s.url for s in _no_section]
      == ["https://example.com/x", "https://example.com/y"])
_srcs = [Source("甲", "https://example.com/a"),
         Source("乙", "https://example.com/b")]
_out5, _dropped5 = CitationAgent.cite(
    "- 要点一 [S1]。\n- 要点二 [S2]。\n- 编的 [S9]。\n\n"
    "## Sources\n- 编的来源 — https://invented.example/z",
    _srcs)
check("5c cite：有效标记保留，无效标记删除并记录",
      "[S1]" in _out5 and "[S2]" in _out5
      and "[S9]" not in _out5 and _dropped5 == ["S9"])
check("5d cite：稿子自带的 Sources 节被扔掉，只列核查过的来源",
      "invented.example" not in _out5
      and _out5.count("https://example.com/a") == 1
      and _out5.count("https://example.com/b") == 1)
_out5b, _dropped5b = CitationAgent.cite("- 没引用的要点。", _srcs)
check("5e cite：正文没引用时不追加 Sources 节",
      "## Sources" not in _out5b and _dropped5b == [])
_out5c, _ = CitationAgent.cite("- 同一个来源 [S1] 和 [S1]。", _srcs)
check("5f cite：同一来源引用两次，Sources 节只列一次",
      _out5c.count("https://example.com/a") == 1)

# ---------- §6 plan 解析 ----------

check("6a 编号列表解析",
      parse_subquestions("1. A\n2. B\n3. C") == ["A", "B", "C"])
check("6b 项目符号列表解析",
      parse_subquestions("- A\n- B") == ["A", "B"])
check("6c 没有编号：一行散文当一个问题",
      parse_subquestions("苹果公司怎么样？") == ["苹果公司怎么样？"])
check("6d 空文本：没有子问题",
      parse_subquestions("") == [])

print(f"\n{PASSED}/{PASSED + FAILED} passed.")
sys.exit(0 if FAILED == 0 else 1)
