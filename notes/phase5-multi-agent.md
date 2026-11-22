# Phase 5：多智能体 + Anthropic orchestrator-worker（2026-11-21）

## 建了什么

| 文件 | 作用 |
|---|---|
| `agents.py`（改） | `managed_agents`：`MultiStepAgent.__init__` 加 `name`/`description`/`managed_agents`；`_setup_managed_agents`（重名校验）；`ManagedAgentTool`（子智能体包成工具，输入 task + additional_args）；`MultiStepAgent.__call__`（派活入口：包任务提示词 → 跑一轮 → 包报告）；ToolCallingAgent 和 CodeAgent 的提示词、工具列表、执行器都带上子智能体 |
| `research.py`（新） | orchestrator-worker 研究流程：`ResearchOrchestrator`（规划 → 并行派活 → 收集 → 综合 → 引用）；`CitationAgent`（确定性引用核查）；`Source`/`ResearchResult`/`ResearchBrief` |
| `tests_multi_agent.py`（新） | 43 个自测，不联网：`43/43 passed` |
| `run_live_check_multi_agent.py`（新） | 联网检查：CodeAgent lead + 1 个 researcher（3 次调用）；结果写 `live_checks/phase5-multi-agent-check.json` |
| `live_checks/phase5-multi-agent-check.json` | 本次联网检查的记录 |

## 对照上游的取舍

- managed_agents 照着上游：登记时要求 name + description、名字
  不能和工具/自己/其他子智能体重名；子智能体包成工具时输入固定
  task + additional_args（nullable object）；`__call__` 包任务提示词
  再包报告——父智能体只看到报告字符串，看不到子智能体内部步骤。
  差别：上游的 `__call__` 还会把 summary 拼进报告，这里只包最终
  答案（summary 是调试信息，不是报告的一部分）。
- 上游把 managed agents 同时放进 tools 和 executor（CodeAgent）；
  这里一样：ToolCallingAgent 的 tools 参数和 `_execute_tool_call`
  都查两处注册表；CodeAgent 的提示词和执行器也都带上。
- orchestrator-worker 照着 Anthropic 那篇博客的五步：lead 规划 →
  并行 researcher → 综合 → CitationAgent。差别有二：
  1. 并行是真的并行（ThreadPoolExecutor），不是上游那种“调工具
     式”的串行——每个 researcher 有自己的 model（factory 发）、
     工具箱、记忆、日志；上下文隔离是结构保证的。
  2. CitationAgent 是确定性的字符串核查，不是模型调用：引用只能
     指向 researcher 交过的来源，编不出来。Anthropic 的 CitationAgent
     也是独立 pass，但确定性核查比“让模型再判断”更能保证验收标准。
- 综合调一次 lead 的 generate，不跑 lead agent 的循环——综合时
  不需要再派活，调一次就够，省调用也省步数。
- `additional_args` 现在只进 `self.state`（CodeAgent 的执行器会
  把它当变量送进去），跑完恢复原样；本阶段用不上，先留着接口。

## 测试里抓到的两个真问题（都已修）

1. `CitationAgent.extract_sources` 的兜底分支每行只抠第一个 URL，
   且不 strip 中文句号——一行两个链接会漏一个，链接末尾带“。”会
   错。修法：兜底分支用 finditer 全抠 + `_clean_url` 统一去尾标点。
2. 测试断言 2c/2d 用的标记算式 128*46 撞了 calc 工具描述里的例子
   （lead 的系统提示词里本来就有它），隔离断言一开始“碰巧通过”。
   修法：researcher 改算 987*65；`_memory_text` 改成正文 + 工具
   调用参数一起拼（工具参数不在 content 里，只拼正文会漏）。

## 验收（FakeModel）

- 宽度优先的研究简报端到端：规划拆 3 个子问题 → 3 个 researcher
  并行 → 综合 → 引用核查；每条引用都可追溯到 researcher 交过的
  来源（S1/S3/S4 被引用、没被引用的 S2 不进正文和 Sources 节）；
  两个 researcher 交了同一条来源时按 URL 去重（4 条）。
- 上下文隔离：各 researcher 的记忆是独立对象；每个 researcher 的
  任务里只有自己的子问题；lead 的历史里没有 researcher 内部的
  calc 参数（987*65），researcher 自己的历史里有。
- 综合只用子智能体输出：综合稿里编的 [S99] 和假链接
  https://invented.example/secret 进不了最终简报（删掉并记进
  dropped_citations，有效引用不受影响）。
- researcher 失败不炸整轮：一个 researcher 的模型抛错时，其他
  两路照常出简报，失败那路交一份失败说明、不交来源。
- 并行数上限：规划出 8 个子问题，max_researchers=5 只派 5 个。

## 联网检查结果（2026-11-21，gemini-2.5-flash）

- 任务“让研究员算 128*46，把它的答案原样作为最终答案返回”：
  lead 的 CodeAgent 一步写出
  `final_answer(researcher_1(task="计算 128*46", additional_args={}))`——
  同块串联的 instructions 生效了；researcher 两步（calc → 5888，
  final_answer 交报告）；最终答案是 researcher 的报告，含 5888，
  state=success，4.1s，in 1148 / out 30 tokens。**PASS**。
- 调用账：本次共 3 次 generate（lead 1 + researcher 2）——比设计
  的 2 次多 1 次，因为 researcher 先调了 calc 再交答案（正常的
  agent 行为，不是接线问题）。2.5-flash 当天累计 8 + 3 = 11 次，
  低于每天 20 次；3.5-flash 今天没碰（额度已耗尽）。

## 回归

- `tests_multi_agent.py` 43/43，`tests_models.py` 25/25，
  `tests_tools.py` 62/62，`tests_agents.py` 45/45，
  `tests_code_agent.py` 54/54，无回归。
- mini-agent 未动（agent.py md5 702ebae7b293cefba48a1a602b60b7df，和
  Phase 3 时一致）。

## 故意留到 Phase 6 的

- 执行器的 Docker 换法（`LocalPythonExecutor` 和以后的 Docker
  执行器是同一个接口 `__call__(code) -> CodeOutput`，换执行器就是
  换一个对象；managed agent 工具在 Docker 执行器里一样能调——
  它是 Python 层面的调用，不进沙箱代码）。
- 流式生成（`generate_stream`）、图片内容消息——上游有，本阶段不需要。
- 评测集（Phase 7）：orchestrator 的端到端评测（多题、成功率、
  引用准确率、token/步数统计）等 Phase 7；这里的 FakeModel 验收
  只覆盖单题。
- 429 是否纳入重试：仍按 Phase 1 的 A4 规则（429 直接抛），不改。
- 完整 orchestrator 的联网端到端（规划 1 + researcher 2–3×1–2 +
  综合 1 ≈ 5–7 次调用）没跑：2.5-flash 当天剩余额度够，但按任务
  要求只做 2 次调用的抽查；完整流程的联网验证等 Phase 7 评测时
  再做（或等 3.5-flash 额度重置后）。
