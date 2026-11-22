# Phase 4：CodeAgent —— 核心（2026-11-21）

## 建了什么

| 文件 | 作用 |
|---|---|
| `executor.py` | AST 本地执行器：白名单导入（11 个模块）、禁用名字/dunder 拦截、操作数上限（settrace 数行数）、墙钟超时（signal，主线程）、有状态命名空间、工具进命名空间、print 收集、final_answer 靠异常传值、`fix_final_answer_code` |
| `agents.py`（新增） | `CodeAgent`：系统提示词（Thought / ```python 代码块 / Observation，自己重新写的话）、代码块抽取（围栏 → ```py 兜底 → 裸代码兜底）、执行器集成、执行失败变纠正 Observation |
| `agents.py`（新增） | `AgentExecutionError`（AgentError 子类，执行失败记 error 循环继续） |
| `memory.py`（改） | `ActionStep` 加 `code_action` 字段；代码步的 `to_messages()` 改走纯文本渲染（见下“真 bug”） |
| `tests_code_agent.py` | 54 个自测，不联网：`54/54 passed` |
| `run_live_check_code_agent.py` | 联网检查：真模型走一遍 CodeAgent（1 次调用）；结果写 `live_checks/phase4-code-agent-check.json` |
| `live_checks/phase4-code-agent-check.json` | 本次联网检查的记录 |

## 对照上游的取舍

- 执行器照着上游 `LocalPythonExecutor` 的**策略**，不是照抄它的实现：
  上游是逐节点走 AST 的解释器（1,768 行），这里是“AST 事前检查 +
  受限 exec”——检查负责“什么不许出现”（白名单导入、禁 open/eval/
  dunder、工具名保留），exec 负责“跑起来”。测试锁定的是策略本身，
  不是实现方式。差别都写在 `executor.py` 的模块 docstring 里。
- `MAX_OPERATIONS` 在这里按“执行的代码行数”数（settrace，只数模型
  代码自己的帧，工具内部的行数不算——否则一次 web_search 就能烧光
  上限）；墙钟超时 30 秒照着上游，用 signal 实现，只在主线程可用，
  非主线程只靠操作数上限。
- `import` 要查 `__builtins__["__import__"]`：受限内置函数里没有它时，
  `import math` 会报 `ImportError: __import__ not found`（测试先抓到
  的真 bug）。修法：给一个只认白名单的安全版 `__import__`（fromlist
  为空返回顶层模块，和内置语义一致），AST 检查是第一道锁、它是第二道。
- `setattr` 不放进可用内置函数（上游有）：字符串拼出来的 dunder
  访问绕不过 AST 检查，干脆不给；`getattr`/`hasattr` 用上游的
  nodunder 版（dunder 名字直接拒）。
- 系统提示词照着上游 `prompts/code_agent.yaml` 的结构（Thought/Code/
  Observation 循环、工具写成 Python def、规则列表），话和例子是自己
  重新写的；多了一条 Phase 0 的教训——“文件只能走 read_file 工具，
  永远别用 open()/import os”。
- 上游的 `parse_code_blobs` 兜底是 markdown + 裸代码，这里一样；
  `fix_final_answer_code` 照着上游的正则逻辑抄过来（只在“既调用又
  赋值”时重命名赋值）。
- 执行器和 CodeAgent 的分工照着上游：执行器只管跑代码、返回
  CodeOutput（output/logs/is_final_answer）；循环、记忆、纠正
  Observation 都在 agents.py。

## 联网检查里抓到的一个真 bug（已修，有回归测试）

CodeAgent 的第二步请求被 Gemini 的 OpenAI 兼容端点 400 拒绝
（`INVALID_ARGUMENT`，两次都发生在第二步，第一步是好的）。
根因：Phase 3 的 `ActionStep.to_messages()` 把 tool_calls 挂在
assistant 消息上、Observation 用带 tool_call_id 的 tool 角色消息——
这对 ToolCallingAgent 是对的（它每步都传 tools 参数），但 CodeAgent
调模型时不传 tools 参数，历史里一旦出现 OpenAI 形状的 tool_calls，
端点就拒收。上游的 `ActionStep.to_messages()` 从来不这么干：
模型原文是普通 assistant 文本，工具调用只是一行 "Calling tools: …"
文本，Observation 是 TOOL_RESPONSE 文本角色。
修法：`code_action` 不为空的步（即 CodeAgent 的步）改走纯文本渲染——
assistant 消息不挂 tool_calls，Observation 和纠正消息都走 user 角色
（mini-agent 原来的做法）；ToolCallingAgent 的步保持原样。
回归测试：`tests_code_agent.py` 的 6e 组（代码步纯文本 + 工具调用步
保持 OpenAI 形状）。

## 联网检查结果（2026-11-21，gemini-2.5-flash）

- 最终任务“直接算出 128*46，一步内用 final_answer 返回答案”：
  1 步、1.0s、in 1031 / out 23 tokens，模型写
  `result = calc("128*46")` + `final_answer(result)`，输出 5888，
  state=success。**PASS**。
- 第一次尝试的任务是“计算 128*46，再把结果乘以 2”（max_steps=2）：
  模型遵守了提示词规则 4（依赖型工具调用先 print、下一步再用），
  第 1 步算 128*46、第 2 步算 5888*2，始终没调 final_answer，
  步数用完走强制答案（11776，值是对的，state=max_steps_error）。
  结论：不是模型失败，是任务步数预算太紧 + 规则 4 本来就会让模型
  分步；评测任务设计时要按这个行为留步数。记录在
  `run_live_check_code_agent.py` 的注释里。
- **Phase 2 的待定项有结论了**：系统提示词里
  `def final_answer(answer: any) -> any:`，模型照常用它调了
  final_answer()——`-> any` 在代码提示词里没问题；Phase 2 的 400
  只发生在 JSON tools 参数里（wire 上不发 any 是对的）。
- 调用账：本次 Phase 4 联网共 8 次 generate（2.5-flash）——
  400 诊断 2 次尝试 + 1 次直接调用诊断 + 修复后验证 2 次任务
  （第一次 2 步，第二次 1 步）+ Phase 3 当天已用的 2 次，
  远低于每天 20 次的免费额度。3.5-flash 今天没碰（额度已耗尽）。

## 验收：CodeAgent vs ToolCallingAgent 步数对比（FakeModel 实测）

| 任务 | CodeAgent | ToolCallingAgent |
|---|---|---|
| A：calc + 读文件 + 求和 | 1 步 | 4 步 |
| B：读文件 + 循环平方求和 | 1 步 | 3 步 |
| C：查数 + 计算 | 1 步 | 3 步 |

任务 A/B/C 的脚本和断言都在 `tests_code_agent.py` 第 7 节；
任务 C 的“查数”用一个 `@tool` 桩工具（`lookup_population`）
代替联网搜索，保持离线可测。代码即动作的优势是实的：一次能做
多件事（变量、循环、组合调用），不是只换了个写法。

## 回归

- `tests_models.py` 25/25，`tests_tools.py` 62/62，`tests_agents.py` 45/45，无回归。
- mini-agent 未动（agent.py md5 702ebae7b293cefba48a1a602b60b7df，和 Phase 3 时一致）。

## 故意留到 Phase 5 的

- `managed_agents`（子智能体即工具）：CodeAgent 提示词里现在没有
  team member 段；Phase 5 按 Anthropic orchestrator-worker 模式加。
- 流式生成（`generate_stream`）、图片内容消息——上游有，本阶段不需要。
- 执行器的 Docker 换法（Phase 6）：`LocalPythonExecutor` 和以后
  的 Docker 执行器是同一个接口（`__call__(code) -> CodeOutput`），
  换执行器就是换一个对象。
- 429 是否纳入重试：仍按 Phase 1 的 A4 规则（429 直接抛），不改。
- 评测集（Phase 7）：步数对比现在是 3 个 FakeModel 脚本任务；
  批量评测（真模型、多任务、token/步数统计）等 Phase 7。
