# Phase 3：MultiStepAgent + ToolCallingAgent（2026-11-21）

## 建了什么

| 文件 | 作用 |
|---|---|
| `memory.py` | 步骤记忆：TaskStep / ActionStep / PlanningStep / FinalAnswerStep / SystemPromptStep，各自 `to_messages()`；`AgentMemory`（reset / 精简 / 全量步骤）；最小版 `CallbackRegistry` |
| `agents.py` | `MultiStepAgent`（循环、规划、步数上限、JSONL 日志）+ `ToolCallingAgent`（结构化 tool_calls + 文本解析兜底 + final_answer 按名字认）；错误类型 AgentError / AgentGenerationError / AgentParsingError / AgentMaxStepsError |
| `tests_agents.py` | 45 个自测，不联网：`45/45 passed` |
| `run_live_check_agents.py` | 联网检查：真模型跑完整循环（2 次调用）；结果写 `live_checks/phase3-agents-check.json` |
| `live_checks/phase3-agents-check.json` | 本次联网检查的记录 |
| `runs/` | 每轮的 JSONL 步日志（mini-agent 的格式） |

## 对照上游的取舍

- 记忆照着上游：四种步骤 + `to_messages()` 是历史的唯一来源；
  `write_memory_to_messages()` 每步从 memory.steps 重新拼消息。
  差别在消息形状：上游 ChatMessage.content 是 `[{"type": "text", ...}]`
  列表，这里是 Phase 1 的字符串 content；上游 ActionStep 用
  TOOL_CALL / TOOL_RESPONSE 两种角色，这里按 OpenAI 形状——
  tool_calls 挂在 assistant 消息上，Observation 是带 tool_call_id
  的 tool 消息（和 mini-agent 的做法一致，真模型吃得下）。
- 循环照着上游：规划步（第 1 步 + 每隔 N 步）、错误分两类
  （AgentGenerationError 直接抛；其他 AgentError 记 step.error，
  下一步以纠正 Observation 喂回模型——mini-agent A2 的做法），
  步数用完 `_handle_max_steps_reached` 再调一次模型强行给答案。
- 工具执行失败不走异常：未知工具、参数校验失败、工具内部抛错，
  全部变成该次调用的 Observation（mini-agent 的工具约定），
  效果和上游“抛 AgentError 再由循环记 error”一样（循环继续），
  但少一层异常转换。final_answer 和其他调用同批时也一样：
  记成 Observation 让模型自己改，不结束循环。
- 文本解析兜底两层：先找带 `"name"` 的 JSON 对象块（大括号配对，
  字符串内的大括号不算），再试 mini-agent 的 `Action: name(args)`
  文本格式；单输入工具的原始参数会包成 dict 再调（registry.call
  只接受 dict 或 JSON 字符串）。
- 不上流式：上游 `_run_stream` 是生成器，这里 `_step_stream` 直接
  返回 ActionOutput（Phase 1 已说明 generate_stream 留到后面）。
- max_steps 默认 8（mini-agent 的默认值；上游是 20），可按次覆盖。
- JSONL 日志沿用 mini-agent 的 runs/ 格式；文件名多加 6 位微秒
  （mini-agent 只精确到秒，同一秒连跑两轮会撞文件——测试里发现的）。

## 验收：与 mini-agent 的对等

- 同一个演示任务“帮我算一下 128*46，再读一下 README.md 的前 5 行”，
  FakeModel 脚本走新 ToolCallingAgent：3 步（calc → read_file →
  final_answer），最终答案含 5888 和 README 前 5 行，JSONL 日志
  记下每一步的工具、Observation 和耗时。
- 规划：interval=1 时 3 个动作步 → 3 个 PlanningStep；
  interval=2、2 个动作步 → 只有第 1 步有规划；None → 没有规划。
- 步数上限：max_steps=2 且模型一直调 calc，`_handle_max_steps_reached`
  强行给答案，RunResult state 是 max_steps_error，记忆里有
  error 含 "max steps" 的 ActionStep。
- 解析失败：第 1 步记 error（"模型没有返回工具调用…"），
  第 2 步输入里有 "Now let's retry" 的纠正 Observation，循环继续。
- 模型收到的 tools 参数：注册表 4 个工具的 OpenAI schema。

## 联网检查结果（2026-11-21，gemini-2.5-flash，2 次调用）

- 任务“算一下 128*46”：step 1 calc → 5888，step 2 final_answer，
  state success，耗时 2.3s，用量 in 1756 / out 48。PASS。
- 只用了 2 次调用（预算内）。注意：2.5-flash 这次只要 2.3 秒，
  远快于 Phase 0 在 3.5-flash 上 20 秒到 4 分钟的波动——
  单一样本，别当结论。
- 3.5-flash 今天额度已耗尽（Phase 1 笔记），没碰；复查等额度重置。

## 回归

- `tests_models.py` 25/25，`tests_tools.py` 62/62，无回归。
- mini-agent 未动（本阶段所有工作都在 smolagents-repro 目录里）。

## 故意留到 Phase 4 的

- `CodeAgent`：系统提示词（Thought / 代码块）、代码块抽取、
  AST 执行器（白名单导入、操作数上限、有状态命名空间）、
  `final_answer()` 作为终止工具——复现的核心新东西。
- `to_code_prompt` 的 output_type "any" 写法（Phase 2 笔记）：
  Phase 4 拼提示词时再看模型吃不吃得下。
- 流式生成（`generate_stream`）、`managed_agents`（子智能体即工具）、
  图片内容消息——上游有，本阶段不需要。
- 429 是否纳入重试：仍按 Phase 1 的 A4 规则（429 直接抛），不改。
