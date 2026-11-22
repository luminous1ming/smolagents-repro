# smolagents 1.26.0 架构笔记（Phase 0，通读源码后写）

> 版本：smolagents 1.26.0（2026-11-21 安装，`pip install 'smolagents[openai]'`）。
> 通读文件：`agents.py`、`memory.py`、`local_python_executor.py`（关键部分）、`tools.py`（关键部分）、`prompts/code_agent.yaml`。
> 注意：README 说 agents.py “~1,000 行”，实测 1.26.0 的 agents.py 是 **1,813 行**
> （models.py 2,102 行、tools.py 1,422 行、local_python_executor.py 1,768 行、memory.py 316 行）。
> 核心循环确实小，但加上多智能体、流式、序列化、Hub 同步等外围后比宣传的要大。

## 一、包结构

| 文件 | 作用 |
|---|---|
| `agents.py` | `MultiStepAgent` 基类 + `CodeAgent` + `ToolCallingAgent` |
| `memory.py` | `AgentMemory` 与四种步骤类型 |
| `models.py` | `Model` 抽象、`ChatMessage`、`OpenAIServerModel`、`LiteLLMModel` 等 |
| `tools.py` | `Tool` 基类、`@tool` 装饰器、工具定义渲染、参数校验 |
| `local_python_executor.py` | AST 解释器 `LocalPythonExecutor`（`PythonExecutor` 接口） |
| `remote_executors.py` | Docker / E2B / Modal 等沙箱执行器，同接口 |
| `monitoring.py` | `Monitor`（token/耗时统计）、`AgentLogger`（rich 输出） |
| `prompts/*.yaml` | 系统提示词模板：`code_agent.yaml`、`toolcalling_agent.yaml` 等 |

## 二、主循环：`MultiStepAgent`

`run(task)` → `_run_stream(task, max_steps)`（生成器），默认 `max_steps=20`：

1. `memory.steps.append(TaskStep(task))`；`run()` 开始时把 `self.state`
   （`additional_args`）和工具一起 `send_variables` / `send_tools` 给执行器。
2. 每轮先看要不要做 **planning step**（`planning_interval` 设了才有；
   第 1 步或每隔 N 步）：单独调一次模型产出计划，记为 `PlanningStep`，
   其 `to_messages()` 会追加一句 “Now proceed and carry out this plan.”
   （刻意换 role，防止模型把计划续写下去）。
3. 每轮建一个 `ActionStep(step_number, timing)`，调子类的 `_step_stream`，
   拿到 `ActionOutput(output, is_final_answer)`。
   - `is_final_answer=True` → 校验（`final_answer_checks`）→ 结束。
   - 模型/解析/执行错误分两类：`AgentGenerationError`（实现侧问题）直接抛；
     其他 `AgentError` 记到 `action_step.error`，循环继续，错误以
     “Error: … Now let's retry” 的 Observation 形式喂回模型（见 memory.py）。
4. 步数用完还没最终答案 → `_handle_max_steps_reached(task)`（再调一次模型强行给答案）。
5. `run()` 返回最终答案，或 `return_full_result=True` 时返回 `RunResult`
  （output + token_usage + 全量 steps + timing + state）。

关键设计：**循环本身不拼消息**。每步开始时 `write_memory_to_messages()`
把 `AgentMemory` 里的全部步骤重新序列化成 `ChatMessage` 列表（见下节），
再交给 `model.generate()`。历史 = 内存的投影，不是增量追加的聊天记录。

## 三、记忆：`AgentMemory`（memory.py）

`AgentMemory(system_prompt)` 存 `SystemPromptStep` + `steps: list[步骤]`。
四种步骤都是 dataclass，各自实现 `to_messages()`：

| 步骤 | 内容 | 序列化成消息 |
|---|---|---|
| `TaskStep` | 任务文本（+图片） | 1 条 user 消息 `"New task:\n..."` |
| `ActionStep` | `model_output`、`tool_calls`、`observations`、`error`、`code_action`、`token_usage`、`timing` | assistant（模型原文）→ tool_call（"Calling tools: …"）→ tool_response（`"Observation:\n..."`，有错时追加重试提示） |
| `PlanningStep` | `plan`（模型产出的计划） | assistant（计划）+ user（"Now proceed…"） |
| `FinalAnswerStep` | 最终答案 | 不进消息（循环已结束） |

另有 `get_succinct_steps()`（去掉 `model_input_messages` 的精简版）、
`get_full_steps()`、`replay()`（把步骤漂亮地打回终端）、
`return_full_code()`（把所有 `code_action` 拼成一个脚本）——
后者对复现很有用：CodeAgent 的完整行为就是一段可复读的 Python。

`CallbackRegistry`：按步骤类型注册回调，`ActionStep` 默认挂
`monitor.update_metrics`（token/耗时统计）。这就是 Phase 0 实测里
token 数的来源。

## 四、`CodeAgent` vs `ToolCallingAgent`

两者共用 `MultiStepAgent` 的循环、记忆、规划、监控，差别只在
**一步之内“模型输出 → 动作”的翻译方式**（即 `_step_stream` 的实现）：

| | `CodeAgent` | `ToolCallingAgent` |
|---|---|---|
| 系统提示 | `prompts/code_agent.yaml`：Thought / Code / Observation 循环，代码块用 `{{code_block_opening_tag}}` 包裹 | `toolcalling_agent.yaml`：JSON tool-calling |
| 模型调用 | `model.generate(messages, stop_sequences=["Observation:", "Calling tools:", 结束标签])` | `model.generate(messages, stop_sequences=[...], tools_to_call_from=...)`，工具以 OpenAI tools 参数传入 |
| 解析 | `parse_code_blobs(output)` 抽代码块 → `fix_final_answer_code` | 优先用模型返回的 `tool_calls`；没有则 `model.parse_tool_calls(chat_message)`（文本解析兜底） |
| 执行 | 整段代码交给 `python_executor`（见五）；记为一次名为 `python_interpreter` 的 ToolCall | `process_tool_calls` → `execute_tool_call` 逐个调工具函数，可并行多调 |
| Observation | 执行器的 stdout（`print` 输出）+ 最后一行返回值 | 每个工具返回值的字符串 |
| 终止 | 代码里调用 `final_answer(x)` → `CodeOutput.is_final_answer=True` | 调 `final_answer` 工具（且一轮只能单独调它） |

CodeAgent 的工具在提示词里不是 JSON，而是 **Python 函数定义**
（`tools.py:get_tools_definition_code` 把每个工具渲染成带签名和 docstring
的 `def`），所以模型写的是自然的 Python 调用，还能用变量、循环、组合。
这就是它比 JSON 方式步数少的原因——一次能做多件事。

`final_answer` 工具是 `_setup_tools` 里默认 `setdefault` 加上的，
两个 agent 都有。`managed_agents`（子智能体）也被包成工具：
要求 name + description，输入固定为 `{"task": ..., "additional_args": ...}`。

## 五、执行器与沙箱：`LocalPythonExecutor`

`local_python_executor.py` 不是 `exec()`，而是一个 **AST 解释器**
（`evaluate_python_code` 逐节点走 AST）：
- 导入白名单（`BASE_BUILTIN_MODULES` + `additional_authorized_imports`），
  非白名单导入直接报 `Import of X is not allowed`；
- 操作数上限 `MAX_OPERATIONS = 10_000_000`，防死循环；
- **有状态命名空间**：变量在各步之间保留（`send_variables` 注入初始变量，
  工具也作为变量注入），所以上一步定义的变量下一步能用；
- `print` 输出被收集为 observation 的一部分。

重要：官方文档明确说本地执行器 **不是安全沙箱**（可以绕过），生产/不可信
代码要走 `remote_executors.py` 的 Docker / E2B / Modal 执行器——它们实现
同一个 `PythonExecutor` 接口（`__call__(code) -> CodeOutput`），
换执行器就是换一个对象。这正是 mini-swe-agent 的做法（`subprocess.run`
换成 `docker exec`）在 smolagents 里的对应物。

## 六、工具系统：`tools.py`

- `Tool`（`BaseTool`）基类：`name`、`description`、`inputs`（JSON Schema）、
  `output_type`、`forward()`。工具是普通 Python 对象，不是 JSON 注册表。
- `@tool` 装饰器：把一个带类型注解和 docstring 的 Python 函数变成工具，
  name / 描述 / 输入 schema 直接从函数签名、docstring、类型提示推导——
  不用手写 JSON Schema。`Tool.from_function` 同理。
- 参数校验 `validate_tool_arguments`；工具定义渲染
  `get_tools_definition_code`（供 CodeAgent 提示词用）。
- `DuckDuckGoSearchTool` 等默认工具在 `default_tools.py`，
  有的要额外依赖（本机实测：`DuckDuckGoSearchTool` 需另装 `ddgs` 包）。

## 七、模型抽象：`models.py`

- `Model` 基类：`generate(messages, stop_sequences, tools_to_call_from) -> ChatMessage`；
  `ChatMessage`（role/content/tool_calls/token_usage）、`TokenUsage`。
- `OpenAIServerModel(model_id, api_base, api_key)`：包一层 `openai` 客户端，
  任何 OpenAI 兼容端点都能接——Phase 0 就是用它接
  `https://generativelanguage.googleapis.com/v1beta/openai/` 跑免费 Gemini。
- `LiteLLMModel`：经 LiteLLM 接 100+ 供应商；另有 `HfApiModel`、`TransformersModel` 等。
- **注意**：`Model.generate` 本身没有重试/退避（超时、429 要靠上层或客户端）。
  用户现有 mini-agent 的 A4 重试（3 次、2s→5s→10s）是复现时要补上的能力。

## 八、与用户现有 mini-agent 的对应关系

| mini-agent（已有） | smolagents 对应物 | 复现时要学的增量 |
|---|---|---|
| `run()` 循环（max 8 步） | `MultiStepAgent.run/_run_stream`（max 20 步、planning、流式） | planning step、错误分类处理、流式生成器 |
| `TOOLS` 注册表（JSON Schema） | `Tool` 类 + `@tool` 装饰器（schema 从签名推导） | schema 推导、工具即 Python 函数 |
| 结构化 tool_calls + Action 文本回退 | `ToolCallingAgent`（tool_calls 优先，`parse_tool_calls` 兜底） | 已有思路，照着规范化 |
| JSONL 步日志 `runs/` | `AgentMemory` 步骤 + `get_full_steps()` | 步骤即数据模型，`to_messages()` 是历史的唯一来源 |
| A4 重试 | （无，模型层不重试） | 保留自己的重试，放到模型封装层 |
| `llm_audit`（逐点调模型） | `managed_agents`（子智能体即工具） | 子智能体编排 |
| （无） | `CodeAgent` + AST 执行器 | **本次复现的核心新东西**：代码即动作 |

## 九、Phase 0 实测记录（2026-11-21，gemini-3.5-flash，见 trajectories/）

模型一次就通，没用 fallback（2.5/2.0-flash 没试）。全程无 429、无解析错误。

| 任务 | 结果 | 耗时 | 动作步数 | tokens（in/out） |
|---|---|---|---|---|
| leopard（README 例题，搜索） | ✅ 正确：155m ÷ 16.11m/s ≈ 9.62s | 119.0s | 4 | 18,007 / 2,017 |
| calc_file（读文件求和×2，无工具） | ❌ 失败：8 步用完，final_answer 为 None | 149.9s | 8（到上限） | 29,024 / 736 |
| tokyo（人口+年份，搜索） | ⚠️ 答案给出但到上限：14.0–14.18M，2024/2025，東京都统计口径有歧义 | 438.6s | 8（到上限） | 36,121 / 982 |
| calc_file2（同题，+ `@tool` read_file） | ✅ 正确：和 65，结果 130 | 247.2s | 1 | 2,238 / 155 |
| primes2（写 primes.txt 再读回，+ `@tool` write/read） | ✅ 文件正确（2,3,5,7,11） | 49.6s | 3 | 7,810 / 275 |
| primes（无工具版） | 未跑：与 calc_file 同一个坑，改跑 primes2 | — | — | — |

合计约 17 分钟、in ~93k / out ~4k tokens，全免费。

**calc_file 失败的根因（重要，复现时必碰）**：`LocalPythonExecutor`
默认禁 `open()` 和 `import os`（白名单只有 queue/re/time/math 等 11 个模块），
读文件必须走 `@tool` 工具。模型被禁后没有停下来，而是连续 8 步尝试绕过沙箱：
`__builtins__` 字典取值、`().__class__.__base__.__subclasses__()`、
`eval("open")`……全被拦截。这是 CodeAgent 在弱模型上的典型失败模式，
复现时要在提示词/工具层就把“文件走工具”讲清楚，而不是指望模型自己发现。

**延迟观察**：单次模型调用耗时极不稳定——同样的短调用，有的 20 秒，
有的近 4 分钟（calc_file2 只有 2,238 个输入 token 却花了 247 秒）。
进程大部分时间是 0 CPU 的网络等待。免费层能跑，但评测（Phase 7）做
批量任务时必须留足时间预算，步数上限也要相应放宽。
