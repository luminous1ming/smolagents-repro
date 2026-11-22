# Phase 2：工具系统（2026-11-21）

## 建了什么

| 文件 | 作用 |
|---|---|
| `tools.py` | `Tool` 基类、`@tool` 装饰器、类型提示→JSON Schema 推导、`validate_tool_arguments`、渲染（`to_code_prompt` / `to_openai_function`）、`ToolRegistry` |
| `default_tools.py` | 默认四件套：`calc`、 `ReadFileTool`、`web_search`、`final_answer` + `default_tools()` 工厂 |
| `tests_tools.py` | 62 个自测，不联网：`62/62 passed` |
| `run_live_check_tools.py` | 联网检查：真模型拿注册表的 tools 参数调 calc；结果写 `live_checks/phase2-tools-check.json` |
| `live_checks/phase2-tools-check.json` | 本次联网检查的记录 |

## 对照上游的取舍

- `Tool` 基类照着上游：name / description / inputs / output_type
  四个类属性 + `forward()`，实例化时校验（坏名字、inputs 缺字段、
  forward 签名与 inputs 不一致、坏 output_type 都在 `__init__` 里拦住）。
  `__call__` 照着上游：懒 `setup()`、单个 dict 参数拆成 kwargs、
  位置参数按签名转 kwargs、调前先校验。
- `@tool` 照着上游：`get_json_schema` 从签名 + Google 风格 docstring +
  类型提示推导（`int→integer、float→number、str→string、bool→boolean、
  list→array、dict→object、Optional→nullable、Literal→enum、docstring
  末尾 (choices: [...])→enum`）；缺 docstring、缺参数描述、缺类型提示、
  缺返回提示都抛错（和上游同款异常名）。
- 渲染照着上游：`to_code_prompt` 把工具写成 `def name(arg: type) -> ...`
  （Phase 4 给 CodeAgent 提示词）；`to_tool_calling_prompt` 一句话介绍。
- 多一个上游没有的 `ToolRegistry`（mini-agent TOOLS dict 的对象版）：
  注册一次，之后按名字调（dict 或 JSON 字符串）、导出 OpenAI tools 参数
  （Phase 3）、导出整段代码定义（Phase 4）。Phase 3 的 MultiStepAgent
  直接拿它当工具箱。
- 工具返回永远是字符串 Observation（或能转成字符串的值），和 mini-agent
  的工具约定一致；`final_answer` 原样返回答案、靠名字认（和上游一样）。

## 移植的四个工具

- `calc`（@tool）：mini-agent 的 AST 白名单算术原样搬过来。
- `read_file`（Tool 子类，要带 `base_dir`）：mini-agent 的“只读当前目录”
  换成构造时传 `base_dir`，测试能用临时目录；Phase 0 的教训——
  CodeAgent 执行器禁 `open()`，文件访问只能走这个工具。
- `web_search`（@tool）：mini-agent 的 DuckDuckGo 即时答案（标准库 urllib，
  不引 ddgs，和 models.py 同一路）。
- `final_answer`（Tool 子类）：上游 FinalAnswerTool 原样复现，
  inputs 只有一个 `answer`（any 类型）。

## 联网检查结果（2026-11-21，gemini-2.5-flash，1 次调用）

- 模型拿到 4 个工具的 OpenAI 参数，返回 1 次 tool_call：
  `calc({"expression": "128*46"})`，注册表执行得 "5888"。PASS。
- 延迟 1.0s，用量 in 265 / out 18。
- **中途发现一个真问题**：第一次跑返回 HTTP 400，报错
  `unrecognized type 'any' at properties.answer`——Gemini 的 OpenAI
  兼容端点不认 `"type": "any"`。修法：`to_openai_function` 里
  "any" 类型就不写 type（只留描述），"any" 本来就是“类型不限”的意思；
  工具内部的 inputs 仍保留 "any"（和上游一致）。修完重跑 PASS，
  测试里也锁定了这个行为（wire 上不发 any、内部保留 any）。
  这是复现里明确比“照抄上游”多的适配：上游的 schema 是按 OpenAI
  端点写的，Gemini 端点要收敛一点。

## 故意留到后面的

- `final_answer` 的结束判定（按名字认）——Phase 3 的循环里做。
- MCP 工具、Hub 加载（`from_hub`/`push_to_hub`）、Gradio demo——上游有，
  本项目不需要。
- `to_code_prompt` 里 output_type 直接写 "any"（上游同款写法），
  Phase 4 拼提示词时再看模型吃不吃得下。
- 429 是否纳入重试：仍按 Phase 1 的 A4 规则（429 直接抛），不改。
- `web_search` 的联网实测：离线测试已用 mock 覆盖三种返回；
  真联网搜索放在 Phase 3 的 ToolCallingAgent 抽查里（不单独烧额度）。
