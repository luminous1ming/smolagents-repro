# Phase 8：作品集打包（2026-11-21，最终阶段）

## 建了什么

| 文件 | 作用 |
|---|---|
| `cli.py` | 命令行入口：`run`（--eval / --task，--agent code/toolcalling，--model fake/live，--max-steps，--mcp-command）、`eval`（转给 eval.py）、`tasks`（列 24 道题）；live 跑完打一行 JSON 摘要并写 `live_checks/cli-<时间戳>.json` |
| `tests_cli.py` | 15 个自测，不联网：`15/15 passed`（子进程跑 CLI；live 分支不测） |
| `mcp_client.py` | stdio MCP 客户端：NDJSON 的 JSON-RPC 2.0（initialize / tools/list / tools/call），`MCPTool` 把 server 工具包成 Tool（schema → inputs，可选参数标 nullable；forward 签名动态拼，见下），`mcp_tools()` 一句话接 server |
| `mcp_test_server.py` | 测试用的 MCP server（mcp_calc / mcp_echo，不联网） |
| `tests_mcp.py` | 20 个自测，不联网：`20/20 passed` |
| `requirements.txt` | 依赖：smolagents[openai]==1.26.0、ddgs==9.16.1（只供原版对照，自己的代码纯标准库） |
| `README.md` | 重写为作品集 README：快速上手、mermaid 架构图、各阶段表、评测表、诚实说明、文件地图、后续可做的事 |
| `notes/phase8-portfolio.md` | 本笔记 |

## MCP 验收

CodeAgent 用 MCP 工具 `mcp_calc` 完成评测题 arith-01（tests_mcp.py
第 4 组）：1 步，state=success，checker 认答案 432。联网又抽了一次
（见下）：真模型 1 步写出 `mcp_calc(expression="128*46")` +
`final_answer(result)`，得 5888——步日志里能看到它确实调了 MCP
工具，不是本地 calc。

## 联网抽查（gemini-2.5-flash，2 次命令，共 3 次调用）

| 命令 | 结果 | 步数 | token in/out | 耗时 |
|---|---|---|---|---|
| `cli.py run --eval arith-01 --agent code --model live` | ✅ 432 | 2 | 2235/38 | 2.7s |
| `cli.py run --task "用 mcp_calc 算 128*46" --model live --mcp-command "python3 mcp_test_server.py"` | ✅ 5888 | 1 | 1211/28 | 1.3s |

记录：`live_checks/cli-20261121-035733.json`、`cli-20261121-035844.json`。
两条命令之间隔了 65 秒（Phase 7 的教训：1 分钟内第 5 次调用撞 429）。
额度账：2.5-flash 当天 11（Phase 5 之前）+ 4（Phase 7 抽查）+ 3（本次）
= 18 次，还剩 2 次；3.5-flash 当天已耗尽，没碰。**本次之后不再联网**，
整轮联网评测仍按 notes/phase7-eval.md 的计划分批跑。

## 测试里抓到的两个真问题（已修）

1. `Tool.__init__` 忽略传进去的 kwargs，属性必须在 `super().__init__()`
   之前就设好（它构造时就校验）；而且 `forward` 的签名必须和 inputs
   的键一一对应，写死的 `**kwargs` 通不过校验。修法：MCPTool 在
   `__init__` 里先设属性，再动态拼一个同签名的 forward（用
   `__signature__` 声明参数名）。
2. 动态签名的可选参数缺默认值——签名层面就必填，
   `validate_tool_arguments` 的 nullable 兜底永远走不到。修法：可选
   参数在签名里给默认 None。另：`calc("1/0")` 返回的是中文“错误：”，
   测试 server 的 isError 判断一开始只认 "Error:"，改成两个前缀都认。

## 全新 checkout 验证（README 的 quickstart 真能跑）

`/tmp/repro-fresh`：干净复制（不含 .venv/runs/eval_runs/trajectories/
live_checks）→ `python3 -m venv .venv` → `pip install -r requirements.txt`
（一次装好）→ 三条演示命令全过：
`run --eval arith-01` ✅ 432；`tasks` 列出 24 道题；
`eval --model fake --limit 2 --systems code` 2/2 通过。
验证完 `/tmp/repro-fresh` 留着（/tmp 本来就是临时的，不用管）。

## 回归

- 全部 9 套：25/25、67/67、45/45、54/54、43/43、45/45（+1 跳过）、
  194/194、20/20、15/15，共 508 个断言，无回归。
- mini-agent 未动。

## 故意留下的未来项（README 里也列了）

- **开源仓库 triage**：roadmap 的扩展项，本阶段没做——核心（CLI +
  MCP + README + 全新 checkout 验证）已满，triage 要接 GitHub API
  取 issues/PRs，联网验证今天额度也不够，留作下一步。
- 整轮联网评测（240–280 次调用，分 6–7 天）、DockerExecutor 对照、
  引用准确率进评测、orchestrator 端到端联网验证、LangGraph 重写——
  见 README 末尾的清单。
- MCP 只验了自写的测试 server，没试过 `npx` 的官方 server；
  MCP 工具进不了 DockerExecutor（子进程在宿主机上），和
  managed_agents 同款限制，写进 README 的诚实说明里了。
