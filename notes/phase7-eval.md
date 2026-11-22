# Phase 7：评测（2026-11-21）

## 建了什么

| 文件 | 作用 |
|---|---|
| `eval_tasks.py` | 24 道自写题（算术/文件/研究各 8 道）+ checker + FakeModel 假脚本 |
| `eval.py` | harness：同一套题跑三个系统（code / toolcalling / upstream），记成功率、步数、token、耗时、429；支持 --limit/--offset/--tasks 分批、--sleep 轮间停顿 |
| `tests_eval.py` | 194 个自测，不联网：`194/194 passed` |
| `default_tools.py`（改） | 加 `WriteFileTool`（ReadFileTool 的写 counterpart，base_dir 内写、自动建子目录）；默认工具箱从 4 个变 5 个 |
| `tests_tools.py` / `tests_agents.py`（改） | 工具数断言 4→5；WriteFileTool 行为测试 |
| `eval_runs/` | 每轮的结果：results.json + summary.md + 各题 workdir + logs/ |

## 评测集设计

题目全部自己写，不用风控数据、不用现成 benchmark 的题：
- **arith（8 道）**：纯算术，答案是一个数，checker 在答案里找这个数；
- **file（8 道）**：读/写/算/排序/转大写，checker 看 workdir 里落盘的
  文件（+ 答案里的关键数）；每道题的 setup 数据文件跑前写进 workdir，
  每个 (题, 系统) 都在独立 workdir 里跑，互不干扰；
- **research（8 道）**：稳定事实（Python 1991、Everest 8848、1984、
  Au 79、登月 1969、Canberra 1913、8 大行星、豹子 9.62 秒），
  checker 只看事实子串——引用本身对不对不在这里打分
  （那是 Phase 5 CitationAgent 的验收标准）；事实特意选稳定的，
  不用“当前人口”这种会变的数（Phase 0 的 tokyo 题就是教训）。

FakeModel 用的“假答案脚本”：CodeAgent 一整块真算/真读写代码、
ToolCallingAgent 分步 tool_calls 序列——证明任务本身是可解的
形状（工具够、答案可核查），不是证明模型会做。

## FakeModel 全量跑（24 题 × 3 系统，离线）

结果：`eval_runs/20261121-035018/results.json`

| 系统 | 通过 | 平均步数 | 总 token | 429 |
|---|---|---|---|---|
| code（自己的 CodeAgent） | 24/24 | 1.0 | 16,023（假用量，按词数算） | 0 |
| toolcalling（自己的 ToolCallingAgent） | 24/24 | 2.4 | 24,362（假用量） | 0 |
| upstream（原版 smolagents CodeAgent） | 24/24 | 1.0 | 0（假模型不报用量） | 0 |

三系统全过，说明 harness、checker、假脚本三处都通了。
跑通之前 FakeModel 先抓到一个真 bug：research-02 的假答案写
"8,848"，checker 找子串 "8848" 匹配不到——修法是 contains_all
先去掉千分位逗号再匹配（eval_tasks.py 里有注释）。

步数对比和 Phase 4 的结论一致：CodeAgent 1 步能做完的事，
ToolCallingAgent 要 2–4 步（文件题 3 步，研究题 2–3 步）。
注意：FakeModel 的步数是“脚本最少步数”，不是模型真实步数——
真模型会多走纠正步，见下。

## 联网抽查（gemini-2.5-flash，learner CodeAgent，只抽 3 道）

| 任务 | 结果 | 步数 | token in/out | 耗时 | 调用 |
|---|---|---|---|---|---|
| arith-01 | ✅ 432 | 2 | 2235/38 | 2.7s | 2 |
| file-01 | ✅ {'sum': 65, 'final_result': 130} | 2 | 2328/166 | 3.6s | 2 |
| research-02 | ❌ 429 | 0 | 0/0 | 4.5s | 1 次尝试 |

记录：`eval_runs/20261121-035028`、`035034`、`035050`。

两点值得记：
- 真模型 2 步做完 FakeModel 1 步的题——Phase 4 的规则 4
  （依赖型调用先 print、下一步再用）在这里又出现了，
  所以评测 max_steps 设 15 是对的，8 步不够。
- research-02 在 1 分钟内第 5 次调用就撞了 429——免费层不只有
  每天 20 次的额度，还有每分钟的限流。harness 因此加了
  `--sleep`（轮间停顿）；整轮联网跑必须带停顿、不能连发。
  429 记账本身验证通过：error_429=True、success=False、steps=0，
  整轮没炸，符合 Phase 1 的 A4 规则（429 直接抛，不重试）。

额度账（2026-11-21 当天）：2.5-flash 当天已用 11 次，本次抽查
成功 4 次 + 429 尝试 1 次 = 15 次，还剩 5 次当日额度；
3.5-flash 当天已耗尽，没碰。

## 整轮联网跑的计划（本次没跑，额度不够）

规模估算：24 题 × 3 系统；按抽查的真模型步数（code ~2–3 步，
toolcalling ~3–5 步，upstream ~3–5 步），约 **240–280 次
generate 调用**。Phase 0 的单次调用耗时 20 秒–4 分钟，
按平均 ~45 秒算，纯调用约 3–3.5 小时墙钟；再加轮间停顿
（--sleep 60–90 秒 × 72 轮 ≈ 1.2–1.8 小时），总计约 4–5 小时，
必须分批。

分批跑法（每天每模型 20 次）：
```bash
# 一天一批：4–6 道题 × 1 个系统，轮间停 60 秒
EVAL_MODEL=gemini-2.5-flash .venv/bin/python eval.py --model live \
    --systems code --tasks arith-01,arith-02,arith-03,arith-04 \
    --sleep 60
# 换模型继续（3.5-flash 额度重置后），或第二天继续
EVAL_MODEL=gemini-3.5-flash .venv/bin/python eval.py --model live \
    --systems code --offset 4 --limit 4 --sleep 60
# 结果是多个 eval_runs/<时间戳>/results.json，汇总时把 records 拼起来
```
按每天 2 个模型各 20 次算，约 6–7 天跑完；只用一个模型约 13 天。
建议先跑 code 系统（主角），再跑 toolcalling 和 upstream 作对照。
研究题额外注意：learner 的 web_search 是 DuckDuckGo 即时答案，
搜不到就返回“没搜到”——研究题失败时要先看是不是搜索后端的问题，
不是 agent 的问题；必要时换上游的 DuckDuckGoSearchTool（ddgs 全量
搜索）对照一次。

## 回归

- `tests_models.py` 25/25，`tests_tools.py` 67/67，`tests_agents.py` 45/45，
  `tests_code_agent.py` 54/54，`tests_multi_agent.py` 43/43，
  `tests_sandbox.py` 45/45（+1 跳过），`tests_eval.py` 194/194。
- mini-agent 未动。

## 故意留到 Phase 8 的

- **MCP 客户端**：核心评测做完后没剩时间做。留给 Phase 8：
  一个 stdio JSON-RPC 的 MCP 客户端，把一个 MCP server 包成工具源
  （tools/list + tools/call），接进 ToolRegistry；验收是 CodeAgent
  能调一个 MCP 工具完成一道评测题。
- **整轮联网跑**：见上，按计划分批跑；跑完把三个系统的成功率/
  步数/token 对照表写进 Phase 8 的作品集 README。
- **DockerExecutor 对照**：Phase 6 留的——文件题走 DockerExecutor
  跑一遍，和本地执行器对比步数/成功率/耗时；需要有 Docker 的机器，
  本 VM 跑不起真容器（见 notes/phase6-sandbox.md）。
- **引用准确率**：研究题目前只打分事实对不对；Phase 5 的
  CitationAgent（引用必须指向 researcher 交过的来源）还没进评测，
  Phase 8 可以加一个“答案里的引用有多少是真实来源”的指标。
- **orchestrator 端到端联网验证**：Phase 5 留的完整流程
  （规划 1 + researcher 2–3×1–2 + 综合 1 ≈ 5–7 次调用），
  可以当作评测集里的一道研究题来跑，额度够的时候再做。
