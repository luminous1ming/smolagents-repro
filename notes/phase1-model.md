# Phase 1：模型抽象层（2026-11-21）

## 建了什么

| 文件 | 作用 |
|---|---|
| `models.py` | `Model` 基类、`ChatMessage`、`TokenUsage`、`ToolCall`、`OpenAICompatibleModel`（stdlib urllib）、`FakeModel`、A4 重试、stop 截断 |
| `tests_models.py` | 25 个自测，不联网：`24/24` 首次跑时 1 个失败是测试本身的计数错误（已修），现 `25/25 passed` |
| `run_live_check.py` | 联网检查：同一段输入走 FakeModel 和真模型，比消息形状；结果写 `live_checks/phase1-model-check.json` |
| `live_checks/phase1-model-check.json` | 本次联网检查的记录（模型、延迟、token、形状） |

## 对照上游的取舍

- 签名照着 smolagents `Model.generate(messages, stop_sequences, tools_to_call_from)`，
  `ChatMessage`/`TokenUsage` 同名同字段，方便 Phase 3 直接对上。
- 上游 `Model.generate` 没有重试——这里把 mini-agent 的 A4 原样搬进来：
  超时/连接错误/5xx 重试 3 次（2s→5s→10s），4xx 不重试。这是复现里
  明确比上游多的能力。
- 只用标准库 urllib，不引 openai 包（和 mini-agent 的 `call_llm` 同一路）。
- stop 序列服务端传一份、客户端再截一次，和上游做法一致。
- `FakeModel` 的用量按词数算，确定性——同一段输入永远得到同一个数。

## 联网检查结果（2026-11-21）

- **原定模型 gemini-3.5-flash 今天跑不了**：免费层每天只给 20 次
  `generate_content` 请求（见下），Phase 0 当天已用了 24 次以上，额度耗尽。
  两次尝试都返回 HTTP 429，错误写明
  `generate_content_free_tier_requests, limit: 20, model: gemini-3.6-flash`
  （请求写的是 gemini-3.5-flash，服务端按 gemini-3.6-flash 计量——
  可能是别名或已改名，记录为实测现象，不下结论）。
- **改用 gemini-2.5-flash 跑通**：额度似乎按模型分别计算，2.5-flash 当天
  200 OK。主检查与 stop 检查都 PASS：
  - 主提示（128*46）：形状一致， live 内容 `"5888"`，延迟 1.0s，
    用量 in 27 / out 4。
  - stop 检查（写 1、2、3，stop=["2"]）：live 内容 `"1\n"`，在 2 之前截断，
    形状一致，延迟 1.0s。
  - 注意：2.5-flash 这次只要 1 秒，远快于 Phase 0 在 3.5-flash 上
    20 秒到 4 分钟的波动——可能是模型不同，也可能是负载不同，单一样本别当结论。
- **3.5-flash 的复查待办**：额度重置后跑
  `PHASE1_MODEL=gemini-3.5-flash .venv/bin/python run_live_check.py`。
  命令本身已支持用环境变量换模型，不用改代码。
- 另：接口返回的 `total_tokens`（21）不等于 prompt+completion（4+2）；
  我的 `TokenUsage` 和上游一样按两者之和算（6），记录为实测现象。

## 额度这件事对后面的影响（重要）

免费层每个模型每天 20 次请求，一个多步 agent 一跑就是 4–8 次。
Phase 3 起必须把联网调用当预算花：循环逻辑一律先在 FakeModel 上跑通，
联网只做抽查；Phase 7 批量评测要么分摊到多天、要么换模型分摊、
要么由 luminous 决定是否开付费。不要在一天内把 20 次烧完又发现要返工。

## 故意留到后面的

- 流式生成（`generate_stream`）、LiteLLM / HfApi / Transformers 模型、
  图片内容消息——上游有，本阶段不需要。
- 工具 schema 生成（`@tool` 从签名推导）——Phase 2。
- **429 是否纳入重试**：现在按 A4 规则，429 是 4xx，直接抛（测试已锁定
  该行为）。免费层额度紧，429 更常见，重试 429（读 Retry-After 再等）
  是个候选改进，等 luminous 或 Phase 3 再定——不要现在悄悄改规则。
- `response_format` 只经 `**kwargs` 透传，没单独测试。
