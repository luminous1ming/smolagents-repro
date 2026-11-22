#!/usr/bin/env python3
"""Phase 7：评测 harness —— 同一套题，跑三个系统，比步数/token/成功率。

三个系统：
  - code：自己的 CodeAgent（agents.py）
  - toolcalling：自己的 ToolCallingAgent（agents.py）
  - upstream：原版 smolagents 1.26.0 的 CodeAgent（装好的包，不改）

评测集在 eval_tasks.py（24 道自写题：算术/文件/研究各 8 道）。
每道题 × 每个系统跑一轮，记：成功与否、步数、token、耗时、
429（模型直接抛 429 时记 error_429，不重试——Phase 1 的 A4 规则）。

跑法（分批是故意的：免费层每天每模型 20 次调用，整轮跑不完）：
    .venv/bin/python eval.py --model fake                    # 全量，离线
    .venv/bin/python eval.py --model fake --limit 5 --offset 10
    .venv/bin/python eval.py --model fake --tasks arith-01,file-01
    EVAL_MODEL=gemini-2.5-flash .venv/bin/python eval.py --model live \\
        --systems code --tasks arith-01                     # 联网抽查

结果写进 eval_runs/<时间戳>/results.json（每轮一条记录），
同时在终端打一张汇总表。每个 (题, 系统) 都在自己的 workdir 里跑，
文件题互相不干扰；agent 的 JSONL 步日志也写进该轮目录的 logs/。

注意：upstream 系统的工具是这里现定义的四个（calc / read_file /
write_file / web_search），和自己的默认工具箱对齐——比的是 agent
循环本身，不是搜索后端。web_search 两边都是 DuckDuckGo 即时答案
（上游的 DuckDuckGoSearchTool 要 ddgs 全量搜索，这里不用，
避免工具强弱干扰对比）。
"""
import argparse
import ast
import io
import json
import os
import sys
import time
import traceback
import urllib.parse
import urllib.request
from contextlib import redirect_stdout
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import CodeAgent, ToolCallingAgent
from default_tools import default_tools
from eval_tasks import TASKS, EvalTask, get_task
from models import ChatMessage, FakeModel, OpenAICompatibleModel, ToolCall

HERE = os.path.dirname(os.path.abspath(__file__))
API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_MAX_STEPS = 15  # Phase 4 的教训：依赖型调用要分步，8 步不够
SYSTEMS = ("code", "toolcalling", "upstream")


# ---------- 假回复：和 tests_code_agent.py 同款格式 ----------

def _code_reply(code: str) -> str:
    """给 CodeAgent 的假回复：Thought + 一个 python 代码块。"""
    return f"Thought: 按计划执行。\n```python\n{code}\n```"


def _tool_call_msg(calls: list[dict]) -> ChatMessage:
    """给 ToolCallingAgent 的假回复：一条结构化 tool_calls 消息。"""
    return ChatMessage(
        role="assistant", content="",
        tool_calls=[ToolCall(id=f"call_{i + 1}", name=c["name"],
                             arguments=json.dumps(c["arguments"], ensure_ascii=False))
                    for i, c in enumerate(calls)])


# ---------- upstream（原版）用的四个工具：和自己的工具箱对齐 ----------

def _upstream_tools(workdir: str):
    """只在跑 upstream 系统时才 import smolagents（fake/learner 不依赖它）。"""
    from smolagents import tool as up_tool

    @up_tool
    def calc(expression: str) -> str:
        """Safe arithmetic: only digits and + - * / ** % ( ).

        Args:
            expression: An arithmetic expression, e.g. "128*46".
        """
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError:
            return "Error: bad expression"
        allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                   ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
                   ast.USub, ast.UAdd, ast.Load)
        for node in ast.walk(tree):
            if not isinstance(node, allowed) or isinstance(node, ast.Constant) \
                    and not isinstance(node.value, (int, float)):
                return "Error: only digits and + - * / ** % ( )"
        try:
            return str(eval(compile(tree, "<calc>", "eval"),
                            {"__builtins__": {}}, {}))  # noqa: S307 - AST 白名单已过滤
        except ZeroDivisionError:
            return "Error: division by zero"
        except Exception as e:  # noqa: BLE001
            return f"Error: {e}"

    @up_tool
    def read_file(path: str) -> str:
        """Read a text file under the working directory.

        Args:
            path: Path relative to the working directory.
        """
        full = os.path.abspath(os.path.join(workdir, path.strip().strip('"').strip("'")))
        if full != workdir and not full.startswith(workdir + os.sep):
            return "Error: only files under the working directory"
        try:
            with open(full, encoding="utf-8") as f:
                return f.read(10000) or "(empty)"
        except FileNotFoundError:
            return f"Error: file not found: {path}"
        except Exception as e:  # noqa: BLE001
            return f"Error: {e}"

    @up_tool
    def write_file(path: str, content: str) -> str:
        """Write text to a file under the working directory.

        Args:
            path: Path relative to the working directory.
            content: Text to write into the file.
        """
        full = os.path.abspath(os.path.join(workdir, path.strip().strip('"').strip("'")))
        if full != workdir and not full.startswith(workdir + os.sep):
            return "Error: only files under the working directory"
        try:
            parent = os.path.dirname(full)
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)
            with open(full, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Wrote {len(content)} characters to {path}"
        except Exception as e:  # noqa: BLE001
            return f"Error: {e}"

    @up_tool
    def web_search(query: str) -> str:
        """Search a short summary of a topic via DuckDuckGo instant answers.

        Args:
            query: Search keywords, e.g. "photosynthesis".
        """
        url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode(
            {"q": query.strip().strip('"').strip("'"), "format": "json",
             "no_html": "1", "skip_disambig": "1"})
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "smolagents-repro/0.1"})
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            return f"Search failed: {e}"
        bits = []
        if data.get("AbstractText"):
            bits.append(data["AbstractText"])
        for item in (data.get("RelatedTopics") or [])[:3]:
            if isinstance(item, dict) and item.get("Text"):
                bits.append(item["Text"])
        return "\n".join(bits) if bits else f"No summary found for '{query}'"

    return [calc, read_file, write_file, web_search]


class UpstreamFakeModel:
    """原版 CodeAgent 用的假模型：按顺序回写死的代码块，不联网。

    只实现 generate()——上游的 Model 基类要求这个签名；
    token_usage 给 0（假用量，评测只看步数和成功率，不看它）。
    """

    def __init__(self, replies: list[str]):
        from smolagents.models import Model as UpModel
        # 上游 Model 是普通基类，generate 由子类实现；这里不调 super().__init__
        # 以外的任何东西，避免上游构造器里和模型 ID 有关的校验。
        UpModel.__init__(self, model_id="fake")
        self.replies = list(replies)
        self.calls = 0

    def generate(self, messages, stop_sequences=None, response_format=None,
                 tools_to_call_from=None, **kwargs):
        from smolagents.models import ChatMessage as UpChatMessage
        from smolagents.models import TokenUsage as UpTokenUsage
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return UpChatMessage(role="assistant", content=reply,
                             token_usage=UpTokenUsage(input_tokens=0, output_tokens=0))


# ---------- 模型工厂：每轮一个新模型（FakeModel 的状态不能跨轮） ----------

def fake_model_factory(task: EvalTask, system: str):
    """FakeModel：按题给脚本回答案——证明任务可解、checker 认账，
    不是证明模型会做（见 eval_tasks.py 的说明）。"""
    if system == "code":
        return FakeModel([_code_reply(task.fake_code)])
    if system == "toolcalling":
        replies = [_tool_call_msg(step) for step in task.fake_calls]
        return FakeModel(replies, repeat=False)
    if system == "upstream":
        return UpstreamFakeModel([_code_reply(task.fake_code)])
    raise ValueError(f"未知系统：{system}")


def live_model_factory(task: EvalTask, system: str, model_id: str, api_key: str):
    """真模型：自己的系统走 OpenAICompatibleModel，原版走 OpenAIServerModel。"""
    if system in ("code", "toolcalling"):
        return OpenAICompatibleModel(model_id=model_id, api_base=API_BASE,
                                     api_key=api_key)
    if system == "upstream":
        from smolagents import OpenAIServerModel
        return OpenAIServerModel(model_id=model_id, api_base=API_BASE,
                                 api_key=api_key)
    raise ValueError(f"未知系统：{system}")


# ---------- 跑一轮 ----------

def _count_steps(system: str, result) -> int:
    """ActionStep 的个数 = 模型实际走的步数。

    两边的 RunResult.steps 都是 dict 列表（自己的 get_full_steps()
    和上游的 get_full_steps() 同款），动作步的 dict 带 step_number，
    TaskStep 只有 task、FinalAnswerStep 只有 output——按这个键数。
    """
    return sum(1 for s in (result.steps or [])
               if isinstance(s, dict) and "step_number" in s)


def _tokens(result) -> tuple[int, int]:
    usage = getattr(result, "token_usage", None)
    if usage is None:
        return 0, 0
    return int(usage.input_tokens or 0), int(usage.output_tokens or 0)


def run_one(task: EvalTask, system: str, model, max_steps: int,
            run_dir: str) -> dict:
    """跑一道题 × 一个系统，返回一条记录。异常（包括 429）都记下来，
    不炸整轮——评测要的是每轮的成绩，不是第一轮的报错。"""
    workdir = os.path.join(run_dir, task.id, system)
    os.makedirs(workdir, exist_ok=True)
    for name, content in task.setup.items():
        with open(os.path.join(workdir, name), "w", encoding="utf-8") as f:
            f.write(content)
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    record = {"task_id": task.id, "kind": task.kind, "system": system,
              "model_id": getattr(model, "model_id", "?"),
              "success": False, "state": "error", "steps": 0,
              "tokens_in": 0, "tokens_out": 0, "latency_s": 0.0,
              "error": None, "error_429": False, "answer": ""}
    t0 = time.perf_counter()
    buf = io.StringIO()
    try:
        if system == "code":
            agent = CodeAgent(tools=default_tools(base_dir=workdir), model=model,
                              max_steps=max_steps, log_dir=log_dir,
                              return_full_result=True)
        elif system == "toolcalling":
            agent = ToolCallingAgent(tools=default_tools(base_dir=workdir),
                                     model=model, max_steps=max_steps,
                                     log_dir=log_dir, return_full_result=True)
        elif system == "upstream":
            from smolagents import CodeAgent as UpCodeAgent
            agent = UpCodeAgent(tools=_upstream_tools(workdir), model=model)
        else:
            raise ValueError(f"未知系统：{system}")
        with redirect_stdout(buf):
            if system == "upstream":
                result = agent.run(task.prompt, max_steps=max_steps,
                                   return_full_result=True)
            else:
                result = agent.run(task.prompt, max_steps=max_steps,
                                   return_full_result=True)
        record["state"] = getattr(result, "state", "success")
        record["steps"] = _count_steps(system, result)
        record["tokens_in"], record["tokens_out"] = _tokens(result)
        answer = str(getattr(result, "output", "") or "")
        record["answer"] = answer[:2000]
        # 步数用完强行给的答案也照样打分（state 会记 max_steps_error）。
        record["success"] = bool(task.check(answer, workdir))
    except Exception as e:  # noqa: BLE001 - 评测里异常就是一轮的成绩
        record["error"] = f"{type(e).__name__}: {str(e)[:500]}"
        record["error_429"] = "429" in str(e)
        record["stdout_tail"] = buf.getvalue()[-2000:]
    record["latency_s"] = round(time.perf_counter() - t0, 1)
    return record


# ---------- 选题：--tasks 精确挑，--limit/--offset 切片 ----------

def select_tasks(tasks_arg: str | None, limit: int | None,
                 offset: int) -> list[EvalTask]:
    if tasks_arg:
        return [get_task(tid.strip()) for tid in tasks_arg.split(",") if tid.strip()]
    if limit is None:
        return TASKS[offset:]
    return TASKS[offset:offset + limit]


# ---------- 汇总表 ----------

def summarize(records: list[dict]) -> str:
    lines = ["| task | kind | system | success | steps | tokens in/out | s |",
             "|---|---|---|---|---|---|---|"]
    for r in records:
        ok = "✅" if r["success"] else "❌"
        err = " 429" if r["error_429"] else (" err" if r["error"] else "")
        lines.append(f"| {r['task_id']} | {r['kind']} | {r['system']} | "
                     f"{ok}{err} | {r['steps']} | "
                     f"{r['tokens_in']}/{r['tokens_out']} | {r['latency_s']} |")
    lines.append("")
    for system in SYSTEMS:
        sub = [r for r in records if r["system"] == system]
        if not sub:
            continue
        ok_n = sum(1 for r in sub if r["success"])
        steps = [r["steps"] for r in sub]
        lines.append(f"- {system}: {ok_n}/{len(sub)} 通过，"
                     f"平均步数 {sum(steps) / len(steps):.1f}，"
                     f"总 token {sum(r['tokens_in'] + r['tokens_out'] for r in sub)}，"
                     f"429 共 {sum(1 for r in sub if r['error_429'])} 次")
    return "\n".join(lines)


# ---------- CLI ----------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 7 评测：同一套题跑三个系统")
    parser.add_argument("--model", choices=["fake", "live"], default="fake",
                        help="fake=离线假模型，live=真模型（走 Gemini 免费层）")
    parser.add_argument("--systems", default=",".join(SYSTEMS),
                        help="逗号分隔：code,toolcalling,upstream")
    parser.add_argument("--tasks", default=None,
                        help="只跑这些题，逗号分隔，如 arith-01,file-01")
    parser.add_argument("--limit", type=int, default=None, help="最多跑几道题")
    parser.add_argument("--offset", type=int, default=0, help="从第几道题开始")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--sleep", type=float, default=0,
                        help="每轮之间停几秒（联网跑时用来躲每分钟限流；"
                             "今天 research-02 在 1 分钟内第 5 次调用就撞了 429）")
    parser.add_argument("--out-dir", default=None,
                        help="结果目录，默认 eval_runs/<时间戳>")
    parser.add_argument("--live-model", default=None,
                        help="live 用的模型名，默认环境变量 EVAL_MODEL 或 gemini-2.5-flash")
    args = parser.parse_args(argv)

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    for s in systems:
        if s not in SYSTEMS:
            parser.error(f"未知系统：{s}（可选 {', '.join(SYSTEMS)}）")
    tasks = select_tasks(args.tasks, args.limit, args.offset)
    if not tasks:
        print("没有选中的题。")
        return 2

    surrogate = None
    live_model_id = args.live_model or os.environ.get("EVAL_MODEL", "gemini-2.5-flash")
    if args.model == "live":
        sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
        from dynamic_credentials import dynamic_credential_entry
        entry = dynamic_credential_entry("custom.gemini")
        surrogate = str(entry["surrogate"]).strip()
        if not surrogate.startswith("hsurr:"):
            raise SystemExit("stored Gemini credential did not return a surrogate")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.out_dir or os.path.join(HERE, "eval_runs", stamp)
    os.makedirs(run_dir, exist_ok=True)

    records = []
    total = len(tasks) * len(systems)
    done = 0
    for task in tasks:
        for system in systems:
            done += 1
            if args.model == "live":
                model = live_model_factory(task, system, live_model_id, surrogate)
            else:
                model = fake_model_factory(task, system)
            record = run_one(task, system, model, args.max_steps, run_dir)
            records.append(record)
            flag = "✅" if record["success"] else "❌"
            err = " 429" if record["error_429"] else (
                f" {record['error'][:80]}" if record["error"] else "")
            print(f"[{done}/{total}] {task.id} {system}: {flag} "
                  f"steps={record['steps']} "
                  f"tokens={record['tokens_in']}/{record['tokens_out']} "
                  f"{record['latency_s']}s{err}", flush=True)
            if args.sleep and done < total:
                time.sleep(args.sleep)

    meta = {"model": args.model,
            "live_model_id": live_model_id if args.model == "live" else None,
            "systems": systems, "max_steps": args.max_steps,
            "tasks": [t.id for t in tasks], "run_dir": run_dir,
            "stamp": stamp}
    with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "records": records}, f, ensure_ascii=False, indent=2)

    table = summarize(records)
    print("\n" + table)
    with open(os.path.join(run_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(f"# 评测 {stamp}\n\n模型：{meta['model']}"
                f"{' (' + live_model_id + ')' if args.model == 'live' else ''}\n\n"
                + table + "\n")
    print(f"\n结果：{os.path.join(run_dir, 'results.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
