#!/usr/bin/env python3
"""Phase 8：命令行入口 —— 一条命令跑任务、跑评测。

子命令：
    run    跑一个任务（--eval 评测题 或 --task 自由文本）
    eval   评测 harness（直接转给 eval.py，见 Phase 7）
    tasks  列出全部评测题

跑法（都在本目录，用项目 venv；不联网的演示不用 key）：
    .venv/bin/python cli.py tasks
    .venv/bin/python cli.py run --eval arith-01 --agent code --model fake
    .venv/bin/python cli.py run --eval file-01 --agent toolcalling --model fake
    .venv/bin/python cli.py eval --model fake --limit 2 --systems code
    .venv/bin/python cli.py run --task "算一下 128*46" --model live
    .venv/bin/python cli.py run --task "算一下 128*46" --model live \\
        --mcp-command "python3 mcp_test_server.py"

--model fake 只能跑 --eval 的题（FakeModel 按题给脚本回答案，见
eval_tasks.py）；自由文本任务要走 --model live（真模型）。
--model live 的 key 从 Secure Vault 取（和 eval.py 同款做法：
dynamic_credentials 换 surrogate，只放内存，从不写文件）。
联网跑完会在终端打一行 JSON 摘要，并写进 live_checks/cli-<时间戳>.json，
方便对照 Phase 7 的抽查记录。
"""
import argparse
import json
import os
import shlex
import sys
import tempfile
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import CodeAgent, ToolCallingAgent
from default_tools import default_tools
from eval_tasks import TASKS, get_task
from models import OpenAICompatibleModel

HERE = os.path.dirname(os.path.abspath(__file__))
API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_MAX_STEPS = 15  # Phase 4 的教训：依赖型调用要分步，8 步不够


def _live_surrogate() -> str:
    """从 Secure Vault 换 Gemini 的 surrogate（只放内存）。"""
    sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
    from dynamic_credentials import dynamic_credential_entry
    entry = dynamic_credential_entry("custom.gemini")
    surrogate = str(entry["surrogate"]).strip()
    if not surrogate.startswith("hsurr:"):
        raise SystemExit("stored Gemini credential did not return a surrogate")
    return surrogate


def _build_agent(agent_name: str, tools, model, max_steps: int, log_dir: str):
    if agent_name == "code":
        return CodeAgent(tools=tools, model=model, max_steps=max_steps,
                         log_dir=log_dir, return_full_result=True)
    if agent_name == "toolcalling":
        return ToolCallingAgent(tools=tools, model=model, max_steps=max_steps,
                                log_dir=log_dir, return_full_result=True)
    raise ValueError(f"未知 agent：{agent_name}（可选 code / toolcalling）")


def _count_steps(result) -> int:
    return sum(1 for s in (result.steps or [])
               if isinstance(s, dict) and "step_number" in s)


def _tokens(result) -> tuple[int, int]:
    usage = getattr(result, "token_usage", None)
    if usage is None:
        return 0, 0
    return int(usage.input_tokens or 0), int(usage.output_tokens or 0)


def _cmd_run(args) -> int:
    """run：跑一个任务，打印答案 + 一行 JSON 摘要。"""
    if bool(args.task) == bool(args.eval):
        print("run 要且只要一个：--task 自由文本 或 --eval 评测题 id")
        return 2
    if args.task and args.model == "fake":
        print("fake 模型没有自由回复，只能跑 --eval 的题；"
              "自由文本任务请用 --model live。")
        return 2
    if args.eval and args.mcp_command:
        print("--mcp-command 目前只支持 --task（评测题的工具箱由 eval.py 定）")
        return 2

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    workdir = args.workdir or tempfile.mkdtemp(prefix="repro-cli-")
    os.makedirs(workdir, exist_ok=True)
    log_dir = os.path.join(workdir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    mcp_client = None
    tools = default_tools(base_dir=workdir)
    if args.mcp_command:
        from mcp_client import MCPClient
        mcp_client = MCPClient(shlex.split(args.mcp_command))
        mcp_registry = mcp_client.tools()
        for name in mcp_registry.names():
            tools.register(mcp_registry[name])
        print(f"MCP 已接上：{tools.names()}")

    try:
        if args.eval:
            # 评测题：复用 eval.py 的工厂和 run_one（workdir/打分都在里面）。
            import eval as eval_module
            task = get_task(args.eval)
            system = "code" if args.agent == "code" else "toolcalling"
            run_dir = os.path.join(HERE, "eval_runs", f"cli-{stamp}")
            if args.model == "live":
                model = eval_module.live_model_factory(
                    task, system, args.live_model, _live_surrogate())
            else:
                model = eval_module.fake_model_factory(task, system)
            record = eval_module.run_one(task, system, model,
                                         args.max_steps, run_dir)
            summary = {"task": task.id, "kind": task.kind,
                       "agent": args.agent, "model": args.model,
                       "live_model": args.live_model if args.model == "live" else None,
                       "steps": record["steps"],
                       "tokens_in": record["tokens_in"],
                       "tokens_out": record["tokens_out"],
                       "latency_s": record["latency_s"],
                       "success": record["success"],
                       "state": record["state"],
                       "answer": record["answer"][:500],
                       "run_dir": run_dir}
            print(f"答案：{record['answer'][:500]}")
            print(f"结果：{'✅ 通过' if record['success'] else '❌ 没过'}"
                  f"（{record['steps']} 步，{record['latency_s']}s）")
        else:
            if args.model == "live":
                model = OpenAICompatibleModel(
                    model_id=args.live_model, api_base=API_BASE,
                    api_key=_live_surrogate())
            else:  # pragma: no cover - 上面已拦住 fake + --task
                raise SystemExit("fake 模型跑不了自由任务")
            agent = _build_agent(args.agent, tools, model,
                                 args.max_steps, log_dir)
            t0 = time.perf_counter()
            result = agent.run(args.task, max_steps=args.max_steps,
                               return_full_result=True)
            latency = round(time.perf_counter() - t0, 1)
            steps = _count_steps(result)
            tokens_in, tokens_out = _tokens(result)
            answer = str(getattr(result, "output", "") or "")
            summary = {"task": args.task, "agent": args.agent,
                       "model": args.model, "live_model": args.live_model,
                       "steps": steps, "tokens_in": tokens_in,
                       "tokens_out": tokens_out, "latency_s": latency,
                       "success": result.state == "success",
                       "state": result.state, "answer": answer[:500],
                       "workdir": workdir}
            print(f"答案：{answer[:500]}")
            print(f"结果：{result.state}（{steps} 步，{latency}s）")
    finally:
        if mcp_client is not None:
            mcp_client.close()

    print(json.dumps(summary, ensure_ascii=False))
    if args.model == "live":
        os.makedirs(os.path.join(HERE, "live_checks"), exist_ok=True)
        record_path = os.path.join(HERE, "live_checks", f"cli-{stamp}.json")
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"记录：{record_path}")
    return 0


def _cmd_tasks(args) -> int:
    for t in TASKS:
        print(f"{t.id:14} {t.kind:9} {t.prompt[:60]}")
    print(f"\n共 {len(TASKS)} 道题（算术/文件/研究各 8 道）。")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="smolagents 复现：命令行入口（run / eval / tasks）")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="跑一个任务")
    p_run.add_argument("--task", default=None, help="自由文本任务（要 --model live）")
    p_run.add_argument("--eval", default=None, help="评测题 id，如 arith-01")
    p_run.add_argument("--agent", choices=["code", "toolcalling"], default="code")
    p_run.add_argument("--model", choices=["fake", "live"], default="fake",
                       help="fake=离线假模型，live=真模型（走 Gemini 免费层）")
    p_run.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    p_run.add_argument("--workdir", default=None,
                       help="文件工具的读写目录，默认临时目录")
    p_run.add_argument("--live-model", default=None,
                       help="live 用的模型名，默认环境变量 EVAL_MODEL 或 gemini-2.5-flash")
    p_run.add_argument("--mcp-command", default=None,
                       help="接一个 MCP server，如 \"python3 mcp_test_server.py\"")
    p_run.set_defaults(func=_cmd_run)

    p_eval = sub.add_parser("eval", help="跑评测（转给 eval.py）")
    # eval 不走 func：main() 里把剩余参数原样交给 eval.main()。

    p_tasks = sub.add_parser("tasks", help="列出全部评测题")
    p_tasks.set_defaults(func=_cmd_tasks)

    args, rest = parser.parse_known_args(argv)
    if args.command == "eval":
        import eval as eval_module
        return eval_module.main(rest)
    if getattr(args, "live_model", None) is None:
        args.live_model = os.environ.get("EVAL_MODEL", "gemini-2.5-flash")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
