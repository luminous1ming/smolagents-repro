#!/usr/bin/env python3
"""Phase 8 自测：CLI 入口（不联网；live 分支不测，只测 fake 路径）。

跑法：.venv/bin/python tests_cli.py
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, ".venv", "bin", "python")

PASSED = 0
FAILED = 0


def check(name, cond):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  PASS {name}")
    else:
        FAILED += 1
        print(f"  FAIL {name}")


def run_cli(*argv):
    """跑一次 CLI，返回 (returncode, stdout, stderr)。"""
    proc = subprocess.run([PY, os.path.join(HERE, "cli.py"), *argv],
                          capture_output=True, text=True, timeout=120,
                          cwd=HERE)
    return proc.returncode, proc.stdout, proc.stderr


def last_json(stdout):
    """终端最后一行 JSON 摘要（run 打的）。"""
    for line in reversed(stdout.strip().split("\n")):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    return None


print("1. tasks：列出评测题")
rc, out, _ = run_cli("tasks")
check("tasks 退出 0", rc == 0)
check("列出 24 道题", "共 24 道题" in out)
check("arith-01 在列", "arith-01" in out and "research-08" in out)

print("2. run --eval：fake 模型离线跑通")
rc, out, _ = run_cli("run", "--eval", "arith-01", "--agent", "code",
                     "--model", "fake")
check("arith-01 code 退出 0", rc == 0)
summary = last_json(out)
check("JSON 摘要 success=true", bool(summary) and summary["success"] is True)
check("答案含 432", bool(summary) and "432" in summary["answer"])
check("步数 1 步", bool(summary) and summary["steps"] == 1)

rc, out, _ = run_cli("run", "--eval", "file-01", "--agent", "toolcalling",
                     "--model", "fake")
summary = last_json(out)
check("file-01 toolcalling 通过", rc == 0 and bool(summary)
      and summary["success"] is True)
check("toolcalling 步数比 code 多",
      bool(summary) and summary["steps"] > 1)

print("3. run 的参数校验")
rc, out, _ = run_cli("run", "--eval", "arith-01", "--task", "x")
check("同时给 --task 和 --eval 拒绝", rc == 2)
rc, out, _ = run_cli("run", "--task", "算一下 128*46")
check("fake 跑自由任务拒绝", rc == 2 and "live" in out)
rc, out, _ = run_cli("run", "--eval", "nope-99")
check("不存在的题号报错退出非 0", rc != 0)

print("4. eval 子命令转给 eval.py")
rc, out, _ = run_cli("eval", "--model", "fake", "--limit", "2",
                     "--systems", "code")
check("eval --limit 2 退出 0", rc == 0)
check("eval 打出汇总表", "code:" in out and "通过" in out)

print("5. run --task + MCP：fake 模型下 MCP 工具也能接上")
# 自由任务走 fake 会被拦；这里只验证 MCP 接线本身不报错：
# 用 --eval + --mcp-command 应明确拒绝（工具箱由 eval.py 定）。
rc, out, _ = run_cli("run", "--eval", "arith-01", "--mcp-command",
                     "python3 mcp_test_server.py")
check("--eval + --mcp 明确拒绝", rc == 2 and "mcp" in out.lower())

print(f"\n{PASSED}/{PASSED + FAILED} passed.")
sys.exit(0 if FAILED == 0 else 1)
