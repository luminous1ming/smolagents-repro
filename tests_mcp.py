#!/usr/bin/env python3
"""Phase 8 自测：MCP 客户端（不联网，server 是本地的 mcp_test_server.py）。

验收：CodeAgent 用 MCP 工具 mcp_calc 完成一道评测题（arith-01）。
跑法：.venv/bin/python tests_mcp.py
"""
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import CodeAgent
from eval_tasks import get_task
from mcp_client import MCPClient, MCPError, MCPToolError, mcp_tools
from models import FakeModel

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = [sys.executable, os.path.join(HERE, "mcp_test_server.py")]

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


def _expect_raises(exc, fn):
    try:
        fn()
    except exc:
        return True
    except Exception as e:  # noqa: BLE001 - 错的异常也算失败
        print(f"    (got {type(e).__name__}: {e})")
        return False
    return False


# ---------- 1. 握手 + 列工具 ----------

print("1. 握手与 tools/list")
client = MCPClient(SERVER)
info = client.initialize()
check("initialize 返回 serverInfo", info.get("serverInfo", {}).get("name") == "mcp-test-server")
check("initialize 幂等（第二次直接返回）", client.initialize() is info)
specs = client.list_tools()
names = [s["name"] for s in specs]
check("列出 mcp_calc 和 mcp_echo", names == ["mcp_calc", "mcp_echo"])
calc_spec = specs[0]
check("mcp_calc 的 schema 是 expression 必填",
      calc_spec["inputSchema"]["required"] == ["expression"])

# ---------- 2. 调工具 ----------

print("2. tools/call")
check("mcp_calc 算出 5888",
      client.call_tool("mcp_calc", {"expression": "128*46"}) == "5888")
check("mcp_echo 带前缀", client.call_tool(
    "mcp_echo", {"text": "hi", "prefix": ">> "}) == ">> hi")
check("mcp_echo 不带前缀也行", client.call_tool("mcp_echo", {"text": "hi"}) == "hi")
check("未知工具报 MCPError",
      _expect_raises(MCPError, lambda: client.call_tool("nope", {})))
check("工具内部报错（1/0）报 MCPToolError",
      _expect_raises(MCPToolError,
                     lambda: client.call_tool("mcp_calc", {"expression": "1/0"})))
check("未知方法报 MCPError",
      _expect_raises(MCPError, lambda: client._request("nope/method")))

# ---------- 3. 注册表接线 ----------

print("3. ToolRegistry 接线")
registry = client.tools()
check("注册表里有两个工具", registry.names() == ["mcp_calc", "mcp_echo"])
check("注册表调 mcp_calc", registry.call("mcp_calc", {"expression": "2+3"}) == "5")
check("缺必填参数报 ValueError",
      _expect_raises(ValueError,
                     lambda: registry.call("mcp_calc", {})))
check("可选参数不传也行（nullable）",
      registry.call("mcp_echo", {"text": "x"}) == "x")
check("mcp_calc 在代码提示词里是 def",
      "def mcp_calc(expression: string) -> string:" in registry.to_code_prompt())
client.close()
check("close 后子进程退了", client._proc.poll() is not None)

# ---------- 4. 验收：CodeAgent 用 MCP 工具完成评测题 ----------

print("4. 验收：CodeAgent + mcp_calc 完成 arith-01")
task = get_task("arith-01")
client2, mcp_registry = mcp_tools(SERVER)
script = ("Thought: 用 MCP 的计算器算。\n```python\n"
          "result = mcp_calc(expression=\"17*23+41\")\n"
          "final_answer(result)\n```")
tmp = tempfile.mkdtemp(prefix="mcp-eval-")
agent = CodeAgent(tools=mcp_registry, model=FakeModel([script]),
                  max_steps=15, log_dir=tmp, return_full_result=True)
result = agent.run(task.prompt, max_steps=15, return_full_result=True)
check("状态 success", result.state == "success")
check("步数 1 步", sum(1 for s in result.steps
                       if isinstance(s, dict) and "step_number" in s) == 1)
check("checker 认答案 432", task.check(str(result.output), tmp))
client2.close()

# ---------- 5. server 挂了 ----------

print("5. server 挂了要明确报错，不是死等")
dead = MCPClient([sys.executable, "-c", "pass"], timeout=5)
check("立即退出的 server 报 MCPError",
      _expect_raises(MCPError, dead.list_tools))
dead.close()

print(f"\n{PASSED}/{PASSED + FAILED} passed.")
sys.exit(0 if FAILED == 0 else 1)
