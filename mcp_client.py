#!/usr/bin/env python3
"""Phase 8：MCP 客户端 —— 把一个 stdio MCP server 包成工具源。

MCP（Model Context Protocol）的 stdio 传输很简单：进程的标准输入/
标准输出上传 JSON-RPC 2.0 消息，每行一条（NDJSON），不用 Content-Length
包头（那是 LSP 的做法）。本文件只用标准库。

用法：
    client = MCPClient(["python3", "mcp_test_server.py"])
    registry = client.tools()          # ToolRegistry，可直接塞给 agent
    client.call_tool("mcp_calc", {"expression": "128*46"})  # -> "5888"
    client.close()

协议流程（按 MCP 规范的最小集）：
    1. 发 initialize（带协议版本 + 客户端信息），等结果；
    2. 发 notifications/initialized（通知，不用等回复）；
    3. 发 tools/list 拿工具列表；发 tools/call 调工具。
服务端可能先发通知或 ping 请求，本客户端读到不认识的消息就跳过
（只认对上 id 的回复）；读到 EOF 说明 server 挂了，直接报错。

注意：MCPTool.forward 是同步调子进程的 stdin/stdout——CodeAgent 在
本地执行器里调它没问题（工具就是普通 Python 函数）；但它进不了
DockerExecutor，因为容器里没有这个子进程（Phase 6 的 managed_agents
同款限制，见 notes/phase6-sandbox.md）。
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools import Tool, ToolRegistry

PROTOCOL_VERSION = "2024-11-05"
CLIENT_INFO = {"name": "smolagents-repro", "version": "0.1.0"}


class MCPError(Exception):
    """MCP 协议层面的错误（server 返回 error、EOF、超时）。"""


class MCPToolError(Exception):
    """调 MCP 工具时 server 说工具本身出错了（isError=true）。"""


class MCPClient:
    """一个 stdio MCP server 的客户端。

    Args:
        command: 启动 server 的命令，如 ["python3", "server.py"] 或
            ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"]。
        timeout: 每条请求等回复的最长秒数。
    """

    def __init__(self, command: list[str], timeout: int = 30):
        self.command = list(command)
        self.timeout = timeout
        self._next_id = 1
        self._proc = subprocess.Popen(
            self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1)
        self._initialized = False

    # ---------- 底层：发一条请求，等一条回复 ----------

    def _send(self, payload: dict) -> None:
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        try:
            self._proc.stdin.write(line)
            self._proc.stdin.flush()
        except BrokenPipeError as e:
            raise MCPError(f"MCP server 已退出，写不进去：{self.command}") from e

    def _read_reply(self, want_id: int) -> dict:
        """读到 id 对上的回复为止；通知和 server 发来的请求直接跳过。"""
        import select
        fd = self._proc.stdout.fileno()
        buf = ""
        while True:
            ready, _, _ = select.select([fd], [], [], self.timeout)
            if not ready:
                raise MCPError(f"等 MCP 回复超时（{self.timeout}s）：{self.command}")
            chunk = os.read(fd, 65536).decode("utf-8", errors="replace")
            if not chunk:  # EOF：server 挂了
                err = self._proc.stderr.read() if self._proc.stderr else ""
                raise MCPError(f"MCP server 退出了：{self.command}\n{err[-500:]}")
            buf += chunk
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                msg = json.loads(line)
                if msg.get("id") == want_id:
                    return msg
                # 通知 / server 发来的请求：本客户端不处理，直接跳过。

    def _request(self, method: str, params: dict | None = None) -> dict:
        req_id = self._next_id
        self._next_id += 1
        payload = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            payload["params"] = params
        self._send(payload)
        reply = self._read_reply(req_id)
        if "error" in reply:
            err = reply["error"]
            raise MCPError(f"MCP {method} 失败：{err.get('message', err)}")
        return reply.get("result", {})

    def _notify(self, method: str, params: dict | None = None) -> None:
        payload = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self._send(payload)

    # ---------- 协议：握手、列工具、调工具 ----------

    def initialize(self) -> dict:
        """握手一次；重复调直接返回上次的结果。"""
        if self._initialized:
            return self._server_info
        result = self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": CLIENT_INFO,
        })
        self._notify("notifications/initialized")
        self._server_info = result
        self._initialized = True
        return result

    def list_tools(self) -> list[dict]:
        """tools/list 的原始工具描述列表。"""
        self.initialize()
        result = self._request("tools/list")
        return result.get("tools", [])

    def call_tool(self, name: str, arguments: dict | None = None) -> str:
        """调一个 MCP 工具，返回所有 text 内容拼起来的字符串。"""
        self.initialize()
        result = self._request("tools/call", {
            "name": name, "arguments": arguments or {}})
        if result.get("isError"):
            texts = [c.get("text", "") for c in result.get("content", [])
                     if c.get("type") == "text"]
            raise MCPToolError(f"MCP 工具 {name} 报错：{''.join(texts) or '未知错误'}")
        texts = [c.get("text", "") for c in result.get("content", [])
                 if c.get("type") == "text"]
        return "\n".join(texts)

    def tools(self) -> ToolRegistry:
        """把 server 的全部工具包成 ToolRegistry（供 agent 直接用）。"""
        registry = ToolRegistry()
        for spec in self.list_tools():
            registry.register(MCPTool(self, spec))
        return registry

    def close(self) -> None:
        """关掉 server 进程（先给它发 SIGTERM，不等它慢慢退）。"""
        try:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
        except Exception:  # noqa: BLE001 - 关闭时不抛错
            pass

    def __enter__(self) -> "MCPClient":
        self.initialize()
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _schema_to_inputs(schema: dict) -> tuple[dict, list[str]]:
    """MCP 的 inputSchema（JSON Schema）→ Tool 的 inputs + 必填列表。

    只认 type: object + properties；每个属性取 type 和 description，
    没写 type 的按 "any" 处理（和 Phase 2 的约定一致：JSON 工具参数里
    不发 "any"，见 tools.py 的 to_openai_function）。不在
    AUTHORIZED_TYPES 里的类型也落到 "any"。inputSchema 的 required
    之外的参数标 "nullable"，这样 validate_tool_arguments 不会把
    可选参数当成必填（见 tools.py 的 required 判定）。
    """
    from tools import AUTHORIZED_TYPES
    props = (schema or {}).get("properties") or {}
    required = list((schema or {}).get("required") or [])
    inputs = {}
    for name, spec in props.items():
        jtype = spec.get("type", "any")
        if jtype not in AUTHORIZED_TYPES:
            jtype = "any"
        entry = {"type": jtype, "description": spec.get("description", "")}
        if name not in required:
            entry["nullable"] = True
        inputs[name] = entry
    return inputs, required


class MCPTool(Tool):
    """一个 MCP 工具在本地的替身：forward 就是一次 tools/call。

    注意 Tool 的两条规矩（见 tools.py）：
    - 属性必须在 super().__init__() 之前就设好（它构造时就校验）；
    - forward 的签名必须和 inputs 的键一一对应。MCP 工具的参数名是
      运行时才知道的，所以这里动态拼一个同签名的 forward（用
      __signature__ 声明参数名），而不是写死的 **kwargs。
    """

    def __init__(self, client: MCPClient, spec: dict):
        import inspect
        self.client = client
        self.spec = spec
        inputs, required = _schema_to_inputs(spec.get("inputSchema") or {})
        self.name = spec["name"]
        self.description = spec.get("description", "")
        self.inputs = inputs
        self.output_type = "string"
        self._required = required

        def _forward(**kwargs):
            return self.client.call_tool(self.name, dict(kwargs))

        # 可选参数（不在 required 里）给默认 None，否则签名层面就必填，
        # validate_tool_arguments 的 nullable 兜底永远走不到。
        _forward.__signature__ = inspect.Signature(parameters=[
            inspect.Parameter(
                n, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=None if n not in required else inspect.Parameter.empty)
            for n in inputs])
        self.forward = _forward
        super().__init__()


def mcp_tools(command: list[str], timeout: int = 30) -> tuple[MCPClient, ToolRegistry]:
    """一句话接一个 MCP server：返回 (client, 工具注册表)。

    用完记得 client.close()，否则子进程会留着。"""
    client = MCPClient(command, timeout=timeout)
    return client, client.tools()
