#!/usr/bin/env python3
"""测试用的 MCP server（只给 tests_mcp.py 和 CLI 演示用，不联网）。

stdio 传输：stdin/stdout 上每行一条 JSON-RPC 2.0 消息。
提供两个工具：
  - mcp_calc(expression): 安全算术（复用 default_tools.calc 的逻辑），
    算不出时按 MCP 惯例回 isError=true；
  - mcp_echo(text, prefix?): 原样拼回 "prefix + text"，prefix 可选——
    供测试“可选参数标 nullable”用。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from default_tools import calc


def _result(req_id, result):
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _error(req_id, code, message):
    return {"jsonrpc": "2.0", "id": req_id,
            "error": {"code": code, "message": message}}


def _tools_list():
    return {"tools": [
        {"name": "mcp_calc",
         "description": "Safe arithmetic: only digits and + - * / ** % ( ).",
         "inputSchema": {
             "type": "object",
             "properties": {
                 "expression": {"type": "string",
                                "description": "Arithmetic expression, e.g. 128*46"}},
             "required": ["expression"]}},
        {"name": "mcp_echo",
         "description": "Echo text back, optionally with a prefix.",
         "inputSchema": {
             "type": "object",
             "properties": {
                 "text": {"type": "string", "description": "Text to echo"},
                 "prefix": {"type": "string",
                            "description": "Optional prefix (may be omitted)"}},
             "required": ["text"]}},
    ]}


def _tools_call(params):
    name = params.get("name")
    args = params.get("arguments") or {}
    if name == "mcp_calc":
        out = calc(args.get("expression", ""))
        if out.startswith(("Error:", "错误：")):
            return {"content": [{"type": "text", "text": out}], "isError": True}
        return {"content": [{"type": "text", "text": out}]}
    if name == "mcp_echo":
        out = (args.get("prefix") or "") + args.get("text", "")
        return {"content": [{"type": "text", "text": out}]}
    return None  # 未知工具，调用方回 -32602


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "method" not in msg:
            continue  # 回复类消息：server 不该收到，跳过
        method = msg["method"]
        req_id = msg.get("id")
        if req_id is None:
            continue  # 通知（比如 notifications/initialized）：不用回
        if method == "initialize":
            reply = _result(req_id, {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mcp-test-server", "version": "0.1.0"}})
        elif method == "tools/list":
            reply = _result(req_id, _tools_list())
        elif method == "tools/call":
            result = _tools_call(msg.get("params") or {})
            if result is None:
                reply = _error(req_id, -32602,
                               f"Unknown tool: {(msg.get('params') or {}).get('name')}")
            else:
                reply = _result(req_id, result)
        else:
            reply = _error(req_id, -32601, f"Method not found: {method}")
        sys.stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
