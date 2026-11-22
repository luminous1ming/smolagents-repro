#!/usr/bin/env python3
"""Phase 2：默认工具 —— mini-agent 的工具集，用 Phase 2 的风格重写。

对照上游（smolagents 1.26.0 default_tools.py）：
  - FinalAnswerTool：name="final_answer"，inputs 只有一个 answer（any 类型），
    forward 原样返回答案——两个 agent 都靠它结束循环。
  - DuckDuckGoSearchTool：name="web_search"；上游用 ddgs 包，
    这里沿用 mini-agent 的标准库做法（DuckDuckGo 即时答案接口），
    不多引依赖。

Phase 0 的教训写进设计里：CodeAgent 的执行器默认禁 open()/import os，
文件访问只能走工具——所以 read_file 是正式的工具（带 base_dir，
只读 base_dir 里面的文件），而不是让模型自己 open。

两种定义方式都演示一遍（上游也是两种混用）：
  - @tool 装饰器：calc、web_search（无状态的纯函数）
  - Tool 子类：ReadFileTool、WriteFileTool（要带 base_dir）、FinalAnswerTool（照着上游写）
"""
import ast
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from tools import Tool, ToolRegistry, tool

# ---------- @tool：无状态工具 ----------


@tool
def calc(expression: str) -> str:
    """安全算术：只允许数字和 + - * / ** % ( )，拒绝任何名字和调用。

    Args:
        expression: 算术表达式，例如 "128*46"。

    Returns:
        计算结果的字符串；写错了、除以 0 或用了不允许的东西时返回错误说明。
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return "错误：表达式写错了"
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
               ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
               ast.USub, ast.UAdd, ast.Load)
    for node in ast.walk(tree):
        if not isinstance(node, allowed) \
                or isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            return "错误：只支持数字和 + - * / ** % ( )"
    try:
        return str(eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, {}))  # noqa: S307 - 已用 ast 白名单过滤
    except ZeroDivisionError:
        return "错误：除数不能为 0"
    except Exception as e:  # noqa: BLE001 - 工具错误要变成 Observation 告诉模型
        return f"错误：{e}"


@tool
def web_search(query: str) -> str:
    """DuckDuckGo 即时答案，免 key；搜不到就如实返回，模型会自己换办法。

    Args:
        query: 搜索关键词，例如 "光合作用"。

    Returns:
        搜到的简介文本；没搜到或请求失败时返回说明原因的字符串。
    """
    query = query.strip().strip('"').strip("'")
    url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode(
        {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"})
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "smolagents-repro/0.1"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001 - 搜索失败也要变成 Observation
        return f"搜索失败：{e}"
    bits = []
    if data.get("AbstractText"):
        bits.append(data["AbstractText"])
    for item in (data.get("RelatedTopics") or [])[:3]:
        if isinstance(item, dict) and item.get("Text"):
            bits.append(item["Text"])
    return "\n".join(bits) if bits else f"没搜到“{query}”的简介，换个关键词试试"


# ---------- Tool 子类：有状态 / 照着上游写的工具 ----------


class ReadFileTool(Tool):
    """读 base_dir 里的文本文件（Phase 0 的教训：文件访问走工具）。

    mini-agent 的 tool_read_file 原样搬过来，只把“当前目录”换成
    构造时传进来的 base_dir——这样测试能用临时目录，agent 跑起来
    模型也读不到 base_dir 外面去。
    """

    name = "read_file"
    description = "读 base_dir 下的文本文件，例如 read_file(notes/demo-domain.md)"
    inputs = {"path": {"type": "string",
                       "description": "相对于 base_dir 的文件路径"}}
    output_type = "string"

    def __init__(self, base_dir: str | None = None):
        self.base_dir = os.path.abspath(base_dir or os.getcwd())
        super().__init__()

    def forward(self, path: str) -> str:
        path = path.strip().strip('"').strip("'")
        full = os.path.abspath(os.path.join(self.base_dir, path))
        if full != self.base_dir and not full.startswith(self.base_dir + os.sep):
            return "错误：只能读 base_dir 下的文件"
        try:
            with open(full, encoding="utf-8") as f:
                return f.read(10000) or "(空文件)"
        except FileNotFoundError:
            return f"错误：找不到文件 {path}"
        except Exception as e:  # noqa: BLE001
            return f"错误：{e}"


class WriteFileTool(Tool):
    """写 base_dir 里的文本文件（ReadFileTool 的写 counterpart）。

    Phase 7 评测的文件任务需要它：模型要能把算好的结果写下来，
    再读回来确认。同样只认 base_dir 里面——写出去会被拒；
    父目录不存在时在 base_dir 内自动建（和 open("w") 的区别是
    不会因为缺目录而报错，模型少走一步）。
    """

    name = "write_file"
    description = "写文本到 base_dir 下的文件，例如 write_file(out.txt, 'hello')"
    inputs = {"path": {"type": "string",
                       "description": "相对于 base_dir 的文件路径"},
              "content": {"type": "string",
                          "description": "要写入的文本内容"}}
    output_type = "string"

    def __init__(self, base_dir: str | None = None):
        self.base_dir = os.path.abspath(base_dir or os.getcwd())
        super().__init__()

    def forward(self, path: str, content: str) -> str:
        path = path.strip().strip('"').strip("'")
        full = os.path.abspath(os.path.join(self.base_dir, path))
        if full != self.base_dir and not full.startswith(self.base_dir + os.sep):
            return "错误：只能写 base_dir 下的文件"
        try:
            parent = os.path.dirname(full)
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)
            with open(full, "w", encoding="utf-8") as f:
                f.write(content)
            return f"已写入 {path}（{len(content)} 字符）"
        except Exception as e:  # noqa: BLE001
            return f"错误：{e}"


class FinalAnswerTool(Tool):
    """结束工具（上游 FinalAnswerTool 的原样复现）。

    forward 不干活、原样返回答案；Phase 3/4 的循环看到模型调了它，
    就知道这一轮可以结束了（上游就是按名字认它的）。
    """

    name = "final_answer"
    description = "Provides a final answer to the given problem."
    inputs = {"answer": {"type": "any", "description": "The final answer to the problem"}}
    output_type = "any"

    def forward(self, answer: Any) -> Any:
        return answer


# ---------- 一键装好默认工具箱 ----------


def default_tools(base_dir: str | None = None) -> ToolRegistry:
    """默认五件套：calc、read_file、write_file、web_search、final_answer。

    Args:
        base_dir: read_file / write_file 允许读写的目录，默认当前工作目录。
    """
    registry = ToolRegistry()
    registry.register(calc)
    registry.register(ReadFileTool(base_dir=base_dir))
    registry.register(WriteFileTool(base_dir=base_dir))
    registry.register(web_search)
    registry.register(FinalAnswerTool())
    return registry
