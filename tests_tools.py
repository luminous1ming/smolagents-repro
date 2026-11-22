#!/usr/bin/env python3
"""Phase 2 自测：工具系统（不联网；联网检查在 run_live_check_tools.py）。

Run from the repro directory:
    .venv/bin/python tests_tools.py

测五件事：
  1. schema 推导：签名 + docstring + 类型提示 → name/描述/JSON Schema；
     int/float/bool/list/dict/Optional/Literal/choices 都对；
     缺 docstring、缺参数描述、缺类型提示、缺返回提示都报错；
  2. Tool 校验：坏名字、inputs 缺字段、forward 签名对不上、坏 output_type，
     实例化时就报错；
  3. 参数校验：未知参数、缺必填、类型不对、enum 之外、单个值简写、
     number 接受 int、boolean 不能当 integer；
  4. 五个默认工具的行为：calc 算术白名单、read_file 只读 base_dir、
     write_file 只写 base_dir（自动建子目录）、
     web_search（mock 掉网络）、final_answer 原样返回；
  5. 注册表：定义一次，到处能用——直接调、dict 调、JSON 字符串调、
     导出 OpenAI tools 参数、导出代码定义；重名注册和未知工具报错。
"""
import json
import os
import sys
import tempfile
import urllib.error
from typing import Literal, Optional
from unittest import mock

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import tools
from tools import (DocstringParsingException, Tool, ToolRegistry,
                   TypeHintParsingException, get_json_schema, tool,
                   validate_tool_arguments)
from default_tools import (FinalAnswerTool, ReadFileTool, WriteFileTool, calc,
                           default_tools, web_search)

PASSED = 0
FAILED = 0


def check(name, ok):
    global PASSED, FAILED
    if ok:
        PASSED += 1
        print(f"PASS {name}")
    else:
        FAILED += 1
        print(f"FAIL {name}")


def raises(name, exc_type, fn):
    try:
        fn()
    except exc_type:
        check(name, True)
    except Exception:
        check(name, False)
    else:
        check(name, False)


# ---------- 1. schema 推导 ----------

# 下面直接断言工具实例上的推导结果（这才是 @tool 真正产出的东西）。
check("calc 工具名", calc.name == "calc")
check("calc 描述来自 docstring", "安全算术" in calc.description)
check("calc inputs 从签名推导",
      calc.inputs == {"expression": {"type": "string", "description": "算术表达式，例如 \"128*46\"。"}})
check("calc output_type 是 string", calc.output_type == "string")
check("web_search 工具名与类型",
      web_search.name == "web_search" and web_search.output_type == "string"
      and web_search.inputs["query"]["type"] == "string")


def _sample_fn(count: int, ratio: float, verbose: bool,
               tags: list[str], config: dict[str, int],
               mode: Optional[str] = None,
               level: Literal["low", "high"] = "low") -> str:
    """示例函数：覆盖几种类型提示。

    Args:
        count: 整数参数。
        ratio: 浮点参数。
        verbose: 布尔参数。
        tags: 字符串列表。
        config: 字典参数。
        mode: 可空的字符串参数。
        level: 只能是 low 或 high。

    Returns:
        返回字符串。
    """
    return ""


sample = get_json_schema(_sample_fn)["function"]
props = sample["parameters"]["properties"]
check("int/float/bool 类型映射",
      props["count"]["type"] == "integer" and props["ratio"]["type"] == "number"
      and props["verbose"]["type"] == "boolean")
check("list/dict 类型映射",
      props["tags"] == {"type": "array", "items": {"type": "string"},
                        "description": "字符串列表。"}
      and props["config"] == {"type": "object",
                              "additionalProperties": {"type": "integer"},
                              "description": "字典参数。"})
check("Optional 标 nullable", props["mode"].get("nullable") is True
      and props["mode"]["type"] == "string")
check("Literal 变成 enum",
      props["level"].get("enum") == ["low", "high"])
check("有默认值的参数不进 required",
      sample["parameters"]["required"] == ["count", "ratio", "verbose", "tags", "config"])
check("返回值进 return", sample["return"]["type"] == "string")


def _choices_fn(drink: str) -> str:
    """喝点东西。

    Args:
        drink: 想喝的饮料 (choices: ["tea", "coffee"])

    Returns:
        饮料名。
    """
    return drink


check("docstring choices 变成 enum",
      get_json_schema(_choices_fn)["function"]["parameters"]["properties"]["drink"]["enum"]
      == ["tea", "coffee"])


def _no_docstring(x: int) -> int:
    return x


raises("缺 docstring 报错", DocstringParsingException,
       lambda: get_json_schema(_no_docstring))


def _no_arg_desc(x: int) -> int:
    """没写参数说明。

    Args:
        y: 写错了名字。
    """
    return x


raises("缺参数描述报错", DocstringParsingException,
       lambda: get_json_schema(_no_arg_desc))


def _no_type_hint(x) -> int:
    """缺类型提示。

    Args:
        x: 参数。
    """
    return x


raises("缺类型提示报错", TypeHintParsingException,
       lambda: get_json_schema(_no_type_hint))


def _no_return_hint(x: int):
    """缺返回提示。

    Args:
        x: 参数。
    """
    return x


raises("@tool 缺返回提示报错", TypeHintParsingException,
       lambda: tool(_no_return_hint))


@tool
def _no_args() -> None:
    """无参数的工具。"""
    return None


check("无参数工具 output_type 是 null", _no_args.output_type == "null")

# ---------- 2. Tool 校验 ----------


class _BadName(Tool):
    name = "bad-name"
    description = "坏名字"
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        return ""


raises("坏工具名实例化报错", TypeError, lambda: _BadName())


class _KeywordName(Tool):
    name = "class"
    description = "关键字当名字"
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        return ""


raises("关键字工具名实例化报错", TypeError, lambda: _KeywordName())


class _BadInputs(Tool):
    name = "bad_inputs"
    description = "inputs 缺 description"
    inputs = {"x": {"type": "string"}}
    output_type = "string"

    def forward(self, x: str) -> str:
        return x


raises("inputs 缺字段实例化报错", TypeError, lambda: _BadInputs())


class _SigMismatch(Tool):
    name = "sig_mismatch"
    description = "签名对不上 inputs"
    inputs = {"x": {"type": "string", "description": "x"}}
    output_type = "string"

    def forward(self, y: str) -> str:
        return y


raises("forward 签名与 inputs 不一致报错", TypeError, lambda: _SigMismatch())


class _BadOutput(Tool):
    name = "bad_output"
    description = "坏 output_type"
    inputs = {}
    output_type = "weird"

    def forward(self) -> str:
        return ""


raises("坏 output_type 实例化报错", TypeError, lambda: _BadOutput())

# ---------- 3. 参数校验 ----------


class _NumberTool(Tool):
    name = "number_tool"
    description = "测类型校验"
    inputs = {"n": {"type": "number", "description": "数字"},
              "kind": {"type": "string", "description": "种类",
                       "enum": ["a", "b"]},
              "opt": {"type": "string", "description": "可选", "nullable": True}}
    output_type = "string"

    def forward(self, n: float, kind: str, opt: str | None = None) -> str:
        return f"{n}-{kind}-{opt}"


_num = _NumberTool()
check("合法参数通过", validate_tool_arguments(_num, {"n": 1.5, "kind": "a"}) ==
      {"n": 1.5, "kind": "a"})
check("number 接受 int", validate_tool_arguments(_num, {"n": 2, "kind": "a"})["n"] == 2)
raises("未知参数报错", ValueError,
       lambda: validate_tool_arguments(_num, {"n": 1, "kind": "a", "zzz": 1}))
raises("缺必填参数报错", ValueError,
       lambda: validate_tool_arguments(_num, {"kind": "a"}))
raises("类型不对报错", ValueError,
       lambda: validate_tool_arguments(_num, {"n": "x", "kind": "a"}))
raises("boolean 不能当 number", ValueError,
       lambda: validate_tool_arguments(_num, {"n": True, "kind": "a"}))
raises("enum 之外报错", ValueError,
       lambda: validate_tool_arguments(_num, {"n": 1, "kind": "c"}))
check("单个值简写（单参数工具）",
      validate_tool_arguments(calc, "128*46") == {"expression": "128*46"})
raises("单个值简写（多参数工具）报错", ValueError,
       lambda: validate_tool_arguments(_num, 1))

# ---------- 4. 默认工具的行为 ----------

check("calc 128*46", calc("128*46") == "5888")
check("calc 除零报错", calc("1/0") == "错误：除数不能为 0")
check("calc 拒绝名字调用", "错误" in calc("__import__('os').system('id')"))
check("calc 拒绝属性访问", "错误" in calc("(1).bit_length()"))
check("calc dict 参数调用", calc({"expression": "2**10"}) == "1024")

with tempfile.TemporaryDirectory() as tmp:
    with open(os.path.join(tmp, "notes.txt"), "w", encoding="utf-8") as f:
        f.write("hello tools")
    reader = ReadFileTool(base_dir=tmp)
    check("read_file 读到内容", reader("notes.txt") == "hello tools")
    check("read_file 拒绝跳出 base_dir", "错误" in reader("../secret.txt"))
    check("read_file 找不到文件", "错误" in reader("nope.txt"))
    check("read_file 绝对路径在 base_dir 内也行",
          reader(os.path.join(tmp, "notes.txt")) == "hello tools")

with tempfile.TemporaryDirectory() as tmp:
    writer = WriteFileTool(base_dir=tmp)
    check("write_file 写完能读回",
          "已写入" in writer("out.txt", "hello write")
          and open(os.path.join(tmp, "out.txt"), encoding="utf-8").read() == "hello write")
    check("write_file 覆盖旧内容",
          "已写入" in writer("out.txt", "v2")
          and open(os.path.join(tmp, "out.txt"), encoding="utf-8").read() == "v2")
    check("write_file 自动建子目录",
          "已写入" in writer("sub/deep.txt", "x")
          and open(os.path.join(tmp, "sub", "deep.txt"), encoding="utf-8").read() == "x")
    check("write_file 拒绝跳出 base_dir", "错误" in writer("../secret.txt", "x"))

# web_search：把 urlopen mock 掉，不真联网。
_fake_resp = mock.MagicMock()
_fake_resp.read.return_value = json.dumps({
    "AbstractText": "Photosynthesis is how plants make food.",
    "RelatedTopics": [{"Text": "Chlorophyll absorbs light."},
                      {"Text": "Not a dict-free entry"}]}).encode()
_fake_resp.__enter__.return_value = _fake_resp
_fake_resp.__exit__.return_value = False
with mock.patch("urllib.request.urlopen", return_value=_fake_resp):
    out = web_search("photosynthesis")
check("web_search 返回简介+相关条目",
      "Photosynthesis is how plants make food." in out
      and "Chlorophyll absorbs light." in out)

_empty_resp = mock.MagicMock()
_empty_resp.read.return_value = json.dumps(
    {"AbstractText": "", "RelatedTopics": []}).encode()
_empty_resp.__enter__.return_value = _empty_resp
_empty_resp.__exit__.return_value = False
with mock.patch("urllib.request.urlopen", return_value=_empty_resp):
    out = web_search("zzzzqqq")
check("web_search 搜不到如实说", "没搜到" in out and "zzzzqqq" in out)

with mock.patch("urllib.request.urlopen",
                side_effect=urllib.error.URLError("no network")):
    out = web_search("x")
check("web_search 失败变 Observation", out.startswith("搜索失败"))

_fa = FinalAnswerTool()
check("final_answer 原样返回字符串", _fa("done") == "done")
check("final_answer 原样返回非字符串", _fa({"a": 1}) == {"a": 1})
check("final_answer output_type 是 any", _fa.output_type == "any")

# ---------- 5. 注册表 ----------

registry = default_tools(base_dir=tempfile.gettempdir())
check("默认工具箱 5 个工具",
      registry.names() == ["calc", "read_file", "write_file", "web_search",
                           "final_answer"])
check("注册表直接调", registry.call("calc", {"expression": "3*7"}) == "21")
check("注册表 JSON 字符串调",
      registry.call("calc", '{"expression": "3*7"}') == "21")
# 注意：字符串参数一律按 JSON 解析；单个值简写只在 validate_tool_arguments
# 和工具直接调用时支持（上面第 3 节已测），注册表层不猜。
raises("注册表坏 JSON 报错", ValueError,
       lambda: registry.call("calc", "{not json"))
raises("注册表重名报错", ValueError, lambda: registry.register(calc))
raises("注册表未知工具报错", KeyError, lambda: registry.call("nope", {}))
check("注册表 final_answer 透传",
      registry.call("final_answer", {"answer": "ok"}) == "ok")

openai_tools = registry.to_openai_tools()
check("导出 OpenAI tools 是 5 项",
      len(openai_tools) == 5
      and [t["function"]["name"] for t in openai_tools]
      == ["calc", "read_file", "write_file", "web_search", "final_answer"])
calc_fn = openai_tools[0]["function"]
check("calc 的 OpenAI schema 对",
      calc_fn["parameters"]["properties"]["expression"]["type"] == "string"
      and calc_fn["parameters"]["required"] == ["expression"])
check("final_answer 的 OpenAI schema 不发 any 类型",
      "type" not in openai_tools[4]["function"]["parameters"]["properties"]["answer"]
      and openai_tools[4]["function"]["parameters"]["properties"]["answer"]["description"]
      == "The final answer to the problem")
check("final_answer 内部 inputs 仍保留 any",
      registry["final_answer"].inputs["answer"]["type"] == "any")

code_prompt = registry.to_code_prompt()
check("代码定义含 calc",
      "def calc(expression: string) -> string:" in code_prompt)
check("代码定义含 read_file",
      "def read_file(path: string) -> string:" in code_prompt)
check("代码定义含 write_file",
      "def write_file(path: string, content: string) -> string:" in code_prompt)
check("代码定义含 final_answer",
      "def final_answer(answer: any) -> any:" in code_prompt)
check("代码定义含工具描述",
      "安全算术" in code_prompt and "Provides a final answer" in code_prompt)

# 工具定义一次，两种渲染都能用（Phase 3/4 的入口就是这两个）。
single = ToolRegistry([calc])
check("单个工具两种渲染一致",
      single.to_openai_tools()[0]["function"]["name"] == "calc"
      and "def calc(expression: string)" in single.to_code_prompt())

print(f"\n{PASSED}/{PASSED + FAILED} passed.")
sys.exit(0 if FAILED == 0 else 1)
