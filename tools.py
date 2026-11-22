#!/usr/bin/env python3
"""Phase 2：工具系统 —— 复现 smolagents 的 tools.py 核心。

对照的上游设计（smolagents 1.26.0，见 notes/architecture.md §六）：
  - `Tool` 基类：name / description / inputs（JSON Schema）/
    output_type + `forward()`；实例化时校验这些属性。
  - `@tool` 装饰器：把带类型注解和 docstring 的 Python 函数变成工具，
    name / 描述 / 输入 schema 直接从函数签名、docstring、类型提示推导，
    不用手写 JSON Schema。
  - 参数校验 `validate_tool_arguments`；工具定义渲染
    `to_code_prompt`（供 CodeAgent 提示词用，把工具写成 Python def）。

自己的取舍（和 Phase 1 同一路）：
  - 只用标准库，不引 smolagents、transformers——schema 推导自己写。
  - 多一个 mini-agent 风格的 `ToolRegistry`：工具注册一次，
    之后按名字调用、能导出 OpenAI tools 参数（Phase 3 给 ToolCallingAgent 用）、
    能导出整段代码定义（Phase 4 给 CodeAgent 提示词用）。
  - 工具返回的永远是字符串 Observation（或能转成字符串的值），
    和 mini-agent 的工具约定一致；Phase 3 的循环直接拿它当 Observation。

这个文件不做 agent 循环，只解决三件事：怎么定义工具、怎么校验参数、
怎么把工具交给模型。真正的工具（calc / read_file / web_search /
final_answer）在 default_tools.py。
"""
import ast
import inspect
import json
import keyword
import re
import textwrap
import types
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

# ---------- 类型提示 → JSON Schema ----------

# smolagents 的 AUTHORIZED_TYPES；工具参数只能是这些类型。
AUTHORIZED_TYPES = ["string", "boolean", "integer", "number",
                    "image", "audio", "array", "object", "any", "null"]

# Python 类型 → JSON Schema 类型（上游 _BASE_TYPE_MAPPING 的同款）。
_BASE_TYPE_MAPPING = {
    int: {"type": "integer"},
    float: {"type": "number"},
    str: {"type": "string"},
    bool: {"type": "boolean"},
    list: {"type": "array"},
    dict: {"type": "object"},
    Any: {"type": "any"},
    type(None): {"type": "null"},
}


class TypeHintParsingException(Exception):
    """类型提示解析失败（上游同名异常）。"""


class DocstringParsingException(Exception):
    """docstring 解析失败（上游同名异常）。"""


# docstring 切段用的正则（照着上游 _function_type_hints_utils.py 的意思重写）：
#   描述 = 开头到 Args:/Returns:/Raises: 之前；Args: 块按参数名切开。
_DESCRIPTION_RE = re.compile(r"^(.*?)(?=\n\s*(Args:|Returns:|Raises:)|\Z)", re.DOTALL)
_ARGS_RE = re.compile(r"\n\s*Args:\n\s*(.*?)[\n\s]*(Returns:|Raises:|\Z)", re.DOTALL)
_ARGS_SPLIT_RE = re.compile(
    r"(?:^|\n)\s*(\w+)\s*(?:\([^)]*?\))?:\s*(.*?)\s*"
    r"(?=\n\s*\w+\s*(?:\([^)]*?\))?:|\Z)", re.DOTALL | re.VERBOSE)
_RETURNS_RE = re.compile(r"\n\s*Returns:\n\s*(?:[^)]*?:\s*)?(.*?)[\n\s]*(Raises:|\Z)", re.DOTALL)


def _parse_google_docstring(docstring: str) -> tuple[str | None, dict, str | None]:
    """从 Google 风格 docstring 里拆出：描述、{参数名: 描述}、返回值描述。"""
    description_match = _DESCRIPTION_RE.search(docstring)
    args_match = _ARGS_RE.search(docstring)
    returns_match = _RETURNS_RE.search(docstring)
    description = description_match.group(1).strip() if description_match else None
    returns = returns_match.group(1).strip() if returns_match else None
    args_dict: dict = {}
    if args_match:
        block = "\n".join(line for line in args_match.group(1).split("\n") if line.strip())
        for name, desc in _ARGS_SPLIT_RE.findall(block):
            args_dict[name] = re.sub(r"\s*\n+\s*", " ", desc.strip())
    return description, args_dict, returns


def _parse_type_hint(hint) -> dict:
    """一个类型提示 → JSON Schema 片段（递归；照着上游 _parse_type_hint）。"""
    origin = get_origin(hint)
    args = get_args(hint)
    if origin is None:
        if hint in _BASE_TYPE_MAPPING:
            return dict(_BASE_TYPE_MAPPING[hint])
        raise TypeHintParsingException(f"解析不了这个类型提示（可能是自定义类）：{hint}")
    if origin is list:
        if not args:
            return {"type": "array"}
        return {"type": "array", "items": _parse_type_hint(args[0])}
    if origin is dict:
        out: dict = {"type": "object"}
        if len(args) == 2:
            out["additionalProperties"] = _parse_type_hint(args[1])
        return out
    if origin is tuple:
        if not args:
            return {"type": "array"}
        return {"type": "array", "prefixItems": [_parse_type_hint(t) for t in args]}
    # Union / X | None（含 Optional）：只剩一种非 None 类型就直接用它，
    # 否则列成 type 列表；带 None 就标 nullable。
    if origin is Union or origin is types.UnionType:
        non_none = [t for t in args if t is not type(None)]
        parsed = [_parse_type_hint(t) for t in non_none]
        if len(parsed) == 1:
            result = dict(parsed[0])
        elif all(isinstance(p.get("type"), str) for p in parsed):
            result = {"type": sorted(p["type"] for p in parsed)}
        else:
            result = {"anyOf": parsed}
        if type(None) in args:
            result["nullable"] = True
        return result
    # Literal[...]：枚举值进 enum，类型按字面量本身推。
    if origin is Literal:
        values = [a for a in args if a is not None]
        base = _parse_type_hint(type(values[0])) if values else {"type": "string"}
        base["enum"] = values
        if None in args:
            base["nullable"] = True
        return base
    raise TypeHintParsingException(f"解析不了这个类型提示（可能是自定义类）：{hint}")


def _type_hints_to_json_schema(func) -> dict:
    """签名的类型提示 → {"type": "object", "properties": {...}, "required": [...]}。

    有默认值的参数不进 required，同时标 nullable（和上游一致：调用方可以不传）。
    """
    type_hints = get_type_hints(func)
    signature = inspect.signature(func)
    properties: dict = {}
    for param_name, param_type in type_hints.items():
        properties[param_name] = _parse_type_hint(param_type)
    required = []
    for param_name, param in signature.parameters.items():
        if param.annotation is inspect.Parameter.empty:
            raise TypeHintParsingException(
                f"参数 {param.name} 缺类型提示（函数 {func.__name__}）")
        if param_name not in properties:
            properties[param_name] = {}
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
        else:
            properties[param_name]["nullable"] = True
    # 返回值是多类型联合时，上游统一按 any 处理。
    if "return" in properties and isinstance(properties["return"].get("type"), list):
        properties["return"]["type"] = "any"
    schema: dict = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def get_json_schema(func) -> dict:
    """从函数推导 JSON Schema（上游 get_json_schema 的同款行为）。

    要求：函数有 docstring；每个参数在 docstring 的 Args: 里有描述；
    每个参数和返回值都有类型提示。返回值描述（Returns:）可选。
    参数描述末尾可写 `(choices: ["a", "b"])`，会解析成 enum。
    返回 {"type": "function", "function": {name, description, parameters, return?}}。
    """
    doc = inspect.getdoc(func)
    if not doc:
        raise DocstringParsingException(
            f"函数 {func.__name__} 没有 docstring，没法推导 schema！")
    description, param_descriptions, return_doc = _parse_google_docstring(doc.strip())
    schema = _type_hints_to_json_schema(func)
    return_dict = schema["properties"].pop("return", None)
    if return_dict is not None and return_doc is not None:
        return_dict["description"] = return_doc
    for arg, arg_schema in schema["properties"].items():
        if arg not in param_descriptions:
            raise DocstringParsingException(
                f"函数 {func.__name__} 的 docstring 里没有参数 '{arg}' 的描述")
        desc = param_descriptions[arg]
        enum_match = re.search(r"\(choices:\s*(.*?)\)\s*$", desc, flags=re.IGNORECASE)
        if enum_match:
            arg_schema["enum"] = [c.strip() for c in json.loads(enum_match.group(1))]
            desc = enum_match.string[:enum_match.start()].strip()
        arg_schema["description"] = desc
    output = {"name": func.__name__, "description": description, "parameters": schema}
    if return_dict is not None:
        output["return"] = return_dict
    return {"type": "function", "function": output}


# ---------- 参数校验 ----------

# JSON Schema 类型 → Python 里的检查办法。number 接受 int 和 float，
# 但 boolean 不能当 integer 用（Python 里 bool 是 int 的子类，要先排除）。
def _value_matches_type(value, json_type) -> bool:
    if isinstance(json_type, list):
        return any(_value_matches_type(value, t) for t in json_type)
    if json_type == "any":
        return True
    if json_type == "null":
        return value is None
    if json_type == "string":
        return isinstance(value, str)
    if json_type == "boolean":
        return isinstance(value, bool)
    if json_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if json_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if json_type == "array":
        return isinstance(value, list)
    if json_type == "object":
        return isinstance(value, dict)
    return True  # image/audio 等本阶段不校验内容


def validate_tool_arguments(tool: "Tool", arguments) -> dict:
    """校验一次工具调用的参数，返回归一化后的 kwargs dict。

    接受 dict 或单个值（只有一个输入参数的工具才能用单个值）。
    校验：未知参数、缺必填参数、类型不对、enum 之外的值——
    任何一项不过都抛 ValueError，调用方（Phase 3 的循环）会把它变成
    Observation 喂回模型，而不是悄悄放行。
    """
    if not isinstance(arguments, dict):
        if len(tool.inputs) == 1:
            arguments = {next(iter(tool.inputs)): arguments}
        else:
            raise ValueError(
                f"工具 '{tool.name}' 有 {len(tool.inputs)} 个参数，"
                f"不能用单个值调用，请传 dict")
    unknown = [k for k in arguments if k not in tool.inputs]
    if unknown:
        raise ValueError(f"工具 '{tool.name}' 没有这些参数：{unknown}")
    required = [k for k, v in tool.inputs.items() if not v.get("nullable")]
    missing = [k for k in required if k not in arguments]
    if missing:
        raise ValueError(f"工具 '{tool.name}' 缺必填参数：{missing}")
    for name, value in arguments.items():
        spec = tool.inputs[name]
        if value is None and spec.get("nullable"):
            continue
        json_type = spec.get("type")
        if json_type is not None and not _value_matches_type(value, json_type):
            raise ValueError(
                f"工具 '{tool.name}' 的参数 '{name}' 应该是 {json_type} 类型，"
                f"实际是 {type(value).__name__}")
        if "enum" in spec and value not in spec["enum"]:
            raise ValueError(
                f"工具 '{tool.name}' 的参数 '{name}' 只能是 {spec['enum']} 之一，"
                f"实际是 {value!r}")
    return dict(arguments)


# ---------- Tool 基类 ----------


class Tool:
    """工具基类（上游 Tool 的精简版）。

    子类要定四个类属性 + 一个 forward 方法：
      - name：工具名，必须是合法 Python 标识符、不能是关键字
      - description：工具是干什么的（给模型看）
      - inputs：{参数名: {"type": JSON 类型, "description": 参数说明}}
      - output_type：返回值的 JSON 类型
      - forward(self, ...) -> ...：真正干活的方法，参数名必须和 inputs 的键一致

    实例化时自动校验这四项（上游的 validate_after_init 同款）；
    第一次调用时才调 setup()（上游的懒初始化，贵操作放这里）。
    """

    name: str
    description: str
    inputs: dict
    output_type: str

    def __init__(self, *args, **kwargs):
        self.is_initialized = False
        self._validate_attributes()

    def _validate_attributes(self):
        # 名字：合法标识符、不是关键字。
        if not isinstance(getattr(self, "name", None), str) or not self.name.isidentifier() \
                or keyword.iskeyword(self.name):
            raise TypeError(
                f"工具名 '{getattr(self, 'name', None)}' 不合法："
                f"必须是合法 Python 标识符、不能是关键字")
        if not isinstance(getattr(self, "description", None), str) \
                or not self.description.strip():
            raise TypeError(f"工具 '{self.name}' 必须有非空的 description")
        if not isinstance(getattr(self, "inputs", None), dict):
            raise TypeError(f"工具 '{self.name}' 的 inputs 必须是 dict")
        for input_name, spec in self.inputs.items():
            if not isinstance(spec, dict) or "type" not in spec or "description" not in spec:
                raise TypeError(
                    f"工具 '{self.name}' 的参数 '{input_name}' 必须有 type 和 description")
            types = spec["type"] if isinstance(spec["type"], list) else [spec["type"]]
            bad = [t for t in types if t not in AUTHORIZED_TYPES]
            if bad:
                raise TypeError(
                    f"工具 '{self.name}' 的参数 '{input_name}' 类型 {bad} 不在允许列表里："
                    f"{AUTHORIZED_TYPES}")
        if getattr(self, "output_type", None) not in AUTHORIZED_TYPES:
            raise TypeError(
                f"工具 '{self.name}' 的 output_type '{self.output_type}' 不合法")
        # forward 签名必须和 inputs 的键一一对应（self 除外）。
        params = [p for p in inspect.signature(self.forward).parameters if p != "self"]
        if set(params) != set(self.inputs):
            raise TypeError(
                f"工具 '{self.name}' 的 forward 参数是 {sorted(params)}，"
                f"但 inputs 声明的是 {sorted(self.inputs)}，两者必须一致")

    def setup(self):
        """贵操作（加载模型、建连接）放这里；第一次调用时才执行一次。"""
        self.is_initialized = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError("子类必须实现 forward()")

    def _bind(self, args, kwargs) -> dict:
        """位置参数按 forward 签名顺序转成 kwargs dict（缺参/多参会报 TypeError）。"""
        sig = inspect.signature(self.forward)
        try:
            bound = sig.bind(*args, **kwargs)
        except TypeError as e:
            raise TypeError(f"调用工具 '{self.name}' 的参数不对：{e}") from e
        bound.apply_defaults()
        return {k: v for k, v in bound.arguments.items() if k != "self"}

    def __call__(self, *args, **kwargs):
        if not self.is_initialized:
            self.setup()
        # 单个 dict 且键全是参数名时，把它拆成 kwargs（上游同款）。
        if len(args) == 1 and not kwargs and isinstance(args[0], dict) \
                and all(k in self.inputs for k in args[0]):
            args, kwargs = (), args[0]
        bound = self._bind(args, kwargs)
        validate_tool_arguments(self, bound)
        return self.forward(**bound)

    # ---------- 渲染：给模型看的两种样子 ----------

    def to_code_prompt(self) -> str:
        """把工具渲染成一段 Python 函数定义（供 CodeAgent 提示词用）。

        上游 get_tools_definition_code / to_code_prompt 的同款：模型看到的是
        一个带签名和 docstring 的 def，写代码时直接调用它。
        """
        args_sig = ", ".join(
            f"{name}: {spec['type']}" for name, spec in self.inputs.items())
        doc = self.description
        if self.inputs:
            args_doc = "\n".join(
                f"{name}: {spec['description']}" for name, spec in self.inputs.items())
            doc += f"\n\nArgs:\n{textwrap.indent(args_doc, '    ')}"
        doc = f'"""{doc}\n"""'
        return (f"def {self.name}({args_sig}) -> {self.output_type}:\n"
                f"{textwrap.indent(doc, '    ')}")

    def to_tool_calling_prompt(self) -> str:
        """一句话介绍（上游 to_tool_calling_prompt 的同款）。"""
        return (f"{self.name}: {self.description}\n"
                f"    Takes inputs: {self.inputs}\n"
                f"    Returns an output of type: {self.output_type}")

    def to_openai_function(self) -> dict:
        """转成 OpenAI tools 参数里的一项（Phase 3 给模型传 tools_to_call_from 用）。

        注意：上游的 "any" 类型不能原样发给模型——Gemini 的 OpenAI 兼容端点
        会 400（实测：`unrecognized type 'any' at properties.answer`）。
        "any" 的意思是“类型不限”， wire 上就直接不写 type（只有描述），
        这才是它在 JSON Schema 里的正确翻译；工具内部的 inputs 仍保留 "any"。
        """
        properties = {}
        required = []
        for name, spec in self.inputs.items():
            prop = {"description": spec["description"]}
            if spec["type"] != "any":
                prop["type"] = spec["type"]
            if "enum" in spec:
                prop["enum"] = spec["enum"]
            properties[name] = prop
            if not spec.get("nullable"):
                required.append(name)
        parameters: dict = {"type": "object", "properties": properties}
        if required:
            parameters["required"] = required
        return {"type": "function",
                "function": {"name": self.name,
                             "description": self.description,
                             "parameters": parameters}}


# ---------- @tool 装饰器 ----------


def tool(tool_function):
    """把一个带类型注解和 docstring 的函数变成工具（上游 @tool 的同款）。

    要求：Google 风格 docstring（描述 + Args: 逐参数说明）；
    每个参数和返回值都有类型提示。name / 描述 / inputs / output_type
    全部从签名和 docstring 推导，不用手写。
    """
    schema = get_json_schema(tool_function)["function"]
    if "return" not in schema:
        if len(schema["parameters"]["properties"]) == 0:
            schema["return"] = {"type": "null"}
        else:
            raise TypeHintParsingException(
                "工具没有返回值类型提示：请给函数加上返回类型注解！")

    class SimpleTool(Tool):
        def __init__(self):
            self.is_initialized = True
            # 不走 Tool.__init__（它会把 is_initialized 重置为 False），
            # 但校验不能省：name/inputs/output_type/forward 签名都要过一遍。
            self._validate_attributes()

    SimpleTool.name = schema["name"]
    SimpleTool.description = schema["description"]
    SimpleTool.inputs = schema["parameters"]["properties"]
    SimpleTool.output_type = schema["return"]["type"]

    def wrapped_function(*args, **kwargs):
        return tool_function(*args, **kwargs)

    SimpleTool.forward = staticmethod(wrapped_function)
    # 校验也照着 forward 签名来：把原函数的签名直接挂上去。
    SimpleTool.forward.__signature__ = inspect.signature(tool_function)
    return SimpleTool()


# ---------- 工具注册表 ----------


class ToolRegistry:
    """工具注册表：工具定义一次，到处都能用。

    mini-agent 的 TOOLS dict 在这里长成了对象：注册、按名字调用、
    导出 OpenAI tools 参数（ToolCallingAgent）、导出代码定义
    （CodeAgent 提示词）。Phase 3 的 MultiStepAgent 直接拿它当工具箱。
    """

    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> Tool:
        """注册一个工具；重名就报错（两个同名工具会让模型分不清）。"""
        if tool.name in self._tools:
            raise ValueError(f"工具名 '{tool.name}' 已经被注册了")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __getitem__(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"没有这个工具：{name}，可用的工具是 {self.names()}")
        return self._tools[name]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        return list(self._tools)

    def call(self, name: str, arguments) -> str:
        """按名字调一个工具，返回 Observation 字符串。

        arguments 可以是 dict，也可以是 JSON 字符串（模型 tool_calls 里
        的 arguments 就是 JSON 字符串，Phase 3 直接传进来）。
        返回值不是字符串时转成 JSON 字符串，保证循环拿到的永远是字符串。
        """
        tool = self[name]  # 未知工具抛 KeyError，调用方会把它变成 Observation
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"工具 '{name}' 的参数不是合法 JSON：{e}") from e
        bound = validate_tool_arguments(tool, arguments)
        result = tool(**bound)
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False)

    def to_openai_tools(self) -> list[dict]:
        """全部工具的 OpenAI tools 参数（给 Model.generate 的 tools_to_call_from）。"""
        return [t.to_openai_function() for t in self._tools.values()]

    def to_code_prompt(self) -> str:
        """全部工具的代码定义（给 CodeAgent 系统提示词拼工具定义用）。"""
        return "\n\n".join(t.to_code_prompt() for t in self._tools.values())
