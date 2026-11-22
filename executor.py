#!/usr/bin/env python3
"""Phase 4：本地 Python 执行器 —— 复现 smolagents 的 local_python_executor.py 核心。

对照的上游设计（smolagents 1.26.0，见 notes/architecture.md §五）：
  - 不是裸 exec()：上游是逐节点走 AST 的解释器（evaluate_ast），
    导入白名单、操作数上限、有状态命名空间、print 收集、final_answer
    靠异常传值。这里用“AST 事前检查 + 受限 exec”的同款策略：
    同样的安全策略（白名单导入、禁 open()/eval()/dunder、操作数上限、
    有状态命名空间），实现方式更短——AST 检查负责“什么不许出现”，
    exec 负责“跑起来”，两者缺一不可。
  - 导入白名单照着上游的 BASE_BUILTIN_MODULES（11 个模块）；
    MAX_OPERATIONS（上游 10_000_000）在这里是“执行的代码行数”上限，
    用 sys.settrace 数（只数模型代码自己的帧，不数工具内部的代码）；
    30 秒墙钟超时照着上游的 MAX_EXECUTION_TIME_SECONDS。
  - final_answer 的终止方式照着上游：代码里调 final_answer(x) 时，
    包装函数抛一个带值的异常，执行器接住它 → is_final_answer=True。
  - fix_final_answer_code 照着上游：模型如果把 final_answer 当变量
    赋值了（又同时调了它），把赋值重命名成 final_answer_variable，
    别让它把终止工具盖掉。

自己的取舍（和 Phase 1-3 同一路）：
  - 只用标准库；不上上游的逐节点解释器（1,768 行，复现成本太高，
    且逐节点求值会把 f-string、推导式、异常语义都重写一遍，容易
    走样）。AST 检查 + 受限 exec 是等价的策略实现，不是偷工减料：
    测试里锁定的就是策略本身（禁什么、拦什么、超限怎么办）。
  - 明确写在文档里：本地执行器不是安全沙箱（上游文档原话），
    不可信代码要走 Phase 6 的 Docker 执行器；这里的检查只防
    “模型手滑 / 试探绕过”，不防有意的攻击。
  - settrace 的操作数上限只数模型代码的帧（co_filename == "<code>"）：
    工具内部（urllib 等标准库）的行数不算，否则一次 web_search 就
    能把上限烧光。墙钟超时只在主线程可用（signal 只能在主线程设），
    非主线程调用时只靠操作数上限——注释里写清楚。
  - setattr 不放进可用内置函数（上游有）：setattr(x, "__class__", ...)
    这种字符串拼出来的 dunder 访问绕不过 AST 检查，干脆不给。
    getattr / hasattr 是上游的 nodunder_getattr（dunder 名字直接拒）。
"""
import ast
import math
import re
import signal
import sys
import threading
from dataclasses import dataclass

# ---------- 上游的常量（照抄） ----------

# 上游 utils.py 的 BASE_BUILTIN_MODULES：默认只允许导入这 11 个模块。
BASE_BUILTIN_MODULES = [
    "collections",
    "datetime",
    "itertools",
    "math",
    "queue",
    "random",
    "re",
    "stat",
    "statistics",
    "time",
    "unicodedata",
]

# 上游 local_python_executor.py 的 DANGEROUS_MODULES：明确禁掉的模块。
# 它们本来就不在白名单里，这里单列是为了报错时说得更明白。
DANGEROUS_MODULES = [
    "builtins",
    "io",
    "multiprocessing",
    "os",
    "pathlib",
    "pty",
    "shutil",
    "socket",
    "subprocess",
    "sys",
]

# 上游的两个上限（操作数上限在这里按“执行的代码行数”数，见模块 docstring）。
MAX_OPERATIONS = 100000
MAX_EXECUTION_TIME_SECONDS = 30
DEFAULT_MAX_LEN_OUTPUT = 50000

# AST 检查里直接禁掉的名字：open/eval/exec 这些一出现就拦，
# 不用等到运行时（Phase 0 的教训：模型被禁后会连试 8 步绕过，
# 早拦早反馈，纠正 Observation 里直接告诉它错在哪）。
FORBIDDEN_NAMES = {
    "open", "eval", "exec", "compile", "__import__",
    "globals", "locals", "vars", "input", "breakpoint",
    "help", "dir", "memoryview", "exit", "quit",
    "license", "copyright", "credits",
}


class InterpreterError(ValueError):
    """执行器错误（上游同名）：语法错、禁用的导入/名字、超限、执行中抛错。

    循环（agents.py）会把它变成纠正 Observation 喂回模型，
    而不是让整轮崩掉。
    """


class _FinalAnswerException(Exception):
    """final_answer(x) 被调用时抛的内部异常，带出答案值（上游同款做法）。

    执行器接住它 → CodeOutput(is_final_answer=True)，不往外漏。
    """

    def __init__(self, value):
        super().__init__("final_answer called")
        self.value = value


@dataclass
class CodeOutput:
    """一次代码执行的产出（上游同名同字段）。"""

    output: object
    logs: str
    is_final_answer: bool


class PrintContainer:
    """收集 print 输出的容器（上游同名）。

    每段代码执行前清空（state["_print_outputs"]），日志只记本段代码的，
    不跨步累积；变量才跨步保留。
    """

    def __init__(self):
        self.value = ""

    def append(self, text):
        self.value += text
        return self

    def __iadd__(self, other):
        self.value += str(other)
        return self

    def __str__(self):
        return self.value

    def __repr__(self):
        return f"PrintContainer({self.value})"


def _nodunder_getattr(obj, name, default=None):
    """上游的 nodunder_getattr：dunder 属性一律拒绝（字符串拼出来的也拦）。"""
    if name.startswith("__") and name.endswith("__"):
        raise InterpreterError(f"Forbidden access to dunder attribute: {name}")
    return getattr(obj, name, default)


def _nodunder_hasattr(obj, name):
    """hasattr 的 dunder 版：dunder 名字直接返回 False，不抛错。"""
    if name.startswith("__") and name.endswith("__"):
        return False
    return hasattr(obj, name)


# 上游 BASE_PYTHON_TOOLS 的同款（去掉 setattr，见模块 docstring）：
# 这些是模型代码里唯一能用的内置函数——没有 open，没有 eval。
def _base_python_tools():
    tools = {
        # 注意：print 不在这里——执行器每次 reset 时把写 PrintContainer 的
        # print 直接放进命名空间（state["print"]），名字解析先查 state。
        "isinstance": isinstance,
        "range": range,
        "float": float,
        "int": int,
        "bool": bool,
        "str": str,
        "set": set,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "round": round,
        "ceil": math.ceil,
        "floor": math.floor,
        "log": math.log,
        "exp": math.exp,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "atan2": math.atan2,
        "degrees": math.degrees,
        "radians": math.radians,
        "pow": pow,
        "sqrt": math.sqrt,
        "len": len,
        "sum": sum,
        "max": max,
        "min": min,
        "abs": abs,
        "enumerate": enumerate,
        "zip": zip,
        "reversed": reversed,
        "sorted": sorted,
        "all": all,
        "any": any,
        "map": map,
        "filter": filter,
        "ord": ord,
        "chr": chr,
        "next": next,
        "iter": iter,
        "divmod": divmod,
        "callable": callable,
        "getattr": _nodunder_getattr,
        "hasattr": _nodunder_hasattr,
        "issubclass": issubclass,
        "type": type,
        "complex": complex,
    }
    return tools


def fix_final_answer_code(code: str) -> str:
    """修掉模型把 final_answer 当变量赋值的写法（上游同款函数）。

    只有“既调了 final_answer()、又给 final_answer 赋值”时才改；
    只赋值没调用时不动——让后面的赋值报错发生（执行器会拦住它，
    循环再以纠正 Observation 告诉模型），免得悄悄改掉模型的历史。
    """
    assignment_pattern = r"(?<!\.)(?<!\w)\bfinal_answer\s*="
    if "final_answer(" not in code or not re.search(assignment_pattern, code):
        return code
    assignment_regex = r"(?<!\.)(?<!\w)(\bfinal_answer)(\s*=)"
    code = re.sub(assignment_regex, r"final_answer_variable\2", code)
    variable_regex = r"(?<!\.)(?<!\w)(\bfinal_answer\b)(?!\s*\()"
    code = re.sub(variable_regex, "final_answer_variable", code)
    return code


def _operation_tracer(max_operations: int):
    """数操作数的 tracer：只数模型代码自己的帧（co_filename == "<code>"）。

    每执行一行模型代码记一次；超过上限就抛 InterpreterError。
    工具内部（urllib 等）的行数不算——否则一次 web_search 就能烧光上限。
    """
    count = {"n": 0}

    def tracer(frame, event, arg):
        if event == "line" and frame.f_code.co_filename == "<code>":
            count["n"] += 1
            if count["n"] > max_operations:
                raise InterpreterError(
                    f"Reached the max number of operations of {max_operations}. "
                    "Maybe there is an infinite loop somewhere in the code, "
                    "or you're just asking too many calculations.")
        return tracer

    return tracer


class LocalPythonExecutor:
    """本地 Python 执行器（上游 LocalPythonExecutor 的同款接口）。

    接口和上游一致：send_tools / send_variables / __call__(code) -> CodeOutput。
    state 是有状态命名空间：变量跨步保留，工具和 print 每次 reset 后重新注入。

    安全策略（测试里全部锁定）：
      - 导入白名单：只有 BASE_BUILTIN_MODULES + 额外授权的模块能 import；
      - 禁用名字：open/eval/exec/compile/__import__/globals/locals 等
        在 AST 检查阶段就拦，错误信息直接告诉模型；
      - dunder：任何 __x__ 形式的属性访问和名字都拦（__builtins__、
        ().__class__.__base__、字符串拼出来的 getattr 也拦）；
      - 工具名是保留字：代码不能给工具名赋值、不能删工具名；
      - 操作数上限（行数）+ 墙钟超时（主线程才有 signal，见模块 docstring）。
    """

    def __init__(self, additional_authorized_imports: list[str] | None = None,
                 max_operations: int = MAX_OPERATIONS,
                 max_print_outputs_length: int = DEFAULT_MAX_LEN_OUTPUT,
                 timeout_seconds: int | None = MAX_EXECUTION_TIME_SECONDS):
        self.additional_authorized_imports = list(additional_authorized_imports or [])
        self.authorized_imports = sorted(
            set(BASE_BUILTIN_MODULES) | set(self.additional_authorized_imports))
        self.max_operations = max_operations
        self.max_print_outputs_length = max_print_outputs_length
        self.timeout_seconds = timeout_seconds
        self._tools: dict = {}
        self.state: dict = {}
        self.reset()

    # ---------- 工具 / 变量注入 ----------

    def send_tools(self, tools) -> None:
        """把工具注进命名空间（上游同名）。

        tools 可以是 ToolRegistry、{名字: Tool} 或 [(名字, Tool)]。
        每个工具变成命名空间里的一个可调用函数；final_answer 特殊处理——
        调它就抛 _FinalAnswerException，执行器以此识别“这一轮结束了”。
        """
        if hasattr(tools, "items"):
            pairs = list(tools.items())
        elif hasattr(tools, "names"):  # ToolRegistry：按名字取
            pairs = [(name, tools[name]) for name in tools.names()]
        else:
            pairs = list(tools)
        for name, tool in pairs:
            self._tools[name] = tool
        self._inject_tools()

    def send_variables(self, variables: dict) -> None:
        """把变量注进命名空间（上游同名）。

        工具名、print、__builtins__ 是保留字，不能被变量盖掉。
        """
        reserved = set(self._tools) | {"print", "__builtins__"}
        clash = [k for k in variables if k in reserved]
        if clash:
            raise ValueError(f"变量名和保留字冲突（工具名/print 不能被覆盖）：{clash}")
        self.state.update(variables)

    def reset(self) -> None:
        """清空命名空间里的变量，重新注入内置函数和工具（每轮 run() 开始时调）。

        工具本身不丢：reset 后 _tools 还在，只是重新注进新的 state。
        """
        self.state = {}
        self.state["__builtins__"] = _base_python_tools()
        # import 语句要查 __builtins__["__import__"]：给一个只认白名单的
        # 安全版（AST 检查已经拦过一遍，这里是运行时第二道锁）。
        self.state["__builtins__"]["__import__"] = self._safe_import
        self.state["_print_outputs"] = PrintContainer()
        self._inject_tools()

    def _safe_import(self, name, globals=None, locals=None, fromlist=(), level=0):
        """受限的 __import__：只放行白名单模块，语义和内置 __import__ 一致。

        fromlist 为空时返回顶层模块（`import a.b` 绑定的是 a），
        否则返回所指模块（`from a.b import c` 再从它上面取 c）。
        """
        import importlib
        if level != 0:
            raise InterpreterError("Relative imports are not allowed")
        root = name.split(".")[0]
        if root not in self.authorized_imports:
            raise InterpreterError(f"Import of {name} is not allowed")
        module = importlib.import_module(name)
        if not fromlist:
            return importlib.import_module(root)
        return module

    def _inject_tools(self) -> None:
        """把 self._tools 写进 state（reset / send_tools 时调）。"""
        for name, tool in self._tools.items():
            if name == "final_answer":
                self.state[name] = self._make_final_answer(tool)
            else:
                self.state[name] = self._make_tool_callable(tool)
        self.state["print"] = self._make_print()

    @staticmethod
    def _make_tool_callable(tool):
        """工具包一层：位置参数/关键字参数都按 Tool.__call__ 的绑定规则走。"""
        def _call(*args, **kwargs):
            return tool(*args, **kwargs)
        _call.__name__ = tool.name
        return _call

    @staticmethod
    def _make_final_answer(tool):
        """final_answer 包装：先调工具拿答案，再抛异常让执行器识别终止。"""
        def _call(*args, **kwargs):
            value = tool(*args, **kwargs)
            raise _FinalAnswerException(value)
        _call.__name__ = "final_answer"
        return _call

    def _make_print(self):
        """print 写进本段代码的 PrintContainer（上游同款：日志只记本段）。"""
        def _print(*args, **kwargs):
            sep = kwargs.get("sep", " ")
            end = kwargs.get("end", "\n")
            self.state["_print_outputs"].append(sep.join(str(a) for a in args) + end)
            return None
        return _print

    # ---------- AST 事前检查 ----------

    def _check_tree(self, tree: ast.AST) -> None:
        """检查整棵 AST：导入、禁用名字、dunder、工具名赋值/删除。

        任何一项不过都抛 InterpreterError——错误信息直接进纠正
        Observation，告诉模型错在哪（Phase 0 的教训：越早说越省步数）。
        """
        tool_names = set(self._tools)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root not in self.authorized_imports:
                        raise InterpreterError(f"Import of {alias.name} is not allowed")
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    raise InterpreterError("Relative imports are not allowed")
                if node.module:
                    root = node.module.split(".")[0]
                    if root not in self.authorized_imports:
                        raise InterpreterError(f"Import of {node.module} is not allowed")
            elif isinstance(node, ast.Name):
                if node.id in FORBIDDEN_NAMES:
                    raise InterpreterError(
                        f"Forbidden function evaluation: '{node.id}' is not among "
                        "the explicitly allowed tools")
                if node.id.startswith("__"):
                    raise InterpreterError(
                        f"Forbidden access to dunder name: {node.id}")
                if isinstance(node.ctx, ast.Store) and node.id in tool_names:
                    raise InterpreterError(
                        f"Cannot assign to tool '{node.id}': tools cannot be overwritten")
                if isinstance(node.ctx, ast.Del) and (
                        node.id in tool_names or node.id == "print"):
                    raise InterpreterError(
                        f"Cannot delete tool '{node.id}': tools cannot be removed")
            elif isinstance(node, ast.Attribute):
                if node.attr.startswith("__") and node.attr.endswith("__"):
                    raise InterpreterError(
                        f"Forbidden access to dunder attribute: {node.attr}")

    # ---------- 执行 ----------

    def __call__(self, code_action: str) -> CodeOutput:
        """执行一段代码（上游同名）。

        返回 CodeOutput(output=最后一条表达式的值，logs=本段 print 输出，
        is_final_answer=是否调了 final_answer)。
        """
        try:
            tree = ast.parse(code_action)
        except SyntaxError as e:
            raise InterpreterError(
                f"Code parsing failed on line {e.lineno} due to: "
                f"{type(e).__name__}: {str(e)}\n{e.text}"
                f"{' ' * (e.offset or 0)}^")
        self._check_tree(tree)

        # 最后一条如果是表达式语句，单独 eval 取值（上游的“最后输出”同款）；
        # 其余语句逐条 exec，报错时能定位到具体哪一行。
        main_body = list(tree.body)
        last_expr = None
        if main_body and isinstance(main_body[-1], ast.Expr):
            last_expr = main_body.pop()

        self.state["_print_outputs"] = PrintContainer()
        tracer = _operation_tracer(self.max_operations)
        old_tracer = sys.gettrace()
        sys.settrace(tracer)
        try:
            output, is_final = self._with_timeout(
                lambda: self._execute(main_body, last_expr, code_action))
        except _FinalAnswerException as e:
            output, is_final = e.value, True
        finally:
            sys.settrace(old_tracer)
        logs = str(self.state["_print_outputs"])
        if len(logs) > self.max_print_outputs_length:
            logs = logs[:self.max_print_outputs_length]
        return CodeOutput(output=output, logs=logs, is_final_answer=is_final)

    def _execute(self, main_body: list, last_expr, code_action: str):
        """逐条执行主语句 + 最后一条表达式求值（在 tracer 和超时保护内跑）。"""
        for node in main_body:
            try:
                code = compile(ast.Module(body=[node], type_ignores=[]),
                               "<code>", "exec")
                exec(code, self.state)  # noqa: S102 - 受限命名空间 + AST 事前检查
            except _FinalAnswerException:
                raise
            except InterpreterError:
                raise
            except Exception as e:  # noqa: BLE001 - 执行错误要包成 InterpreterError
                segment = ast.get_source_segment(code_action, node) or "?"
                raise InterpreterError(
                    f"Code execution failed at line '{segment}' due to: "
                    f"{type(e).__name__}: {e}")
        output = None
        if last_expr is not None:
            try:
                expr_code = compile(ast.Expression(last_expr.value), "<code>", "eval")
                output = eval(expr_code, self.state)  # noqa: S307 - 同上
            except _FinalAnswerException:
                raise
            except InterpreterError:
                raise
            except Exception as e:  # noqa: BLE001 - 同上
                segment = ast.get_source_segment(code_action, last_expr) or "?"
                raise InterpreterError(
                    f"Code execution failed at line '{segment}' due to: "
                    f"{type(e).__name__}: {e}")
        return output, False

    def _with_timeout(self, fn):
        """墙钟超时（上游 MAX_EXECUTION_TIME_SECONDS 的同款）。

        signal 只能在主线程设；非主线程调用时跳过墙钟超时，
        只靠操作数上限兜底（见模块 docstring）。
        """
        if self.timeout_seconds is None:
            return fn()
        if threading.current_thread() is not threading.main_thread():
            return fn()

        def handler(signum, frame):
            raise InterpreterError(
                f"Code execution timed out after {self.timeout_seconds} seconds")

        old_handler = signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, self.timeout_seconds)
        try:
            return fn()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
