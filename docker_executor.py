#!/usr/bin/env python3
"""Phase 6：Docker 执行器 —— 把模型代码放进容器里跑，和本地执行器同接口。

对照的上游设计（smolagents 1.26.0，见 notes/architecture.md §五）：
  - 上游的远程执行器（Docker / E2B / Modal，remote_executors.py）和本地
    执行器是同一个 PythonExecutor 接口：send_tools / send_variables /
    __call__(code) -> CodeOutput。换执行器就是换一个对象（mini-swe-agent
    的 one-call-swap 教训：subprocess.run 换成 docker exec 只改一处）。
    这里照着这个接口写 DockerExecutor，CodeAgent 的 executor 参数
    本来就留了口子（Phase 4），一传就换，不用改循环。

安全模型（和本地执行器正好反过来，要想清楚）：
  - 本地执行器是“进程内否决”：AST 检查禁 open()/import os/dunder，
    因为代码和主进程跑在同一个 Python 里，一旦绕过就直达宿主机。
    它不是安全边界（上游文档原话），只防模型手滑和试探绕过。
  - Docker 执行器是“容器内放行、容器外隔离”：容器里跑的是完整 Python，
    不做 AST 限制（模型想 import os、open() 都行——隔离靠容器，不靠
    检查），安全来自容器的限制：--network none（默认断网）、--read-only
    只读根文件系统、--memory/--cpus/--pids-limit 资源上限、--cap-drop ALL、
    --user nobody、宿主机只挂一个工作目录进来。
  - 网络白名单（allowed_hosts）是第二层：容器放行网络时，runner 里
    的 socket 守卫只让白名单域名解析/连接。这是 Python 层的守卫，
    不是内核层的——容器里跑二进制程序能绕过它，所以默认还是断网。

本机实测结论（2026-11-21，见 notes/phase6-sandbox.md）：这台 VM
跑不了真正的容器——/dev/fuse 不存在（fuse 模块也没有，modprobe 失败），
seccomp 过滤器拦住了 setns，crun 和 runc 都起不来。所以本阶段的
“隔离”只能做到：执行器构造出正确的隔离命令（测试锁定了每一面旗），
并用 FakeRuntime 在宿主机上跑通 runner 的全部逻辑；真实隔离没有
验证，留给有 Docker 的机器跑 DOCKER_EXECUTOR_REAL_RUNTIME=podman
再验。测试里这件事写得明明白白，不假装 mock 能证明隔离。

工具怎么进容器：
  - @tool 函数：从闭包里取出原函数，取源码 + 它用到的模块 import，
    在容器里重新定义。只支持标准库、源码能拿到的工具。
  - Tool 子类：取 forward 方法的源码，把 self.xxx 换成状态字典
    _tool_state_<工具名>["xxx"]（实例属性只留能 JSON 序列化的；base_dir
    指到 runner 的 _WORKDIR 变量——容器里是 /work，FakeRuntime 在
    宿主机上跑时是宿主工作目录）；final_answer 特殊处理（哨兵 + 退出）。
  - 拿不到源码、状态不能 JSON 序列化、或者子智能体工具（forward
    里调宿主模型）——send_tools 时直接报 ValueError 说清楚，不悄悄
    丢工具。managed_agents 在容器里不支持（Phase 6 的明确边界）。

变量怎么进容器：send_variables 只收能 JSON 序列化的值，写进
_input.json；跑完 runner 把命名空间里新增/改动过的 JSON 值
（__STATE__）带回来，写回 self.state，变量跨步保留。
"""
import ast
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import uuid

from executor import CodeOutput, InterpreterError, PrintContainer

# ---------- 默认配置（和上游的沙箱默认值同一量级） ----------

DEFAULT_IMAGE = "python:3.12-slim"
DEFAULT_MEMORY = "256m"          # --memory 上限（swap 也锁成同样大小）
DEFAULT_CPUS = "0.5"             # --cpus 上限
DEFAULT_PIDS_LIMIT = 64          # --pids-limit（防 fork 炸弹）
DEFAULT_TIMEOUT_SECONDS = 30     # 和本地执行器一样的墙钟超时
CONTAINER_WORKDIR = "/work"      # 容器里的工作目录（宿主机只挂这一个目录）
CONTAINER_USER = "65534:65534"   # nobody:nogroup，最小权限跑代码

# runner 和宿主机之间用行首哨兵传数据（stdout 是唯一的回传通道，
# print 的日志也要走这里，所以哨兵前缀要足够特别，模型正常输出
# 撞上的概率可以忽略）。
_MARKER_RESULT = "__RESULT__"
_MARKER_STATE = "__STATE__"
_MARKER_LOGS = "__LOGS__"
_MARKER_FINAL = "__FINAL__"
_MARKER_ERROR = "__ERROR__"


# ---------- 容器运行时（可注入，测试用 FakeRuntime） ----------


class ContainerResult:
    """一次容器运行的结果（subprocess.CompletedProcess 的精简版）。"""

    def __init__(self, returncode: int, stdout: str, stderr: str):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class SubprocessRuntime:
    """真正的容器运行时：调 docker / podman 二进制跑容器。

    接口约定（FakeRuntime 也遵守同一份）：
      - run_container(command, timeout) -> ContainerResult；
      - 超时时抛 InterpreterError（和本地执行器的超时报错同款），
        并尽力把容器删掉（podman/docker run 超时后容器可能还活着）。
    """

    def __init__(self, binary: str, runtime_args: list[str] | None = None):
        self.binary = binary
        self.runtime_args = list(runtime_args or [])

    @classmethod
    def detect(cls, runtime_args: list[str] | None = None):
        """自动找可用的运行时：先 docker，再 podman；都没有就报错。

        只认二进制在不在 PATH 里——能不能真的起容器要到第一次
        run 才知道（这台 VM 的 podman 就是例子：二进制在，但起不来）。
        """
        for binary in ("docker", "podman"):
            if shutil.which(binary):
                return cls(binary, runtime_args)
        raise ValueError(
            "没找到容器运行时：请先装 Docker 或 Podman"
            "（docker.io 或 podman 包），再用 DockerExecutor")

    def _container_name(self, command: list[str]) -> str | None:
        """从命令里找出 --name 的值（超时清理时用）。"""
        try:
            i = command.index("--name")
            return command[i + 1]
        except (ValueError, IndexError):
            return None

    def run_container(self, command: list[str], timeout: float | None) -> ContainerResult:
        try:
            proc = subprocess.run(command, capture_output=True, text=True,
                                  timeout=timeout)
        except subprocess.TimeoutExpired:
            name = self._container_name(command)
            if name:
                # podman/docker run 超时后容器可能还活着，尽力删掉。
                try:
                    subprocess.run([self.binary, "rm", "-f", name],
                                   capture_output=True, timeout=10)
                except Exception:  # noqa: BLE001 - 清理失败不盖过超时本身
                    pass
            raise InterpreterError(
                f"Code execution timed out after {timeout} seconds") from None
        return ContainerResult(proc.returncode, proc.stdout, proc.stderr)


# ---------- runner：容器里实际跑的脚本 ----------

# 注意：这是普通三引号字符串（不是 f-string），里面的 { } 不用转义；
# __TOOLS_SRC__ / __TOOLS_MAP__ 两个占位符由宿主机 .replace() 填。
_RUNNER_TEMPLATE = r'''
"""Docker 执行器在容器里跑的 runner（宿主机每步现生成）。

只干四件事：按白名单守住 socket、把工具和变量放进命名空间、
跑模型代码（最后一条表达式求值）、用行首哨兵把结果/状态/日志/
final_answer/错误传回宿主机。隔离不靠这份脚本——靠容器的
--network/--read-only/--memory/--cap-drop 这些旗；脚本只负责
“跑起来 + 说清楚发生了什么”。
"""
import ast
import json
import socket
import sys

_WORKDIR = sys.argv[1]
_LOGS = []
_real_stdout = sys.stdout


def _emit(kind, payload):
    _real_stdout.write(kind + "::" + json.dumps(payload, ensure_ascii=False) + "\n")
    _real_stdout.flush()


# ---- 输入：变量 + 网络白名单（宿主机写进 _input.json） ----
with open(_WORKDIR + "/_input.json", encoding="utf-8") as _f:
    _input = json.load(_f)

_ALLOWED_HOSTS = _input.get("allowed_hosts")
if _ALLOWED_HOSTS:
    # Python 层的 socket 守卫：只让白名单域名解析和连接。
    # 注意：这不是内核层隔离——容器里跑二进制程序能绕过它，
    # 所以默认还是 --network none；白名单只是“需要联网时”的第二层。
    _orig_getaddrinfo = socket.getaddrinfo
    _orig_create_connection = socket.create_connection

    def _host_allowed(host):
        h = (host or "").lower().rstrip(".")
        return any(h == d or h.endswith("." + d) for d in _ALLOWED_HOSTS)

    def _guard_getaddrinfo(host, port, *args, **kwargs):
        if not _host_allowed(host):
            raise OSError(
                f"Network access to {host!r} is not allowed in the sandbox")
        return _orig_getaddrinfo(host, port, *args, **kwargs)

    def _guard_create_connection(address, *args, **kwargs):
        if not _host_allowed(address[0]):
            raise OSError(
                f"Network access to {address[0]!r} is not allowed in the sandbox")
        return _orig_create_connection(address, *args, **kwargs)

    socket.getaddrinfo = _guard_getaddrinfo
    socket.create_connection = _guard_create_connection


# ---- 工具（宿主机序列化好的源码） ----
__TOOLS_SRC__
_TOOL_FUNCTIONS = {__TOOLS_MAP__}


class _FinalAnswer(Exception):
    """final_answer(x) 在容器里的等价物：带值退出，runner 认出来。"""

    def __init__(self, value):
        super().__init__("final_answer called")
        self.value = value


def _final_answer(answer):
    raise _FinalAnswer(answer)


_ns = {"__builtins__": __builtins__}
_ns.update(_input.get("variables", {}))
for _name, _fn in _TOOL_FUNCTIONS.items():
    _ns[_name] = _fn
_ns["final_answer"] = _final_answer


def _print(*args, **kwargs):
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    _LOGS.append(sep.join(str(a) for a in args) + end)


_ns["print"] = _print


def _jsonable(value):
    """值能不能 JSON 序列化（能就原样带回，不能就用 repr 字符串）。"""
    try:
        json.dumps(value)
        return {"json": value}
    except (TypeError, ValueError):
        return {"repr": repr(value)}


# ---- 跑模型代码（最后一条表达式求值，和本地执行器同款） ----
try:
    with open(_WORKDIR + "/_code.py", encoding="utf-8") as _f:
        _code = _f.read()
    _tree = ast.parse(_code)
    _body = list(_tree.body)
    _last = None
    if _body and isinstance(_body[-1], ast.Expr):
        _last = _body.pop()
    _before = set(_ns)
    for _node in _body:
        _stmt = compile(ast.Module(body=[_node], type_ignores=[]), "<code>", "exec")
        exec(_stmt, _ns)
    _result = None
    if _last is not None:
        _result = eval(compile(ast.Expression(_last.value), "<code>", "eval"), _ns)
    _emit("__RESULT__", _jsonable(_result))
    _state = {}
    for _k, _v in _ns.items():
        if _k in _before or _k.startswith("_"):
            continue
        try:
            json.dumps(_v)
            _state[_k] = _v
        except (TypeError, ValueError):
            pass
    _emit("__STATE__", _state)
    _emit("__LOGS__", "".join(_LOGS))
except _FinalAnswer as _e:
    _emit("__FINAL__", _jsonable(_e.value))
    _emit("__LOGS__", "".join(_LOGS))
except SystemExit as _e:
    if _e.code not in (None, 0):
        _emit("__ERROR__", f"SystemExit: {_e.code}")
    _emit("__LOGS__", "".join(_LOGS))
except Exception as _e:  # noqa: BLE001 - 容器里的任何错都要带回宿主机
    _emit("__ERROR__", f"{type(_e).__name__}: {_e}")
    _emit("__LOGS__", "".join(_LOGS))
'''


# ---------- 工具序列化（宿主机侧） ----------


def _module_imports(module, source_text: str) -> list[str]:
    """从工具所在模块的顶层 import 里，挑出这段源码实际用到的那些。

    只认标准库名字出现过的 import 行——工具源码里没提到的 import
    不带进容器（少带少错）；第三方包的 import 带进去也会在容器里
    报 ModuleNotFoundError，错得明白。
    """
    try:
        module_source = inspect.getsource(module)
    except (OSError, TypeError):
        return []
    try:
        tree = ast.parse(module_source)
    except SyntaxError:
        return []
    lines = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if re.search(rf"\b{re.escape(alias.asname or alias.name.split('.')[0])}\b",
                             source_text):
                    lines.append(f"import {alias.name}" +
                                 (f" as {alias.asname}" if alias.asname else ""))
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if re.search(rf"\b{re.escape(alias.asname or alias.name)}\b", source_text):
                    lines.append(f"from {node.module} import {alias.name}" +
                                 (f" as {alias.asname}" if alias.asname else ""))
    return lines


def _strip_decorators(source: str) -> str:
    """去掉源码开头的 @装饰器 行（@tool 本身在容器里没有定义）。"""
    lines = textwrap.dedent(source).splitlines()
    while lines and lines[0].lstrip().startswith("@"):
        lines.pop(0)
    return "\n".join(lines)


def _serialize_plain_function(func, name: str) -> tuple[str, list[str]]:
    """序列化一个普通函数：源码 + 它用到的模块 import。"""
    try:
        source = _strip_decorators(inspect.getsource(func))
    except (OSError, TypeError) as e:
        raise ValueError(
            f"工具 '{name}' 没有源码，无法放进容器：{e}") from e
    imports = _module_imports(inspect.getmodule(func), source)
    return source, imports


def _serialize_tool(tool) -> tuple[str, list[str]]:
    """把一个工具序列化成容器里能定义的源码。

    返回 (源码, import 行)。final_answer 不走这里——
    runner 里有专门的哨兵版。子智能体工具（forward 里调宿主模型）
    序列化不了：状态里没有模型，源码也拿不到闭包——直接报错。
    """
    name = tool.name
    if name == "final_answer":
        raise ValueError("final_answer 由 runner 特殊处理，不参与序列化")

    forward = tool.forward
    # @tool 的工具：forward 是 staticmethod 包着的 wrapped_function，
    # 真正的函数在它的闭包里（inspect.getclosurevars 能拿到）。
    closure = inspect.getclosurevars(forward)
    original = closure.nonlocals.get("tool_function")
    if original is not None:
        return _serialize_plain_function(original, name)

    # Tool 子类的工具：forward 是绑定方法，取类上的源码，把 self.xxx
    # 换成状态字典查找；实例属性只留能 JSON 序列化的，base_dir 指到 /work。
    try:
        source = textwrap.dedent(inspect.getsource(type(tool).forward))
    except (OSError, TypeError) as e:
        raise ValueError(
            f"工具 '{name}' 没有源码，无法放进容器：{e}") from e
    # def forward(self, ...) → def <工具名>(...)：两种写法都要接住，
    # 括号不能丢（`def forward(self)` 变成 `def read_file()`，不是 `def read_file(`）。
    if re.search(r"def forward\(self, ", source):
        source = re.sub(r"def forward\(self, ", f"def {name}(", source, count=1)
    elif re.search(r"def forward\(self\):", source):
        source = re.sub(r"def forward\(self\):", f"def {name}():", source, count=1)
    else:
        raise ValueError(f"工具 '{name}' 的 forward 签名认不出来，无法序列化进容器")
    if "self." in source:
        # 状态变量名按工具区分（两个子类工具不能共用一个 _tool_state）。
        # 状态值写成 Python 字面量（repr，不是 JSON——容器里跑的是 Python
        # 源码，JSON 的 true/false 在 Python 里是 NameError）；base_dir
        # 不写死 "/work"，写 _WORKDIR 这个名字——runner 在容器里把它
        # 绑成实际的工作目录，FakeRuntime 在宿主机上跑时也一样对得上。
        state_var = f"_tool_state_{name}"
        state_items = []
        for key, value in vars(tool).items():
            if key == "base_dir" and isinstance(value, str):
                state_items.append(f"{key!r}: _WORKDIR")
                continue
            try:
                json.dumps(value)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"工具 '{name}' 的状态 '{key}' 不能 JSON 序列化，"
                    f"无法放进容器：{e}") from e
            state_items.append(f"{key!r}: {value!r}")
        source = re.sub(r"self\.([A-Za-z_][A-Za-z0-9_]*)",
                        rf'{state_var}["\1"]', source)
        source = f"{state_var} = {{{', '.join(state_items)}}}\n" + source
    if "self." in source:
        raise ValueError(f"工具 '{name}' 的 forward 里有 self. 访问，无法序列化进容器")
    try:
        ast.parse(source)
    except SyntaxError as e:
        raise ValueError(f"工具 '{name}' 序列化后的源码解析失败：{e}") from e
    imports = _module_imports(inspect.getmodule(type(tool)), source)
    return source, imports


# ---------- Docker 执行器 ----------


class DockerExecutor:
    """Docker 执行器（上游 remote_executors.py 的 DockerExecutor 同款接口）。

    和 LocalPythonExecutor 一样的用法：send_tools / send_variables /
    reset / __call__(code) -> CodeOutput，state 里同样有 _print_outputs，
    所以 CodeAgent 传 executor=DockerExecutor(...) 就能换，不用改循环。

    Args:
        image: 容器镜像，默认 python:3.12-slim（第一次跑会自动拉）。
        memory / cpus / pids_limit: 资源上限（--memory/--cpus/--pids-limit）。
        network: "none"（默认，断网）或 "default"（走默认网络，
            再配合 allowed_hosts 的 socket 守卫做第二层）。
        allowed_hosts: 允许联网的域名白名单（None = 不守卫）；
            配了它 network 必须是 "default"，否则白名单没有意义。
        timeout_seconds: 每步墙钟超时（容器里超时也算，运行时负责杀容器）。
        workdir: 挂进容器的宿主机目录（默认新建临时目录）；
            容器只能读写这个目录，宿主机别的东西一律挂不进去。
        runtime: 容器运行时实例（SubprocessRuntime 或测试用的 Fake）；
            不传就自动找 docker / podman。
        runtime_args: 传给 docker/podman 的额外参数（放 `run` 前面），
            比如 podman 的 --root/--storage-driver。
    """

    def __init__(self, image: str = DEFAULT_IMAGE,
                 memory: str = DEFAULT_MEMORY,
                 cpus: str = DEFAULT_CPUS,
                 pids_limit: int = DEFAULT_PIDS_LIMIT,
                 network: str = "none",
                 allowed_hosts: list[str] | None = None,
                 timeout_seconds: int | None = DEFAULT_TIMEOUT_SECONDS,
                 workdir: str | None = None,
                 runtime=None,
                 runtime_args: list[str] | None = None):
        if network not in ("none", "default"):
            raise ValueError(f"network 只能是 'none' 或 'default'，不是 {network!r}")
        if allowed_hosts is not None and network != "default":
            raise ValueError("配了 allowed_hosts 就要 network='default'，"
                             "否则白名单没有意义（容器根本没网）")
        self.image = image
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.network = network
        self.allowed_hosts = ([h.lower().rstrip(".") for h in allowed_hosts]
                              if allowed_hosts is not None else None)
        self.timeout_seconds = timeout_seconds
        if runtime is None:
            runtime = SubprocessRuntime.detect(runtime_args)
        self.runtime = runtime
        self.runtime_args = list(runtime_args or [])
        self.workdir = workdir or tempfile.mkdtemp(prefix="docker-executor-")
        os.makedirs(self.workdir, exist_ok=True)
        # 容器里用 nobody 跑，工作目录要让它写得进去。
        os.chmod(self.workdir, 0o777)
        self._tools: dict = {}
        self.state: dict = {}
        self.reset()

    # ---------- 工具 / 变量注入（和本地执行器同款签名） ----------

    def send_tools(self, tools) -> None:
        """把工具记下来（真正序列化是每步跑的时候，见 _build_runner）。

        tools 可以是 ToolRegistry、{名字: Tool} 或 [(名字, Tool)]。
        这里只做一件事：final_answer 之外的工具现在就试序列化一次，
        拿不到源码的、状态带不走的立刻报错——别等模型调了才炸。
        """
        if hasattr(tools, "items"):
            pairs = list(tools.items())
        elif hasattr(tools, "names"):  # ToolRegistry：按名字取
            pairs = [(name, tools[name]) for name in tools.names()]
        else:
            pairs = list(tools)
        for name, tool in pairs:
            if name == "final_answer":
                self._tools[name] = tool
                continue
            _serialize_tool(tool)  # 只校验，不存结果（每步现生成）
            self._tools[name] = tool
    def send_variables(self, variables: dict) -> None:
        """把变量记进 state（只收能 JSON 序列化的——要过 _input.json）。

        工具名、print、__builtins__ 是保留字，不能被变量盖掉。
        """
        reserved = set(self._tools) | {"print", "__builtins__"}
        clash = [k for k in variables if k in reserved]
        if clash:
            raise ValueError(f"变量名和保留字冲突（工具名/print 不能被覆盖）：{clash}")
        for key, value in variables.items():
            try:
                json.dumps(value)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"变量 '{key}' 不能 JSON 序列化，无法送进容器：{e}") from e
        self.state.update(variables)

    def reset(self) -> None:
        """清空命名空间里的变量（每轮 run() 开始时调；工具本身不丢）。"""
        self.state = {}
        self.state["_print_outputs"] = PrintContainer()

    # ---------- 命令构造（单独抽出来，测试不用跑容器也能验隔离配置） ----------

    def _build_command(self, run_name: str) -> list[str]:
        """构造 docker/podman run 命令（隔离配置全在这里，测试锁死它）。

        旗的含义：
          --network none：默认断网（allowed_hosts 配了才走默认网络）；
          --read-only + --tmpfs /tmp：根文件系统只读，临时文件只写 /tmp；
          --memory/--memory-swap/--cpus/--pids-limit：资源上限；
          --cap-drop ALL + --security-opt no-new-privileges：去特权；
          --user 65534:65534：nobody 跑代码，不用 root；
          -v workdir:/work：宿主机只挂这一个目录进来，别的东西看不见。
        """
        binary = self.runtime.binary if hasattr(self.runtime, "binary") else "docker"
        network = "none" if self.network == "none" else "default"
        cmd = [binary, *self.runtime_args, "run", "--rm",
               "--name", run_name,
               "--network", network,
               "--memory", self.memory,
               "--memory-swap", self.memory,
               "--cpus", self.cpus,
               "--pids-limit", str(self.pids_limit),
               "--read-only",
               "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
               "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges",
               "--user", CONTAINER_USER,
               "-e", "HOME=/tmp",
               "-e", "PYTHONDONTWRITEBYTECODE=1",
               "-v", f"{self.workdir}:{CONTAINER_WORKDIR}:rw",
               "-w", CONTAINER_WORKDIR,
               self.image,
               "python", f"{CONTAINER_WORKDIR}/_runner.py", CONTAINER_WORKDIR]
        return cmd

    def _build_runner(self) -> str:
        """生成这一步的 runner 脚本（工具源码 + 占位符替换）。"""
        sources = []
        imports: list[str] = []
        mapping = []
        for name, tool in self._tools.items():
            if name == "final_answer":
                continue
            source, tool_imports = _serialize_tool(tool)
            sources.append(source)
            for line in tool_imports:
                if line not in imports:
                    imports.append(line)
            mapping.append(f'"{name}": {name}')
        header = "\n".join(imports)
        body = "\n\n".join(sources)
        runner = _RUNNER_TEMPLATE.replace(
            "__TOOLS_SRC__", header + "\n\n" + body if (header or body) else "")
        return runner.replace("__TOOLS_MAP__", ", ".join(mapping))

    def _write_step_files(self, code: str) -> None:
        """把这一步要进容器的三个文件写进工作目录（每步都重写，模型
        上一步改了 _runner.py 也没用——下一步会盖掉）。"""
        with open(os.path.join(self.workdir, "_runner.py"), "w", encoding="utf-8") as f:
            f.write(self._build_runner())
        with open(os.path.join(self.workdir, "_code.py"), "w", encoding="utf-8") as f:
            f.write(code)
        with open(os.path.join(self.workdir, "_input.json"), "w", encoding="utf-8") as f:
            variables = {k: v for k, v in self.state.items()
                         if k != "_print_outputs"}
            json.dump({"variables": variables,
                       "allowed_hosts": self.allowed_hosts}, f, ensure_ascii=False)

    # ---------- 执行 ----------

    def __call__(self, code_action: str) -> CodeOutput:
        """在容器里执行一段代码（上游同名）。

        返回 CodeOutput(output=最后一条表达式的值，logs=本段 print 输出，
        is_final_answer=是否调了 final_answer)。output 尽量保持原类型
        （能 JSON 就原值，否则是 repr 字符串——容器内外只能走 JSON）。
        """
        try:
            ast.parse(code_action)
        except SyntaxError as e:
            raise InterpreterError(
                f"Code parsing failed on line {e.lineno} due to: "
                f"{type(e).__name__}: {str(e)}\n{e.text}"
                f"{' ' * (e.offset or 0)}^")
        self.state["_print_outputs"] = PrintContainer()
        self._write_step_files(code_action)
        run_name = f"smolagents-repro-{uuid.uuid4().hex[:12]}"
        command = self._build_command(run_name)
        result = self.runtime.run_container(command, self.timeout_seconds)
        return self._parse_result(result)

    def _parse_result(self, result: ContainerResult) -> CodeOutput:
        """解析容器 stdout 里的哨兵，拼成 CodeOutput（更新 state 和日志）。"""
        markers: dict[str, str] = {}
        for line in (result.stdout or "").splitlines():
            for marker in (_MARKER_RESULT, _MARKER_STATE, _MARKER_LOGS,
                           _MARKER_FINAL, _MARKER_ERROR):
                if line.startswith(marker + "::"):
                    markers[marker] = line[len(marker) + 2:]
                    break
        logs = ""
        if _MARKER_LOGS in markers:
            try:
                logs = json.loads(markers[_MARKER_LOGS])
            except json.JSONDecodeError:
                logs = markers[_MARKER_LOGS]
        if len(logs) > 50000:
            logs = logs[:50000]
        self.state["_print_outputs"] = PrintContainer()
        self.state["_print_outputs"].append(logs)
        if _MARKER_ERROR in markers:
            raise InterpreterError(markers[_MARKER_ERROR])
        if _MARKER_STATE in markers:
            try:
                new_state = json.loads(markers[_MARKER_STATE])
                if isinstance(new_state, dict):
                    for key, value in new_state.items():
                        if key not in self._tools and key != "print":
                            self.state[key] = value
            except json.JSONDecodeError:
                pass
        if _MARKER_FINAL in markers:
            return CodeOutput(output=self._unjson(markers[_MARKER_FINAL]),
                              logs=logs, is_final_answer=True)
        if _MARKER_RESULT in markers:
            return CodeOutput(output=self._unjson(markers[_MARKER_RESULT]),
                              logs=logs, is_final_answer=False)
        # 一个哨兵都没有：runner 没跑起来（镜像拉取失败、运行时报错……），
        # 把 stderr 原样报出来，别让模型对着空结果猜。
        detail = (result.stderr or "").strip()[-2000:]
        raise InterpreterError(
            f"容器没有返回结果（exit {result.returncode}）"
            + (f"：{detail}" if detail else "，且没有 stderr"))

    @staticmethod
    def _unjson(payload: str):
        """把哨兵里的 {"json": v} / {"repr": s} 还原成值。"""
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return payload
        if isinstance(data, dict) and "json" in data:
            return data["json"]
        if isinstance(data, dict) and "repr" in data:
            return data["repr"]
        return data
