#!/usr/bin/env python3
"""Phase 6 自测：Docker 执行器 + 沙箱隔离（不联网）。

Run from the repro directory:
    .venv/bin/python tests_sandbox.py

测六件事：
  1. 构造与校验：没运行时报错、network/allowed_hosts 参数组合、
     变量必须能 JSON 序列化、拿不到源码的工具立刻报错；
  2. 隔离命令构造：断网、只读根文件系统、资源上限、去特权、nobody、
     宿主机只挂工作目录——不用跑容器就能验（FakeRuntime 只记命令）；
  3. Fake 执行（只跑无害代码）：算术 + print、变量跨步保留、
     final_answer 终止、工具在容器里能调（calc/read_file）、
     语法错/运行错/超时/socket 守卫；
  4. 验收对子：同一批恶意代码，本地执行器直接拒绝（InterpreterError），
     Docker 执行器的命令是隔离配置（真容器隔离见第 5 组）；
  5. 真容器隔离：默认跳过——这台 VM 经实测跑不起容器（见
     notes/phase6-sandbox.md），设 DOCKER_EXECUTOR_REAL_RUNTIME=podman
     或 docker 才跑；跑起来就验 canary 文件读不到、宿主机目录看不见、
     外发请求被断网拦住；
  6. CodeAgent 接线：executor=DockerExecutor(...) 一传就换，
     FakeModel 一步算出 5888。

FakeRuntime 的诚实边界（写在明处）：
  - 它在宿主机上用系统 Python 跑 runner，只验证执行器的逻辑——
    命令构造、哨兵解析、状态回传、超时、工具序列化、socket 守卫。
  - 它不模拟断网（宿主机有网）、不模拟只读文件系统、不模拟资源上限、
    也不隔离文件访问。所以第 3 组的通过只证明“执行器逻辑对”，
    不证明“隔离有效”。隔离有效只能由第 5 组在真容器上证明。
  - 因此恶意代码绝不经过 FakeRuntime 执行：第 4 组只验命令配置，
    第 5 组只在真容器上跑。
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import docker_executor
from agents import CodeAgent
from default_tools import default_tools
from docker_executor import (CONTAINER_WORKDIR, ContainerResult,
                             DockerExecutor, SubprocessRuntime)
from executor import InterpreterError, LocalPythonExecutor
from models import FakeModel
from tools import Tool

PASSED = 0
FAILED = 0
SKIPPED = 0


def check(name, ok):
    global PASSED, FAILED
    if ok:
        PASSED += 1
        print(f"PASS {name}")
    else:
        FAILED += 1
        print(f"FAIL {name}")


def skip(name, reason):
    global SKIPPED
    SKIPPED += 1
    print(f"SKIP {name}（{reason}）")


def raises(name, exc_type, fn):
    try:
        fn()
    except exc_type:
        check(name, True)
    except Exception as e:
        print(f"  （{name} 抛了别的错：{type(e).__name__}: {e}）")
        check(name, False)
    else:
        check(name, False)


def _code_reply(code: str) -> str:
    """FakeModel 给 CodeAgent 的回复：Thought + 一个代码块。"""
    return f"Thought: 按计划执行。\n```python\n{code}\n```"


class FakeRuntime:
    """测试替身：不跑容器，在宿主机上用系统 Python 跑 runner。

    只验证执行器逻辑（见模块 docstring 的诚实边界）。run_container
    收到完整命令后，只取其中 -v 挂的工作目录和 _runner.py 的位置，
    用宿主机 Python 跑它——命令本身原样记下来，供隔离断言检查。
    """

    def __init__(self):
        self.commands: list[list[str]] = []
        self.binary = "docker"  # 仅供 _build_command 取名字用

    def run_container(self, command: list[str], timeout):
        self.commands.append(list(command))
        workdir = None
        for i, arg in enumerate(command):
            if arg == "-v":
                workdir = command[i + 1].split(":")[0]
        runner = os.path.join(workdir, "_runner.py")
        try:
            proc = subprocess.run([sys.executable, runner, workdir],
                                  capture_output=True, text=True,
                                  timeout=timeout, cwd=workdir)
        except subprocess.TimeoutExpired:
            raise InterpreterError(
                f"Code execution timed out after {timeout} seconds") from None
        return ContainerResult(proc.returncode, proc.stdout, proc.stderr)


def _make_executor(**kwargs):
    """FakeRuntime 版 DockerExecutor（工作目录用临时目录）。"""
    tmp = tempfile.mkdtemp(prefix="sandbox-test-")
    kwargs.setdefault("runtime", FakeRuntime())
    kwargs.setdefault("workdir", tmp)
    return DockerExecutor(**kwargs), tmp


def _flags(command: list[str]) -> set[str]:
    """命令里的旗（--xxx 形式）集合。"""
    return {a for a in command if a.startswith("--")}


MALICIOUS_SNIPPETS = [
    "import os\nprint(os.getcwd())",
    "open('/etc/passwd').read()",
    "import subprocess\nsubprocess.run(['id'])",
    "().__class__.__base__.__subclasses__()",
]


# ---------- 1. 构造与校验 ----------

_real_which = shutil.which
docker_executor.shutil.which = lambda name: None
raises("no runtime binary -> ValueError",
      ValueError, lambda: DockerExecutor())
docker_executor.shutil.which = _real_which

raises("bad network -> ValueError",
      ValueError, lambda: DockerExecutor(runtime=FakeRuntime(), network="host"))
raises("allowed_hosts with network none -> ValueError",
      ValueError, lambda: DockerExecutor(runtime=FakeRuntime(),
                                         allowed_hosts=["api.duckduckgo.com"]))

ex, _tmp = _make_executor()
raises("non-JSON variable -> ValueError",
      ValueError, lambda: ex.send_variables({"f": lambda: 1}))
raises("reserved variable name -> ValueError",
      ValueError, lambda: ex.send_variables({"print": "x"}))

# 拿不到源码的工具（exec 现造的类，inspect.getsource 找不到文件）：
_ns = {}
exec("from tools import Tool\n"
     "class DynTool(Tool):\n"
     "    name = 'dyn'\n"
     "    description = 'd'\n"
     "    inputs = {}\n"
     "    output_type = 'string'\n"
     "    def forward(self):\n"
     "        return 'x'\n", _ns)
raises("exec-defined tool has no source -> ValueError",
      ValueError, lambda: ex.send_tools([("dyn", _ns["DynTool"]())]))


# ---------- 2. 隔离命令构造（不跑容器） ----------

ex, tmp = _make_executor()
cmd = ex._build_command("test-name")
flags = _flags(cmd)
check("command: --network none by default", "--network" in cmd
      and cmd[cmd.index("--network") + 1] == "none")
check("command: --read-only", "--read-only" in flags)
check("command: memory limits", cmd[cmd.index("--memory") + 1] == "256m"
      and cmd[cmd.index("--memory-swap") + 1] == "256m")
check("command: --cpus 0.5", cmd[cmd.index("--cpus") + 1] == "0.5")
check("command: --pids-limit 64", cmd[cmd.index("--pids-limit") + 1] == "64")
check("command: --cap-drop ALL", cmd[cmd.index("--cap-drop") + 1] == "ALL")
check("command: --security-opt no-new-privileges",
      "no-new-privileges" in cmd)
check("command: --user nobody", cmd[cmd.index("--user") + 1] == "65534:65534")
check("command: --tmpfs /tmp", any(a.startswith("/tmp:") for a in cmd))
check("command: exactly one -v, workdir only",
      cmd.count("-v") == 1
      and cmd[cmd.index("-v") + 1] == f"{tmp}:{CONTAINER_WORKDIR}:rw")
check("command: -w /work + python /work/_runner.py /work",
      cmd[cmd.index("-w") + 1] == CONTAINER_WORKDIR
      and cmd[-3:] == ["python", f"{CONTAINER_WORKDIR}/_runner.py",
                       CONTAINER_WORKDIR])
check("command: image python:3.12-slim", "python:3.12-slim" in cmd)
# 除了工作目录挂载（-v 的值就是 workdir:/work:rw），命令里不出现
# 别的宿主机绝对路径。
host_paths = [a for a in cmd if a.startswith("/") and a != tmp
              and a != f"{tmp}:{CONTAINER_WORKDIR}:rw"
              and not a.startswith(CONTAINER_WORKDIR)
              and a not in ("/tmp:rw,noexec,nosuid,size=64m",)]
check("command: no host path besides workdir", host_paths == [])

ex2, _tmp2 = _make_executor(network="default",
                            allowed_hosts=["api.duckduckgo.com"])
cmd2 = ex2._build_command("test-name")
check("command: allowed_hosts -> --network default",
      cmd2[cmd2.index("--network") + 1] == "default")
ex3, _tmp3 = _make_executor(memory="512m", cpus="1")
cmd3 = ex3._build_command("x")
check("command: custom memory/cpus honored",
      cmd3[cmd3.index("--memory") + 1] == "512m"
      and cmd3[cmd3.index("--cpus") + 1] == "1")


# ---------- 3. Fake 执行（只跑无害代码） ----------

ex, tmp = _make_executor()
out = ex("x = 40 + 2\nprint('hi')\nx")
check("fake: arithmetic + print + int result",
      out.output == 42 and out.logs == "hi\n" and not out.is_final_answer)

ex("n = 7")
out2 = ex("n * 6")
check("fake: state persists across calls",
      out2.output == 42 and ex.state.get("n") == 7)

out3 = ex('final_answer("done")')
check("fake: final_answer terminates",
      out3.is_final_answer and out3.output == "done")

with open(os.path.join(tmp, "nums.txt"), "w", encoding="utf-8") as f:
    f.write("10\n20\n35\n")
ex.send_tools(default_tools(base_dir=tmp))
out4 = ex('calc(expression="128*46")')
check("fake: calc tool in container", str(out4.output) == "5888")
out5 = ex('read_file(path="nums.txt")')
check("fake: read_file tool reads workdir",
      "10\n20\n35" in str(out5.output))
out6 = ex('read_file(path="../secret.txt")')
check("fake: read_file refuses outside workdir",
      "只能读" in str(out6.output))

raises("fake: syntax error -> InterpreterError",
      InterpreterError, lambda: ex("def broken("))
check("fake: syntax error runs no container",
      len(ex.runtime.commands) == 7)  # 前面 7 次成功调用各 1 条命令
raises("fake: runtime error -> InterpreterError",
      InterpreterError, lambda: ex("1 / 0"))
out_err = None
try:
    ex("print('before')\nraise ValueError('boom')")
except InterpreterError:
    out_err = str(ex.state["_print_outputs"])
check("fake: logs survive an error",
      out_err is not None and "before" in out_err)

ex_slow, _tmp_slow = _make_executor(timeout_seconds=2)
raises("fake: sleep past timeout -> InterpreterError",
      InterpreterError, lambda: ex_slow("import time\ntime.sleep(30)"))

ex_guard, _tmp_guard = _make_executor(network="default",
                                      allowed_hosts=["api.duckduckgo.com"])
raises("fake: non-listed host blocked by socket guard",
      InterpreterError,
      lambda: ex_guard("import urllib.request\n"
                       "urllib.request.urlopen('https://example.com', timeout=5)"))
try:
    ex_guard("import urllib.request\n"
             "urllib.request.urlopen('https://example.com', timeout=5)")
except InterpreterError as e:
    # 注意：这台 VM 的出站流量走 hatch-egress-proxy，urllib 实际连的是
    # 代理主机，所以守卫报的名字是代理，不是 example.com——守卫本身
    # 照常触发了（"not allowed in the sandbox"）。
    check("fake: guard error says not allowed",
          "not allowed in the sandbox" in str(e))
runner_src = ex_guard._build_runner()
check("fake: guard code in runner when allowed_hosts set",
      "_ALLOWED_HOSTS" in runner_src and "getaddrinfo" in runner_src)
check("fake: allowed_hosts in _input.json",
      json.load(open(os.path.join(_tmp_guard, "_input.json"),
                     encoding="utf-8"))["allowed_hosts"] == ["api.duckduckgo.com"])


# ---------- 4. 验收对子：本地拒绝 vs Docker 命令隔离 ----------

local = LocalPythonExecutor()
for snippet in MALICIOUS_SNIPPETS:
    raises(f"local refuses: {snippet.splitlines()[0][:28]}",
          InterpreterError, lambda s=snippet: local(s))

ex, tmp = _make_executor()
for snippet in MALICIOUS_SNIPPETS:
    cmd = ex._build_command("malicious-probe")
    check(f"docker command isolated for: {snippet.splitlines()[0][:28]}",
          "--network" in cmd and cmd[cmd.index("--network") + 1] == "none"
          and "--read-only" in _flags(cmd)
          and "--cap-drop" in _flags(cmd)
          and cmd.count("-v") == 1)
# 注意：上面只验了“命令是隔离配置”——恶意代码没有真的跑过 FakeRuntime。
# 真的隔离见第 5 组（需要真容器）。


# ---------- 5. 真容器隔离（默认跳过） ----------

_real_runtime = os.environ.get("DOCKER_EXECUTOR_REAL_RUNTIME")
if _real_runtime not in ("docker", "podman"):
    skip("real container containment",
         "未设 DOCKER_EXECUTOR_REAL_RUNTIME；本 VM 经实测跑不起容器，"
         "真实隔离未验证（见 notes/phase6-sandbox.md）")
else:
    import secrets
    with tempfile.TemporaryDirectory() as host_tmp:
        canary = os.path.join(host_tmp, "canary.txt")
        token = "CANARY-" + secrets.token_hex(8)
        with open(canary, "w", encoding="utf-8") as f:
            f.write(token)
        workdir = os.path.join(host_tmp, "work")
        os.makedirs(workdir)
        rex = DockerExecutor(runtime=SubprocessRuntime(_real_runtime),
                             workdir=workdir, timeout_seconds=60)
        # (a) 读宿主机文件：/etc/passwd 是容器自己的，canary 读不到。
        out = rex(f"import os\nprint(open('/etc/passwd').read()[:200])")
        check("real: container /etc/passwd has no host marker",
              "hatch" not in str(out.output))
        try:
            rex(f"print(open({canary!r}).read())")
            check("real: host canary file unreadable", False)
        except InterpreterError:
            check("real: host canary file unreadable", True)
        # (b) 宿主机目录看不见。
        out = rex("import os\nprint(os.listdir('/'))")
        check("real: host dirs invisible from container",
              token not in str(out.output) and "workspace" not in str(out.output))
        # (c) 外发请求被断网拦住。
        try:
            rex("import urllib.request\n"
                "urllib.request.urlopen('https://example.com', timeout=5)")
            check("real: network blocked", False)
        except InterpreterError:
            check("real: network blocked", True)
        # (d) 无害代码在真容器里照常跑。
        out = rex('final_answer(128*46)')
        check("real: benign code works in container",
              out.is_final_answer and out.output == 5888)


# ---------- 6. CodeAgent 接线 ----------

with tempfile.TemporaryDirectory() as tmp:
    with open(os.path.join(tmp, "nums.txt"), "w", encoding="utf-8") as f:
        f.write("10\n20\n35\n")
    agent = CodeAgent(
        tools=default_tools(base_dir=tmp),
        model=FakeModel([_code_reply(
            'result = calc(expression="128*46")\nfinal_answer(result)')]),
        executor=DockerExecutor(runtime=FakeRuntime(), workdir=tmp))
    result = agent.run("算一下 128*46", return_full_result=True)
    steps = [s for s in result.steps if "step_number" in s]
    check("code agent with docker executor: one step, 5888",
          str(result.output) == "5888" and len(steps) == 1)

print(f"\n{PASSED}/{PASSED + FAILED} passed"
      + (f", {SKIPPED} skipped" if SKIPPED else "")
      + (f", {FAILED} failed" if FAILED else ""))
sys.exit(1 if FAILED else 0)
