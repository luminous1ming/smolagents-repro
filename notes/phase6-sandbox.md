# Phase 6：Sandboxing —— Docker 执行器（2026-11-21）

## 建了什么

| 文件 | 作用 |
|---|---|
| `docker_executor.py` | Docker 执行器：和本地执行器同接口（send_tools / send_variables / reset / `__call__(code) -> CodeOutput`），CodeAgent 传 `executor=` 一换就行；工具源码序列化进容器（@tool 函数取闭包原函数，Tool 子类取 forward 源码、self.xxx 换状态字典，base_dir 指到 runner 的 _WORKDIR）；变量走 `_input.json`（只收 JSON 值），跑完状态带回；runner 用行首哨兵传结果/状态/日志/final_answer/错误；网络白名单是 runner 里的 socket 守卫（第二层）；超时、资源上限、只读根文件系统、去特权、nobody 用户都在命令里 |
| `docker_executor.py`（同文件） | `SubprocessRuntime`（调 docker/podman 二进制，超时杀容器）+ 运行时自动探测（先 docker 再 podman）；`ContainerResult` |
| `agents.py`（改） | CodeAgent 的 executor 参数类型放宽成 `LocalPythonExecutor \| DockerExecutor \| None`（构造口子 Phase 4 已留好，本阶段只改标注和 import） |
| `tests_sandbox.py` | 45 个自测，不联网：`45/45 passed`，另有 1 个真容器测试默认跳过（见下） |
| `tests_models.py` 等五套旧测试 | 重跑无回归：25/25、62/62、45/45、54/54、43/43 |

## 本地执行器不是安全边界（再写一遍，写清楚为什么）

本地执行器（Phase 4）的安全靠的是“进程内否决”：AST 事前检查禁
open()/import os/dunder，再加受限 `__builtins__`。但代码和主进程跑在
同一个 Python 解释器里——检查是“尽量拦”，不是“拦得住”：
解释器本身的求值语义（异常、生成器、描述符、各种边角）太大了，
任何一处检查漏掉，模型代码就能拿到宿主机的文件、进程和网络。
上游文档原话就是 local executor "is not a security sandbox"、
"can be bypassed"。所以本地执行器只防“模型手滑 / 试探绕过”，
防不住有意的攻击；不可信的代码必须走容器。

Docker 执行器的安全模型正好反过来：容器里跑完整 Python，不做 AST
限制（模型想 import os、open() 都行），安全来自容器本身的限制——
断网、只读根文件系统、内存/CPU/进程数上限、去特权、nobody 用户、
宿主机只挂一个工作目录。检查从“代码里不许出现什么”变成了
“容器能碰到什么”，攻击面从解释器语义缩小到容器边界。

## 本机跑不起真容器（实测过程，2026-11-21）

任务要求先花最多 30 分钟争取一个真的容器运行时，结论是不行。
按顺序试的（都在 root 下，systemd 在跑，apt 能用）：

1. `docker`/`podman` 都没装。`apt install podman` 成功（4.9.3），
   但默认 overlay 驱动要 fuse：`fuse: device not found`，`modprobe fuse`
   失败（`/lib/modules/7.0.0-44-generic` 里根本没有 fuse 模块）。
2. 换 vfs 驱动（`--root /var/tmp/podman-vfs --storage-driver vfs`）：
   镜像能拉下来，但 crun 起容器时要连 systemd 的 sd-bus，
   `cannot open sd-bus`。
3. `--cgroup-manager=cgroupfs` 绕开 sd-bus：crun 在配回环网卡时
   `ioctl(SIOCSIFFLAGS): Operation not permitted`。
4. 装 runc（1.3.4）换运行时：`join container mntns: setns:
   operation not permitted`。
5. 查到根因：`/proc/self/status` 里 `Seccomp: 2`（4 个过滤器）——
   VM 的 seccomp 过滤器拦住了 setns 这类调用；`unshare -m` 本身
   是好的（挂载命名空间能建），但 runc/crun 起容器必须 setns
   进容器的命名空间，这一步被拦死。docker.io 也会走同一条路
   （dockerd → containerd → runc），没必要再花时间试。

结论：这台 VM 上没有可用的真实容器运行时。podman 二进制留在了
机器上（以后如果 seccomp 策略放开，它就是现成的），镜像
python:3.12-slim 也拉下来了（在 vfs 的 graphroot 里）。

## 诚实声明：真实隔离没有验证

因为没有真容器，本阶段的“隔离”只做到了两层，各自的证据不同，
不要混为一谈：

- 证明了的（测试锁定）：执行器构造出的容器命令是隔离配置——
  `--network none`、`--read-only`、`--memory/--memory-swap 256m`、
  `--cpus 0.5`、`--pids-limit 64`、`--cap-drop ALL`、
  `--security-opt no-new-privileges`、`--user 65534:65534`、
  宿主机只挂一个工作目录（命令里除此之外没有任何宿主机绝对路径）；
  runner 的全部逻辑在 FakeRuntime 上跑通了（算术、状态回传、
  final_answer、工具序列化、超时、socket 守卫）。
- 没证明的：一个真的容器能不能关住恶意代码。FakeRuntime 在宿主机
  上跑 runner，恶意代码绝不经过它执行（测试里这点是故意的）；
  mock 测试不能证明隔离。真容器测试写在 `tests_sandbox.py`
  第 5 组，默认跳过——在有 Docker 的机器上设
  `DOCKER_EXECUTOR_REAL_RUNTIME=podman`（或 docker）再跑，
  它会验 canary 文件读不到、宿主机目录看不见、外发请求被断网拦住。

## 对照上游的取舍

- 接口照着上游 `PythonExecutor`：send_tools / send_variables /
  reset / `__call__`，CodeOutput 三个字段原样。`state` 里同样留
  `_print_outputs`——CodeAgent 的报错分支（agents.py 里读
  `executor.state.get("_print_outputs")`）不用改。
- 工具进容器是上游远程执行器也要解决的问题，这里的做法是“源码
  序列化”：@tool 函数取闭包里的原函数，Tool 子类取 forward 源码、
  self.xxx 换状态字典。限制写在模块 docstring 里：只支持标准库、
  源码能拿到的工具；拿不到源码、状态不能 JSON 序列化、子智能体
  工具（forward 里调宿主模型）都在 send_tools 时直接报错，不悄悄丢。
- 变量只收 JSON 值（走 _input.json），跑完把 JSON 状态带回——
  本地执行器的“变量跨步保留”在容器版里是“每步重建 runner、
  状态靠 JSON 往返”，语义一样，实现不同。
- 最后一条表达式的值尽量保持原类型（能 JSON 就原值，否则 repr
  字符串）——容器内外只能走 JSON，这是和本地版唯一的行为差别。
- 网络白名单是 Python 层的 socket 守卫（getaddrinfo /
  create_connection），不是内核层隔离——文档和代码里都写明了，
  默认还是断网。另记一笔：这台 VM 的出站流量走 hatch-egress-proxy，
  urllib 实际连的是代理主机，守卫报的名字会是代理而不是目标域名
  （测试里踩到过，已在测试注释里写明）。
- 超时是两层：运行时用 subprocess.run(timeout=...) 杀客户端进程，
  超时后还尽力 `rm -f` 删容器（podman/docker run 超时后容器
  可能还活着）。

## 验收标准对照

- 恶意代码（`import os`、`open('/etc/passwd')`、`subprocess`、
  `().__class__.__base__.__subclasses__()`）在本地执行器下全部被
  InterpreterError 拒绝（Phase 4 的策略，本阶段复测通过）；
  同一批代码在 Docker 执行器下对应的是隔离配置命令（第 2 组断言）。
- 真容器隔离：未验证（见上），留给有 Docker 的机器。
- CodeAgent 接线：`CodeAgent(executor=DockerExecutor(...))`，
  FakeModel 一步算出 5888，通过（第 6 组）。

## 联网检查

本阶段不需要模型调用，全部离线证明（FakeModel / 直接调执行器），
没有用任何 Gemini 额度。真容器跑通后建议补一次联网抽查
（CodeAgent + DockerExecutor 跑一个多工具任务），留给 Phase 7
之前或之后做。

## 留给 Phase 7 的事

- 在有 Docker 的机器上跑 `DOCKER_EXECUTOR_REAL_RUNTIME=podman
  .venv/bin/python tests_sandbox.py`，把第 5 组真容器隔离测试
  跑通，补上“真实隔离”的证据。
- 评测集里的文件任务（写文件、读文件）走 DockerExecutor 跑一遍，
  和本地执行器对比步数/成功率——容器版每步多一次容器启动开销，
  步数一样、耗时会不会差很多，值得量一下。
- managed_agents（子智能体即工具）在 Docker 执行器里不支持，
  Phase 5 的 orchestrator-worker 如果要进容器，需要另外设计
  （子智能体在宿主机跑、只把结果送进容器，或整个多智能体
  都在容器里跑——都不在本阶段范围）。
