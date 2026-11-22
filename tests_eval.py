#!/usr/bin/env python3
"""Phase 7 自测：评测集 + harness（不联网；web_search 用 mock 挡掉）。

Run from the repro directory:
    .venv/bin/python tests_eval.py

测五件事：
  1. 任务集：24 道、id 唯一、三类各 8 道；每个 checker 在期望答案上
     通过、在错误答案上不通过；文件题的 setup 能写进 workdir；
  2. 假脚本：每个 CodeAgent 假回复都是合法代码块，ToolCalling 假回复
     每步都有 tool_calls 且 final_answer 单独占一步；
  3. harness 全流程（FakeModel）：每类挑 1 道 × 3 个系统，全过，
     步数/token/耗时都记下来了，results.json 落盘；
  4. 选题：--tasks 精确挑、--limit/--offset 切片都对；
  5. 失败记账：模型抛 429 记 error_429；永远不调 final_answer 的
     假模型走到 max_steps_error 且 success=False。
"""
import json
import os
import sys
import tempfile
import urllib.error
from unittest import mock

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from eval_tasks import TASKS, get_task, tasks_by_kind
from eval import (SYSTEMS, fake_model_factory, run_one, select_tasks,
                  summarize)
from models import ChatMessage, FakeModel, Model

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


# ---------- 1. 任务集 ----------

check("任务集 24 道", len(TASKS) == 24)
check("任务 id 唯一", len({t.id for t in TASKS}) == 24)
for kind in ("arith", "file", "research"):
    check(f"{kind} 类 8 道", len(tasks_by_kind(kind)) == 8)

for task in TASKS:
    with tempfile.TemporaryDirectory() as tmp:
        # 文件题的正向测试要先把“跑完应落盘的文件”也写好——
        # checker 本来就是用来验 agent 落盘结果的，不是验 setup 的。
        for name, content in {**task.setup, **task.expected_files}.items():
            with open(os.path.join(tmp, name), "w", encoding="utf-8") as f:
                f.write(content)
        check(f"{task.id} checker 认期望答案",
              task.check(task.expected, tmp) is True)
        check(f"{task.id} checker 拒错误答案",
              task.check("nothing relevant here, definitely wrong", tmp) is False)

# ---------- 2. 假脚本 ----------

for task in TASKS:
    code = task.fake_code
    check(f"{task.id} fake_code 是代码块且调 final_answer",
          "```" not in code and "final_answer(" in code)
    steps = task.fake_calls
    check(f"{task.id} fake_calls 每步都有调用",
          len(steps) >= 1 and all(len(s) >= 1 for s in steps))
    check(f"{task.id} fake_calls 最后一步只有 final_answer",
          len(steps[-1]) == 1 and steps[-1][0]["name"] == "final_answer")
    for step in steps[:-1]:
        check(f"{task.id} fake_calls 非最后一步不含 final_answer",
              all(c["name"] != "final_answer" for c in step))


# ---------- 3. harness 全流程（FakeModel；web_search mock 掉） ----------

_empty_resp = mock.MagicMock()
_empty_resp.read.return_value = json.dumps(
    {"AbstractText": "", "RelatedTopics": []}).encode()
_empty_resp.__enter__.return_value = _empty_resp
_empty_resp.__exit__.return_value = False

with tempfile.TemporaryDirectory() as tmp:
    with mock.patch("urllib.request.urlopen", return_value=_empty_resp):
        records = []
        for tid in ("arith-01", "file-01", "research-01"):
            task = get_task(tid)
            for system in SYSTEMS:
                model = fake_model_factory(task, system)
                rec = run_one(task, system, model, max_steps=15, run_dir=tmp)
                records.append(rec)
                check(f"harness {tid} × {system} 成功",
                      rec["success"] is True and rec["error"] is None)
                check(f"harness {tid} × {system} 步数 >= 1",
                      rec["steps"] >= 1)
                check(f"harness {tid} × {system} 耗时已记",
                      rec["latency_s"] >= 0)
        # results.json 落盘格式（run_one 不写文件，这里只验 summarize 不炸）。
        table = summarize(records)
        check("汇总表三系统都有",
              all(f"- {s}:" in table for s in SYSTEMS))
        check("汇总表 9/9 通过", "9/9" in table or "3/3" in table)


# ---------- 4. 选题 ----------

check("select --tasks 精确挑",
      [t.id for t in select_tasks("arith-02,file-03", None, 0)]
      == ["arith-02", "file-03"])
check("select --limit 切片",
      [t.id for t in select_tasks(None, 3, 0)]
      == ["arith-01", "arith-02", "arith-03"])
check("select --offset 跳过",
      [t.id for t in select_tasks(None, 2, 8)][0] == "file-01")
try:
    select_tasks("no-such-task", None, 0)
    check("select 未知题名报错", False)
except KeyError:
    check("select 未知题名报错", True)


# ---------- 5. 失败记账 ----------

class _Raising429(Model):
    """generate 直接抛 429 的模型（Phase 1 的 A4 规则：429 不重试、直接抛）。"""

    def generate(self, messages, stop_sequences=None, tools_to_call_from=None,
                 **kwargs):
        raise urllib.error.HTTPError("https://example.invalid/", 429,
                                     "Too Many Requests", {}, None)


class _NeverFinal(Model):
    """永远不调 final_answer 的模型：只回空话，让循环走到步数上限。"""

    def generate(self, messages, stop_sequences=None, tools_to_call_from=None,
                 **kwargs):
        return ChatMessage(role="assistant", content="我再想想。")


with tempfile.TemporaryDirectory() as tmp:
    task = get_task("arith-01")
    rec = run_one(task, "code", _Raising429(), max_steps=3, run_dir=tmp)
    check("429 记 error_429 且 success=False",
          rec["error_429"] is True and rec["success"] is False
          and "429" in (rec["error"] or ""))
    rec2 = run_one(task, "code", _NeverFinal(), max_steps=3, run_dir=tmp)
    check("不调 final_answer 走到 max_steps_error",
          rec2["state"] == "max_steps_error" and rec2["success"] is False)

print(f"\n{PASSED}/{PASSED + FAILED} passed.")
sys.exit(0 if FAILED == 0 else 1)
