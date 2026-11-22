#!/usr/bin/env python3
"""Phase 7：评测任务集 —— 24 道自写题目，答案都可核查。

对照 GAIA 的思路（多步、可核查答案），但题目全部自己写——
不用任何风控数据，也不用现成 benchmark 的题目（避免“训过”
的嫌疑，答案就是常识 / 当场算出来的数 / 当场写出来的文件）。

三类，每类 8 道：
  - arith：算术/代码题，答案是一个数，checker 在最终答案里找这个数；
  - file：文件操作题，checker 看 workdir 里落盘的文件（+ 答案里
    的关键数），每道题的 setup 是跑前写进 workdir 的数据文件；
  - research：网络研究题，答案是可核查的事实（+ 来源名），checker
    看答案里有没有期望的子串。事实都选稳定的（不用“当前人口”
    这种会变的数——Phase 0 的 tokyo 题就是前车之鉴）；引用本身
    对不对不在这里打分（那是 Phase 5 CitationAgent 的验收标准）。

每道题还带两份“假答案脚本”，供 FakeModel 全量跑通用——
证明任务本身是可解的形状（工具够、答案可核查），不是证明
模型会做：
  - fake_code：给 CodeAgent 的一整块代码（Thought + ```python），
    算术题是真算，文件题是真读写，研究题直接交期望答案；
  - fake_calls：给 ToolCallingAgent 的分步 tool_calls 序列
    （每步一个列表；final_answer 必须单独占一步）。

用法：
    from eval_tasks import TASKS, get_task
    task = get_task("arith-01")
    task.check("432", "/tmp/workdir")  # -> True / False
"""
import re
from dataclasses import dataclass, field


# ---------- checker 工厂：题目只写“期望”，判定逻辑集中在这里 ----------

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _numbers_in(text: str) -> list[float]:
    """答案里出现的所有数（去逗号千分位），供 contains_number 用。"""
    return [float(x.replace(",", "")) for x in _NUMBER_RE.findall(text or "")]


def contains_number(value: float, tol: float = 0.01):
    """答案里出现期望的数（允许小误差，80 和 80.0 都算）。"""
    def check(answer: str, workdir: str) -> bool:
        return any(abs(n - value) <= tol for n in _numbers_in(answer))
    return check


def contains_all(*substrings: str):
    """答案里出现全部期望子串（不区分大小写；先去掉千分位逗号，
    否则 "8,848" 匹配不到 "8848"——FakeModel 全量跑抓到的真 bug）。"""
    need = [s.lower().replace(",", "") for s in substrings]

    def check(answer: str, workdir: str) -> bool:
        text = (answer or "").lower().replace(",", "")
        return all(s in text for s in need)
    return check


def _read(workdir: str, path: str) -> str | None:
    try:
        with open(f"{workdir}/{path}", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def file_equals(path: str, expected: str, need_answer: tuple[str, ...] = ()):
    """落盘文件内容 == 期望（去首尾空白），答案里也要出现关键子串。"""
    def check(answer: str, workdir: str) -> bool:
        content = _read(workdir, path)
        if content is None or content.strip() != expected.strip():
            return False
        text = (answer or "").lower()
        return all(s.lower() in text for s in need_answer)
    return check


def file_lines(path: str, expected_lines: list[str],
               need_answer: tuple[str, ...] = ()):
    """落盘文件按行 == 期望行列表，答案里也要出现关键子串。"""
    def check(answer: str, workdir: str) -> bool:
        content = _read(workdir, path)
        if content is None:
            return False
        lines = [line for line in content.strip().split("\n")]
        if lines != expected_lines:
            return False
        text = (answer or "").lower()
        return all(s.lower() in text for s in need_answer)
    return check


def file_float(path: str, expected: float, tol: float = 0.01,
               need_answer: tuple[str, ...] = ()):
    """落盘文件是一个数，误差内算对；答案里也要出现关键子串。"""
    def check(answer: str, workdir: str) -> bool:
        content = _read(workdir, path)
        if content is None:
            return False
        try:
            value = float(content.strip())
        except ValueError:
            return False
        if abs(value - expected) > tol:
            return False
        text = (answer or "").lower()
        return all(s.lower() in text for s in need_answer)
    return check


# ---------- 题目数据 ----------


@dataclass
class EvalTask:
    """一道评测题。check(answer, workdir) -> bool；setup 是跑前
    写进 workdir 的数据文件；fake_code / fake_calls 是 FakeModel
    用的“假答案脚本”（见模块 docstring）。"""

    id: str
    kind: str  # "arith" | "file" | "research"
    prompt: str
    expected: str  # 给人看的期望答案
    check: object = None  # (answer, workdir) -> bool
    setup: dict = field(default_factory=dict)  # 文件名 -> 内容
    expected_files: dict = field(default_factory=dict)  # 文件题跑完应落盘的文件
    fake_code: str = ""  # CodeAgent 假回复：整块代码
    fake_calls: list = field(default_factory=list)  # ToolCalling 假回复：每步一组调用


def _final_answer_call(text: str) -> list:
    """ToolCalling 假脚本的通用收尾：单独一步调 final_answer。"""
    return [{"name": "final_answer", "arguments": {"answer": text}}]


TASKS: list[EvalTask] = [
    # ---------- arith：算术/代码（8 道） ----------
    EvalTask(
        id="arith-01", kind="arith",
        prompt="Compute 17 * 23 + 41 and report the result.",
        expected="432",
        check=contains_number(432),
        fake_code='final_answer(str(17 * 23 + 41))',
        fake_calls=[
            [{"name": "calc", "arguments": {"expression": "17*23+41"}}],
            _final_answer_call("432")],
    ),
    EvalTask(
        id="arith-02", kind="arith",
        prompt="What is the sum of all integers from 1 to 100? Report the number.",
        expected="5050",
        check=contains_number(5050),
        fake_code="final_answer(str(sum(range(1, 101))))",
        fake_calls=[
            [{"name": "calc", "arguments": {"expression": "100*101/2"}}],
            _final_answer_call("5050")],
    ),
    EvalTask(
        id="arith-03", kind="arith",
        prompt="Compute 2^10 + 3^5 and report the result.",
        expected="1267",
        check=contains_number(1267),
        fake_code="final_answer(str(2**10 + 3**5))",
        fake_calls=[
            [{"name": "calc", "arguments": {"expression": "2**10+3**5"}}],
            _final_answer_call("1267")],
    ),
    EvalTask(
        id="arith-04", kind="arith",
        prompt="The Fibonacci sequence starts 1, 1, 2, 3, 5, 8, ... "
               "What is the 12th Fibonacci number? Report the number.",
        expected="144",
        check=contains_number(144),
        fake_code=("fib = [1, 1]\n"
                   "while len(fib) < 12:\n"
                   "    fib.append(fib[-1] + fib[-2])\n"
                   "final_answer(str(fib[11]))"),
        fake_calls=[
            [{"name": "calc", "arguments": {"expression": "1+1+2+3+5+8+13+21+34+55+89+144"}}],
            _final_answer_call("144")],
    ),
    EvalTask(
        id="arith-05", kind="arith",
        prompt="A train travels 240 km in 3 hours at constant speed. "
               "How fast is it in km/h? Report the number.",
        expected="80",
        check=contains_number(80),
        fake_code="final_answer(str(240 / 3))",
        fake_calls=[
            [{"name": "calc", "arguments": {"expression": "240/3"}}],
            _final_answer_call("80")],
    ),
    EvalTask(
        id="arith-06", kind="arith",
        prompt="What is 15% of 840? Report the number.",
        expected="126",
        check=contains_number(126),
        fake_code="final_answer(str(840 * 0.15))",
        fake_calls=[
            [{"name": "calc", "arguments": {"expression": "840*0.15"}}],
            _final_answer_call("126")],
    ),
    EvalTask(
        id="arith-07", kind="arith",
        prompt="How many seconds are there in 3 days? Report the number.",
        expected="259200",
        check=contains_number(259200),
        fake_code="final_answer(str(3 * 24 * 60 * 60))",
        fake_calls=[
            [{"name": "calc", "arguments": {"expression": "3*24*60*60"}}],
            _final_answer_call("259200")],
    ),
    EvalTask(
        id="arith-08", kind="arith",
        prompt="What is the sum of the squares of the integers from 1 to 5? "
               "Report the number.",
        expected="55",
        check=contains_number(55),
        fake_code="final_answer(str(sum(n * n for n in range(1, 6))))",
        fake_calls=[
            [{"name": "calc", "arguments": {"expression": "1+4+9+16+25"}}],
            _final_answer_call("55")],
    ),

    # ---------- file：文件操作（8 道；setup 文件在 workdir 根目录） ----------
    EvalTask(
        id="file-01", kind="file",
        prompt="Read the file numbers.txt in the current working directory. "
               "It contains one integer per line. Sum all the numbers, "
               "multiply the sum by 2, write the final result to out.txt, "
               "and report both the sum and the final result.",
        expected="sum=65, result=130",
        check=file_equals("out.txt", "130", need_answer=("65", "130")),
        expected_files={"out.txt": "130"},
        setup={"numbers.txt": "12\n7\n23\n9\n14\n"},
        fake_code=('content = read_file("numbers.txt")\n'
                   "total = sum(int(x) for x in content.strip().split())\n"
                   'write_file("out.txt", str(total * 2))\n'
                   'final_answer("The sum is 65 and the final result is 130.")'),
        fake_calls=[
            [{"name": "read_file", "arguments": {"path": "numbers.txt"}}],
            [{"name": "calc", "arguments": {"expression": "65*2"}}],
            [{"name": "write_file", "arguments": {"path": "out.txt", "content": "130"}}],
            _final_answer_call("The sum is 65 and the final result is 130.")],
    ),
    EvalTask(
        id="file-02", kind="file",
        prompt="Create a file called primes.txt in the current working directory "
               "containing the first 5 prime numbers, one per line, then read it "
               "back to confirm the contents and report them.",
        expected="2, 3, 5, 7, 11",
        check=file_lines("primes.txt", ["2", "3", "5", "7", "11"],
                         need_answer=("2", "3", "5", "7", "11")),
        expected_files={"primes.txt": "2\n3\n5\n7\n11\n"},
        fake_code=('write_file("primes.txt", "2\\n3\\n5\\n7\\n11\\n")\n'
                   'content = read_file("primes.txt")\n'
                   'final_answer("The first 5 primes are 2, 3, 5, 7, 11.")'),
        fake_calls=[
            [{"name": "write_file",
              "arguments": {"path": "primes.txt", "content": "2\n3\n5\n7\n11\n"}}],
            [{"name": "read_file", "arguments": {"path": "primes.txt"}}],
            _final_answer_call("The first 5 primes are 2, 3, 5, 7, 11.")],
    ),
    EvalTask(
        id="file-03", kind="file",
        prompt="Count the words in the file words.txt in the current working "
               "directory, write the count to count.txt, and report it.",
        expected="14",
        check=file_equals("count.txt", "14", need_answer=("14",)),
        expected_files={"count.txt": "14"},
        setup={"words.txt": "the quick brown fox\njumps over the lazy dog\n"
                            "hello world from the agent\n"},
        fake_code=('content = read_file("words.txt")\n'
                   "count = len(content.split())\n"
                   'write_file("count.txt", str(count))\n'
                   'final_answer("The file has 14 words.")'),
        fake_calls=[
            [{"name": "read_file", "arguments": {"path": "words.txt"}}],
            [{"name": "write_file", "arguments": {"path": "count.txt", "content": "14"}}],
            _final_answer_call("The file has 14 words.")],
    ),
    EvalTask(
        id="file-04", kind="file",
        prompt="Compute the average of the temperatures in the file temps.txt "
               "in the current working directory (one number per line), "
               "write it to avg.txt, and report it.",
        expected="22.0",
        check=file_float("avg.txt", 22.0, need_answer=("22",)),
        expected_files={"avg.txt": "22.0"},
        setup={"temps.txt": "21\n23\n19\n25\n22\n"},
        fake_code=('content = read_file("temps.txt")\n'
                   "temps = [float(x) for x in content.strip().split()]\n"
                   "avg = sum(temps) / len(temps)\n"
                   'write_file("avg.txt", str(avg))\n'
                   'final_answer("The average temperature is 22.0.")'),
        fake_calls=[
            [{"name": "read_file", "arguments": {"path": "temps.txt"}}],
            [{"name": "calc", "arguments": {"expression": "(21+23+19+25+22)/5"}}],
            [{"name": "write_file", "arguments": {"path": "avg.txt", "content": "22.0"}}],
            _final_answer_call("The average temperature is 22.0.")],
    ),
    EvalTask(
        id="file-05", kind="file",
        prompt="Create a file called squares.csv in the current working directory "
               "with two columns, n and n_squared, for n from 1 to 5, one row "
               "per line including a header row, then report the contents.",
        expected="n,n_squared / 1,1 / 2,4 / 3,9 / 4,16 / 5,25",
        check=file_equals("squares.csv",
                          "n,n_squared\n1,1\n2,4\n3,9\n4,16\n5,25",
                          need_answer=("25",)),
        expected_files={"squares.csv": "n,n_squared\n1,1\n2,4\n3,9\n4,16\n5,25\n"},
        fake_code=('rows = ["n,n_squared"] + [f"{n},{n*n}" for n in range(1, 6)]\n'
                   'write_file("squares.csv", "\\n".join(rows) + "\\n")\n'
                   'final_answer("Created squares.csv: 1,1 / 2,4 / 3,9 / 4,16 / 5,25.")'),
        fake_calls=[
            [{"name": "write_file",
              "arguments": {"path": "squares.csv",
                            "content": "n,n_squared\n1,1\n2,4\n3,9\n4,16\n5,25\n"}}],
            _final_answer_call("Created squares.csv: 1,1 / 2,4 / 3,9 / 4,16 / 5,25.")],
    ),
    EvalTask(
        id="file-06", kind="file",
        prompt="Append the word 'cherry' as a new line to the file fruits.txt "
               "in the current working directory, then report the new line count.",
        expected="3 lines: apple, banana, cherry",
        check=file_lines("fruits.txt", ["apple", "banana", "cherry"],
                         need_answer=("3",)),
        expected_files={"fruits.txt": "apple\nbanana\ncherry\n"},
        setup={"fruits.txt": "apple\nbanana\n"},
        fake_code=('content = read_file("fruits.txt")\n'
                   'lines = [line for line in content.strip().split("\\n") if line]\n'
                   'lines.append("cherry")\n'
                   'write_file("fruits.txt", "\\n".join(lines) + "\\n")\n'
                   'final_answer("The file now has 3 lines: apple, banana, cherry.")'),
        fake_calls=[
            [{"name": "read_file", "arguments": {"path": "fruits.txt"}}],
            [{"name": "write_file",
              "arguments": {"path": "fruits.txt", "content": "apple\nbanana\ncherry\n"}}],
            _final_answer_call("The file now has 3 lines: apple, banana, cherry.")],
    ),
    EvalTask(
        id="file-07", kind="file",
        prompt="Sort the numbers in the file unsorted.txt in the current working "
               "directory in ascending order, write them to sorted.txt "
               "(one per line), and report the result.",
        expected="1, 3, 5, 7, 9",
        check=file_equals("sorted.txt", "1\n3\n5\n7\n9",
                          need_answer=("1", "3", "5", "7", "9")),
        expected_files={"sorted.txt": "1\n3\n5\n7\n9\n"},
        setup={"unsorted.txt": "9\n3\n7\n1\n5\n"},
        fake_code=('content = read_file("unsorted.txt")\n'
                   "nums = sorted(int(x) for x in content.strip().split())\n"
                   'write_file("sorted.txt", "\\n".join(str(n) for n in nums) + "\\n")\n'
                   'final_answer("Sorted: 1, 3, 5, 7, 9.")'),
        fake_calls=[
            [{"name": "read_file", "arguments": {"path": "unsorted.txt"}}],
            [{"name": "write_file",
              "arguments": {"path": "sorted.txt", "content": "1\n3\n5\n7\n9\n"}}],
            _final_answer_call("Sorted: 1, 3, 5, 7, 9.")],
    ),
    EvalTask(
        id="file-08", kind="file",
        prompt="Read the file notes.txt in the current working directory and "
               "write an uppercase version to upper.txt, then report the contents.",
        expected="HELLO AGENT / SECOND LINE",
        check=file_equals("upper.txt", "HELLO AGENT\nSECOND LINE",
                          need_answer=("HELLO AGENT",)),
        expected_files={"upper.txt": "HELLO AGENT\nSECOND LINE\n"},
        setup={"notes.txt": "Hello Agent\nSecond Line\n"},
        fake_code=('content = read_file("notes.txt")\n'
                   'write_file("upper.txt", content.upper())\n'
                   'final_answer("HELLO AGENT\\nSECOND LINE")'),
        fake_calls=[
            [{"name": "read_file", "arguments": {"path": "notes.txt"}}],
            [{"name": "write_file",
              "arguments": {"path": "upper.txt", "content": "HELLO AGENT\nSECOND LINE\n"}}],
            _final_answer_call("HELLO AGENT, SECOND LINE")],
    ),

    # ---------- research：网络研究（8 道；事实稳定，checker 只看事实） ----------
    EvalTask(
        id="research-01", kind="research",
        prompt="In what year was the Python programming language first released, "
               "and who created it? Give the answer with the source name.",
        expected="1991, Guido van Rossum",
        check=contains_all("1991", "guido"),
        fake_code='final_answer("Python was first released in 1991, created by Guido van Rossum (Wikipedia).")',
        fake_calls=[
            [{"name": "web_search", "arguments": {"query": "Python programming language first released year creator"}}],
            _final_answer_call("Python was first released in 1991, created by Guido van Rossum (Wikipedia).")],
    ),
    EvalTask(
        id="research-02", kind="research",
        prompt="What is the tallest mountain on Earth, and how high is it "
               "in metres? Give the answer with the source name.",
        expected="Mount Everest, 8848 metres",
        check=contains_all("everest", "8848"),
        fake_code='final_answer("Mount Everest is the tallest mountain on Earth, 8,848 metres high (Wikipedia).")',
        fake_calls=[
            [{"name": "web_search", "arguments": {"query": "tallest mountain on Earth height metres"}}],
            _final_answer_call("Mount Everest is the tallest mountain on Earth, 8,848 metres high (Wikipedia).")],
    ),
    EvalTask(
        id="research-03", kind="research",
        prompt="How many seconds would it take for a leopard at full speed to "
               "run through Pont des Arts? Give the number of seconds.",
        expected="about 9.62 seconds",
        check=contains_number(9.62, tol=0.15),
        fake_code='final_answer("About 9.62 seconds: the bridge is 155 metres long and a leopard runs at about 16.11 m/s.")',
        fake_calls=[
            [{"name": "web_search", "arguments": {"query": "Pont des Arts bridge length metres"}}],
            [{"name": "web_search", "arguments": {"query": "leopard top speed metres per second"}}],
            _final_answer_call("About 9.62 seconds: 155 metres at 16.11 m/s.")],
    ),
    EvalTask(
        id="research-04", kind="research",
        prompt="Who wrote the novel '1984', and in what year was it first "
               "published? Give the answer with the source name.",
        expected="George Orwell, 1949",
        check=contains_all("orwell", "1949"),
        fake_code='final_answer("George Orwell wrote 1984, first published in 1949 (Wikipedia).")',
        fake_calls=[
            [{"name": "web_search", "arguments": {"query": "novel 1984 author first published year"}}],
            _final_answer_call("George Orwell wrote 1984, first published in 1949 (Wikipedia).")],
    ),
    EvalTask(
        id="research-05", kind="research",
        prompt="What is the chemical symbol for gold, and what is its atomic "
               "number? Give the answer with the source name.",
        expected="Au, 79",
        check=contains_all("au", "79"),
        fake_code='final_answer("Gold has the chemical symbol Au and atomic number 79 (Wikipedia).")',
        fake_calls=[
            [{"name": "web_search", "arguments": {"query": "gold chemical symbol atomic number"}}],
            _final_answer_call("Gold has the chemical symbol Au and atomic number 79 (Wikipedia).")],
    ),
    EvalTask(
        id="research-06", kind="research",
        prompt="In what year did the first human land on the Moon, and who was "
               "the first person to step onto the lunar surface? "
               "Give the answer with the source name.",
        expected="1969, Neil Armstrong",
        check=contains_all("1969", "armstrong"),
        fake_code='final_answer("The first Moon landing was in 1969; Neil Armstrong was the first person to step onto the lunar surface (NASA).")',
        fake_calls=[
            [{"name": "web_search", "arguments": {"query": "first Moon landing year first person step surface"}}],
            _final_answer_call("The first Moon landing was in 1969; Neil Armstrong was the first person to step onto the lunar surface (NASA).")],
    ),
    EvalTask(
        id="research-07", kind="research",
        prompt="What is the capital of Australia, and in which year was the "
               "city founded? Give the answer with the source name.",
        expected="Canberra, 1913",
        check=contains_all("canberra", "1913"),
        fake_code='final_answer("The capital of Australia is Canberra, founded in 1913 (Wikipedia).")',
        fake_calls=[
            [{"name": "web_search", "arguments": {"query": "capital of Australia founded year"}}],
            _final_answer_call("The capital of Australia is Canberra, founded in 1913 (Wikipedia).")],
    ),
    EvalTask(
        id="research-08", kind="research",
        prompt="How many planets are in the Solar System, and which one is the "
               "largest? Give the answer with the source name.",
        expected="8 planets, Jupiter is the largest",
        check=contains_all("8", "jupiter"),
        fake_code='final_answer("There are 8 planets in the Solar System; Jupiter is the largest (NASA).")',
        fake_calls=[
            [{"name": "web_search", "arguments": {"query": "how many planets Solar System largest planet"}}],
            _final_answer_call("There are 8 planets in the Solar System; Jupiter is the largest (NASA).")],
    ),
]


def get_task(task_id: str) -> EvalTask:
    """按 id 取题；没有就报错（拼写错误要立刻发现，别悄悄跳过）。"""
    for task in TASKS:
        if task.id == task_id:
            return task
    raise KeyError(f"没有这道题：{task_id}")


def tasks_by_kind(kind: str) -> list[EvalTask]:
    return [t for t in TASKS if t.kind == kind]
