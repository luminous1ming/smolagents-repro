#!/usr/bin/env python3
"""Phase 5：多智能体研究流程 —— Anthropic orchestrator-worker 模式的复现。

对照的设计稿：Anthropic《How we built our multi-agent research system》
（2025-06-13）：lead agent 规划 → 并行 spawn 若干 researcher 子智能体
（各自独立的上下文和工具）→ 综合结果 → CitationAgent 把每条结论
映射回来源。演示场景是多源网络研究 + 引用（不做风控审核）。

自己的取舍（和 Phase 1–4 同一路）：
  - 子智能体就是 agents.py 的 MultiStepAgent，登记成 lead 的
    managed_agents（agents.py 的成果）；orchestrator 只负责
    “规划 → 并行派活 → 收集 → 综合 → 引用”这条流水线，不自己
    再写一套循环。
  - 并行是真的并行：ThreadPoolExecutor，每个 researcher 有自己的
    model（factory 每人发一个）、自己的工具箱、自己的记忆和日志——
    上下文隔离是结构保证的，不是约定。注意：共享同一个 FakeModel
    实例会让各 researcher 的脚本回复串台，测试里必须用 factory。
  - 综合用 lead 的模型调一次 generate（不是再跑一轮 lead agent 的
    循环——综合时不需要再派活，调一次就够，省调用也省步数）。
  - CitationAgent 是确定性的核查，不是模型调用：引用只能指向
    researcher 真正返回过的来源，编不出来（见下面的类 docstring）。
    这是对 Anthropic 做法的取舍：他们的 CitationAgent 也是一个
    独立的 pass，但确定性核查比“让模型再判断一次”更能保证
    “没有发明出来的来源”这条验收标准。
"""
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from agents import MultiStepAgent, ToolCallingAgent
from default_tools import default_tools
from models import ChatMessage, Model

# ---------- 提示词模板 ----------

# researcher 的派活任务：格式是死的——最后必须交 Sources: 节，
# orchestrator 就靠这一节抠来源；没这一节，引用就无从核查。
RESEARCHER_TASK_TEMPLATE = """Research this sub-question and report back:

{question}

Use web_search to find up-to-date information from multiple sources. Then return your findings as your final answer in this format:

- A short summary (2-4 sentences) of what you found.
- Then a section starting with exactly "Sources:" on its own line, with one source per line in the form "Title — URL". Only list sources you actually used; do not invent titles or URLs.
"""

PLAN_PROMPT_TEMPLATE = """Break the research topic below into {n} focused sub-questions that can be investigated independently. Cover different angles of the topic. Reply with only a numbered list, one sub-question per line, no extra text.

Topic: {topic}
"""

SYNTHESIS_PROMPT_TEMPLATE = """You are synthesising a research brief from reports written by researcher sub-agents. Write a concise brief (5-8 bullet points) answering the topic below. Each bullet must end with one or more citation markers like [S1], [S2] referring to the numbered source list. Use ONLY the sources in the list — do not invent sources, titles or URLs, and do not cite a source for a claim it does not support.

Topic: {topic}

Researcher reports:
{reports}

Numbered source list:
{sources}

Write the brief now (bullets only, no Sources section — that will be added afterwards)."""


# ---------- 数据模型 ----------


@dataclass
class Source:
    """一个来源：标题 + 链接（researcher 在 Sources: 节里交上来的）。"""

    title: str
    url: str

    def markdown(self, ref: str) -> str:
        return f"[{ref}] {self.title} — {self.url}"


@dataclass
class ResearchResult:
    """一个 researcher 交回来的东西：子问题 + 回答 + 它给的来源。"""

    question: str
    answer: str
    sources: list[Source] = field(default_factory=list)
    agent_name: str = ""


@dataclass
class ResearchBrief:
    """一次研究流程的完整产出：主题 + 子问题 + 各 researcher 的结果
    + 带引用的简报正文 + 被引用过的来源 + 被删掉的无效引用。"""

    topic: str
    sub_questions: list[str]
    results: list[ResearchResult]
    brief: str
    sources: list[Source] = field(default_factory=list)
    dropped_citations: list[str] = field(default_factory=list)

    def markdown(self) -> str:
        return f"# Research brief: {self.topic}\n\n{self.brief}"


# ---------- 子问题解析 ----------

_SUBQ_RE = re.compile(r"^\s*(?:\d+[.)、）]|\-|\*|•)\s*(.+?)\s*$")


def parse_subquestions(text: str) -> list[str]:
    """解析规划输出：认编号/项目符号行；一行都认不出时，
    把非空行当问题（模型只回了一行散文时的兜底）。"""
    questions = []
    for line in text.splitlines():
        m = _SUBQ_RE.match(line)
        if m and m.group(1).strip():
            questions.append(m.group(1).strip())
    if not questions:
        questions = [line.strip() for line in text.splitlines() if line.strip()]
    return questions


def _dedupe_sources(sources: list[Source]) -> list[Source]:
    """按 URL 去重，顺序保留（两个 researcher 交了同一个来源时只留一份）。"""
    seen: set[str] = set()
    out: list[Source] = []
    for s in sources:
        if s.url not in seen:
            seen.add(s.url)
            out.append(s)
    return out


# ---------- CitationAgent：确定性引用核查 ----------


class CitationAgent:
    """引用核查（Anthropic 的 CitationAgent 的确定性版）。

    两件事，都是纯字符串操作，不调模型：
      1. extract_sources(text)：从 researcher 的回答里抠来源。只认
         "Sources:" 节里的行（"标题 — 链接" 或裸链接）；没有 Sources:
         节时，全文扫 URL 兜底。按 URL 去重，顺序保留。
      2. cite(draft, sources)：把综合稿里的 [S1]、[S2]……逐个核对：
         编号在来源列表里的保留，不在的删掉并记进 dropped；
         稿子自己带的 Sources 节直接扔掉，最后只列出正文里真正
         引用到的来源（没被引用的不进 Sources 节）。
         结论：最终简报里的每一条引用都一定指向 researcher 交过的
         来源；模型编出来的链接、编出来的编号，进不了正文。
    """

    _URL_RE = re.compile(r"https?://[^\s)>\]]+")
    _SOURCE_LINE_RE = re.compile(
        r"^\s*(?:[-*•\d.)]+\s*)?(.*?)\s*[—–-]\s*(https?://\S+)\s*$")
    _MARKER_RE = re.compile(r"\[S(\d+)\]")
    _SOURCES_HEADING_RE = re.compile(r"(?im)^#{1,3}\s*sources?\s*[:：]?\s*$")
    _SOURCES_LABEL_RE = re.compile(r"(?im)^\s*sources?\s*[:：]\s*$")

    @classmethod
    def extract_sources(cls, text: str) -> list[Source]:
        section = text
        m = cls._SOURCES_LABEL_RE.search(text)
        if m:
            section = text[m.end():]
        sources: list[Source] = []
        seen: set[str] = set()
        for line in section.splitlines():
            line = line.strip()
            if not line:
                continue
            sm = cls._SOURCE_LINE_RE.match(line)
            if sm:
                title = sm.group(1).strip()
                url = cls._clean_url(sm.group(2))
                title = re.sub(r"^[\[\"'`]+|[\]\"'`]+$", "", title).strip() or url
                if url not in seen:
                    seen.add(url)
                    sources.append(Source(title=title, url=url))
            else:
                # 一行里可能有多个 URL（兜底分支）：全抠出来，标题取
                # 上一个 URL 之后、本 URL 之前的文字；没有就用 URL 本身。
                prev_end = 0
                for um in cls._URL_RE.finditer(line):
                    url = cls._clean_url(um.group(0))
                    title = line[prev_end:um.start()].strip(" -—–:") or url
                    prev_end = um.end()
                    if url not in seen:
                        seen.add(url)
                        sources.append(Source(title=title, url=url))
        return sources

    @staticmethod
    def _clean_url(url: str) -> str:
        """链接末尾的标点（句号、逗号、中文句号）不是链接的一部分。"""
        return url.rstrip(".,。;!?")

    @classmethod
    def cite(cls, draft: str, sources: list[Source]) -> tuple[str, list[str]]:
        """核查并定稿：返回 (最终简报, 被删掉的无效引用编号)。

        步骤：先扔掉稿子自己带的 Sources 节（里面的链接没核查过，
        不能信）；再逐个核对 [S#] 标记；最后只列正文引用到的来源。
        """
        head = cls._SOURCES_HEADING_RE.split(draft, maxsplit=1)[0].rstrip()
        dropped: list[str] = []

        def _keep(m):
            idx = int(m.group(1))
            if 1 <= idx <= len(sources):
                return m.group(0)
            if f"S{idx}" not in dropped:
                dropped.append(f"S{idx}")
            return ""

        cleaned = cls._MARKER_RE.sub(_keep, head)
        cited = sorted({int(n) for n in cls._MARKER_RE.findall(cleaned)})
        lines = [cleaned.rstrip()]
        if cited:
            lines.append("")
            lines.append("## Sources")
            for i in cited:
                lines.append(sources[i - 1].markdown(f"S{i}"))
        return "\n".join(lines), dropped


# ---------- Orchestrator ----------


class ResearchOrchestrator:
    """orchestrator-worker 研究流程（Anthropic 模式的复现）。

    一次 research() 走五步：
      1. plan：调一次 lead 模型，把主题拆成子问题；
      2. spawn：给每个子问题建一个 researcher（lead 的 managed agent），
         ThreadPoolExecutor 真并行跑；
      3. collect：收各 researcher 的报告，抠出它们交的来源；
      4. synthesise：调一次 lead 模型，把报告综合成带 [S#] 标记的简报草稿；
      5. cite：CitationAgent 确定性核查，产出最终简报。

    research() 之后，self.lead 和 self.researchers 留着给调用方检查
    （测试用它们断言上下文隔离；线上调试也能看每一步的记忆）。
    researcher 出错（模型抛异常）时不炸掉整轮：那一路交一份
    “失败说明”，其他路照常综合——research() 的结果里能看到它。
    """

    def __init__(self, model: Model, researcher_model_factory=None,
                 tools_factory=None, max_researchers: int = 5,
                 base_dir: str | None = None,
                 researcher_max_steps: int = 8):
        self.model = model
        self.researcher_model_factory = researcher_model_factory or (lambda: model)
        self.tools_factory = tools_factory or (lambda: default_tools(base_dir=base_dir))
        self.max_researchers = max_researchers
        self.researcher_max_steps = researcher_max_steps
        self.lead: MultiStepAgent | None = None
        self.researchers: list[MultiStepAgent] = []

    def plan(self, topic: str, n_subquestions: int = 3) -> list[str]:
        """一步规划：调一次 lead 模型，把主题拆成 n 个子问题。"""
        prompt = PLAN_PROMPT_TEMPLATE.format(n=n_subquestions, topic=topic)
        msg = self.model.generate([ChatMessage(role="user", content=prompt)])
        return parse_subquestions(msg.content or "")

    def research(self, topic: str, n_subquestions: int = 3) -> ResearchBrief:
        sub_questions = self.plan(topic, n_subquestions)[:self.max_researchers]
        self.researchers = [self._make_researcher(i, q)
                            for i, q in enumerate(sub_questions)]
        # lead 拥有这些 researcher（managed_agents）；派活时直接调
        # lead 的 managed agent（即调工具），不走 lead 自己的循环。
        self.lead = ToolCallingAgent(
            tools=self.tools_factory(), model=self.model,
            name="lead", description="Lead research agent",
            managed_agents=self.researchers)
        tasks = [RESEARCHER_TASK_TEMPLATE.format(question=q)
                 for q in sub_questions]
        with ThreadPoolExecutor(max_workers=max(len(self.researchers), 1)) as pool:
            reports = list(pool.map(self._run_researcher,
                                    [r.name for r in self.researchers], tasks))
        results = []
        for q, r, report in zip(sub_questions, self.researchers, reports):
            results.append(ResearchResult(
                question=q, answer=report,
                sources=CitationAgent.extract_sources(report),
                agent_name=r.name))
        all_sources = _dedupe_sources([s for r in results for s in r.sources])
        draft = self._synthesize(topic, results, all_sources)
        brief_text, dropped = CitationAgent.cite(draft, all_sources)
        return ResearchBrief(topic=topic, sub_questions=sub_questions,
                             results=results, brief=brief_text,
                             sources=all_sources, dropped_citations=dropped)

    def _make_researcher(self, index: int, question: str) -> ToolCallingAgent:
        """建一个 researcher：自己的模型（factory 发）、自己的工具箱、
        自己的记忆——上下文隔离是结构保证的。"""
        return ToolCallingAgent(
            tools=self.tools_factory(), model=self.researcher_model_factory(),
            max_steps=self.researcher_max_steps,
            name=f"researcher_{index + 1}",
            description=("Researches one sub-question with web search "
                         "and returns findings with sources."))

    def _run_researcher(self, name: str, task: str) -> str:
        """派活：调 lead 的 managed agent（即调一次工具）。
        出错不炸整轮，交一份失败说明，其他路照常综合。"""
        try:
            return self.lead.managed_agents[name](task)
        except Exception as e:  # noqa: BLE001 - 子智能体失败要留痕、不能炸整轮
            return (f"Researcher {name} failed before returning a report: "
                    f"{type(e).__name__}: {e}")

    def _synthesize(self, topic: str, results: list[ResearchResult],
                    all_sources: list[Source]) -> str:
        """综合：调一次 lead 模型，把各 researcher 的报告写成带 [S#] 的简报草稿。"""
        reports = "\n\n".join(
            f"--- Report from {r.agent_name} (sub-question: {r.question}) ---\n{r.answer}"
            for r in results)
        numbered = "\n".join(f"[S{i}] {s.title} — {s.url}"
                             for i, s in enumerate(all_sources, 1))
        prompt = SYNTHESIS_PROMPT_TEMPLATE.format(topic=topic, reports=reports,
                                                  sources=numbered)
        msg = self.model.generate([ChatMessage(role="user", content=prompt)])
        return msg.content or ""
