"""Generation with Gemma 3 via Ollama, wired through CitationQueryEngine.

Three things happen here beyond plain RAG:

* **Language control.** The answer language is chosen per query (auto-detected
  from the question's script, or forced). Russian article titles stay in
  Cyrillic in citations even when the prose is English — the Russian title is
  the citation anchor and must remain verifiable against the printed edition.

* **Citation grounding.** ``CitationQueryEngine`` numbers the sources it feeds
  the model and asks for ``[N]`` markers.

* **Citation verification.** The model's ``[N]`` markers are checked against
  the sources actually retrieved; out-of-range (invented) markers are stripped
  before the answer is shown. This is what makes "always cites" true rather
  than "usually cites".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from llama_index.core import PromptTemplate
from llama_index.core.query_engine import CitationQueryEngine

from .config import Config
from .retrieve import HybridRetriever

_CYRILLIC = re.compile(r"[\u0400-\u04FF]")
_CITE = re.compile(r"\[(\d+)\]")

LANG_NAMES = {"ru": "Russian", "en": "English"}


def detect_language(text: str) -> str:
    """Answer in the language the question was asked in.

    Script ratio decides when it is clear-cut. Queries here are often mixed —
    an English question quoting Cyrillic article titles or section headings —
    so in the ambiguous middle the opening word decides, since that is what
    sets a question's language. Genuinely ambiguous cases are why the UI keeps
    an explicit ru/en override.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "en"
    ratio = sum(1 for c in letters if _CYRILLIC.match(c)) / len(letters)
    if ratio >= 0.8:
        return "ru"
    if ratio <= 0.2:
        return "en"
    first = re.search(r"[^\W\d_]+", text, re.UNICODE)
    return "ru" if first and _CYRILLIC.match(first.group(0)[0]) else "en"


def _qa_template(lang: str) -> PromptTemplate:
    name = LANG_NAMES[lang]
    return PromptTemplate(
        "You are a research assistant for the Russian Orthodox Encyclopedia "
        "(Православная энциклопедия). Below are numbered sources, each starting "
        "with 'Source N:'. The sources are in Russian.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Instructions:\n"
        f"- Write the answer in {name}, regardless of the language of the sources.\n"
        "- Use ONLY the information in the sources above. If they do not answer "
        "the question, say so plainly rather than drawing on outside knowledge.\n"
        "- Cite every factual claim with the source number in square brackets, "
        "e.g. [1] or [2][3]. Every paragraph must carry at least one citation.\n"
        "- Keep proper names, article titles and technical terms in their "
        "original Cyrillic or Greek form; you may add a transliteration in "
        "parentheses on first mention.\n"
        "- Do not invent source numbers. Only cite sources shown above.\n\n"
        "Question: {query_str}\n"
        "Answer: "
    )


def _refine_template(lang: str) -> PromptTemplate:
    name = LANG_NAMES[lang]
    return PromptTemplate(
        "You are refining an existing answer using additional numbered sources.\n"
        "Question: {query_str}\n"
        "Existing answer: {existing_answer}\n"
        "New sources below:\n"
        "------------\n"
        "{context_msg}\n"
        "------------\n"
        f"Rewrite the answer in {name}, keeping every citation marker accurate "
        "and adding citations for any new material. If the new sources add "
        "nothing, repeat the existing answer unchanged.\n"
        "Answer: "
    )


@dataclass
class Source:
    n: int
    title: str
    heading: str | None
    volume: object
    pages: str | None
    url: str
    citation: str
    score: float | None = None


@dataclass
class Answer:
    text: str
    sources: list[Source] = field(default_factory=list)
    language: str = "en"
    dropped_citations: list[int] = field(default_factory=list)
    uncited: bool = False
    timing: dict = field(default_factory=dict)


def _verify_citations(text: str, nodes) -> tuple[str, list[int], list[int]]:
    """Strip invented [N] markers; return (clean_text, used, dropped)."""
    n_sources = len(nodes)
    used: list[int] = []
    dropped: list[int] = []

    def repl(m: re.Match) -> str:
        i = int(m.group(1))
        if 1 <= i <= n_sources:
            if i not in used:
                used.append(i)
            return m.group(0)
        dropped.append(i)
        return ""

    clean = _CITE.sub(repl, text)
    clean = re.sub(r"[ \t]{2,}", " ", clean)
    clean = re.sub(r"\s+([.,;:])", r"\1", clean)
    return clean.strip(), used, dropped


class Assistant:
    """Query -> hybrid retrieval -> rerank -> Gemma 3 -> verified citations."""

    def __init__(self, cfg: Config, retriever: HybridRetriever | None = None):
        self.cfg = cfg
        self.retriever = retriever or self._build_retriever()
        self._llm = None
        self._engines: dict[str, CitationQueryEngine] = {}

    def _build_retriever(self) -> HybridRetriever:
        from qdrant_client import QdrantClient

        from .embed import Embedder

        return HybridRetriever(
            self.cfg,
            embedder=Embedder(self.cfg.embed_model),
            client=QdrantClient(url=self.cfg.qdrant_url),
        )

    @property
    def llm(self):
        if self._llm is None:
            from llama_index.llms.ollama import Ollama

            c = self.cfg.llm
            self._llm = Ollama(
                model=c.model,
                base_url=c.ollama_url,
                request_timeout=c.request_timeout,
                context_window=c.num_ctx,
                temperature=c.temperature,
                # num_ctx must be passed through to Ollama explicitly: its own
                # default is far smaller than 128K and will silently truncate
                # the retrieved sections, which reads as the model "ignoring"
                # context.
                additional_kwargs={"num_ctx": c.num_ctx},
            )
        return self._llm

    def _engine(self, lang: str) -> CitationQueryEngine:
        if lang not in self._engines:
            self._engines[lang] = CitationQueryEngine.from_args(
                index=None,
                retriever=self.retriever,
                llm=self.llm,
                citation_chunk_size=self.cfg.llm.citation_chunk_size,
                citation_qa_template=_qa_template(lang),
                citation_refine_template=_refine_template(lang),
            )
        return self._engines[lang]

    def ask(self, question: str, language: str = "auto") -> Answer:
        lang = detect_language(question) if language == "auto" else language
        response = self._engine(lang).query(question)

        nodes = response.source_nodes
        clean, used, dropped = _verify_citations(str(response), nodes)

        sources: list[Source] = []
        for i in used:
            md = nodes[i - 1].node.metadata
            sources.append(
                Source(
                    n=i,
                    title=md.get("article_title") or "",
                    heading=md.get("heading"),
                    volume=md.get("volume"),
                    pages=md.get("page_numbers"),
                    url=md.get("source_url") or "",
                    citation=md.get("citation") or "",
                    score=nodes[i - 1].score,
                )
            )

        t = self.retriever.last_timing
        return Answer(
            text=clean,
            sources=sources,
            language=lang,
            dropped_citations=dropped,
            uncited=not used and bool(nodes),
            timing={"embed": t.embed, "search": t.search, "rerank": t.rerank},
        )
