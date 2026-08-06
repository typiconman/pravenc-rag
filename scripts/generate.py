"""Generation via a hosted OpenRouter model (or local Ollama), through CitationQueryEngine.

Beyond plain RAG this module does three things:

* **Language control.** Answer language is per-query (auto-detected or forced).
  Russian article titles stay Cyrillic in citations even in English answers —
  the Russian title is the citation anchor and must stay verifiable.

* **Citation grounding.** ``CitationQueryEngine`` numbers the retrieved sources
  and asks for ``[N]`` markers.

* **Citation verification.** Every bracketed marker the model emits is checked
  against the sources actually retrieved. Anything that is not a valid in-range
  ``[N]`` — an out-of-range number, ``[New Source]``, ``[Author, pages]``,
  ``[4, 5]`` — is stripped and reported. Smaller/looser models fabricate exactly
  these; stripping them is what keeps "always cites" honest.

Provider and model are set in config; ``ask(..., model=...)`` overrides the
model per call so the UI and ``compare`` can pit models against each other.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from llama_index.core import PromptTemplate
from llama_index.core.query_engine import CitationQueryEngine

from .config import Config
from .retrieve import HybridRetriever

_CYRILLIC = re.compile(r"[\u0400-\u04FF]")
# Any bracketed token, so we can judge each one — not just well-formed [N].
_CITE_ANY = re.compile(r"\[([^\[\]]+)\]")

LANG_NAMES = {"ru": "Russian", "en": "English"}


def detect_language(text: str) -> str:
    """Answer in the language the question was asked in.

    Script ratio decides when clear-cut; in the mixed middle (an English
    question quoting Cyrillic titles) the opening word decides. The UI keeps an
    explicit ru/en override for genuinely ambiguous cases.
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
        "- Cite ONLY by bracketed source number. Never write author names, "
        "titles, years, or the word 'Source' inside the brackets, and never "
        "cite a number not shown above. Uncitable claims must be omitted.\n"
        "- Keep proper names, article titles and technical terms in their "
        "original Cyrillic or Greek form; you may add a transliteration in "
        "parentheses on first mention.\n"
        "- Be concise: answer the question from the sources, without padding.\n\n"
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
        "and adding citations for new material. Cite ONLY by bracketed source "
        "number shown above — never author/title/year, never a number not "
        "listed. If the new sources add nothing, repeat the existing answer.\n"
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
    model: str = ""
    dropped_citations: list[str] = field(default_factory=list)  # fabricated/invalid markers
    uncited: bool = False
    timing: dict = field(default_factory=dict)

    @property
    def suspect(self) -> bool:
        """Many fabricated markers => the model is confabulating; distrust it."""
        return len(self.dropped_citations) >= 3 or self.uncited


def _verify_citations(text: str, nodes) -> tuple[str, list[int], list[str]]:
    """Keep valid ``[N]`` citations; strip fabrications.

    Accepts the canonical ``[N]`` and also recovers comma-grouped citations the
    model sometimes emits against instructions — ``[3, 4]`` becomes ``[3][4]``
    when both numbers are in range. Within a group, out-of-range numbers are
    dropped while valid ones are kept (``[7, 8]`` with 7 sources -> ``[7]``).
    Anything not a bare number or comma-list of numbers — ``New Source``,
    ``Arrance, 130-131``, ``Source 9`` — is stripped entirely.

    Returns (clean_text, used_source_numbers, dropped_markers). Dropped markers
    are strings so callers can show exactly what was rejected.

    Note: this strips ANY non-conforming bracketed text, so a legitimate
    non-citation like "[sic]" would go too — acceptable here, where brackets in
    an answer are effectively always citations, and leaving fabricated ones is
    the far worse failure.
    """
    n_sources = len(nodes)
    used: list[int] = []
    dropped: list[str] = []

    def repl(m: re.Match) -> str:
        body = m.group(1).strip()
        parts = [p.strip() for p in body.split(",")]
        # A citation group only if every part is a bare integer (e.g. "3" or "3, 4").
        if parts and all(p.isdigit() for p in parts):
            kept: list[int] = []
            for p in parts:
                i = int(p)
                if 1 <= i <= n_sources:
                    if i not in used:
                        used.append(i)
                    kept.append(i)
                else:
                    dropped.append(p)          # out-of-range invented number
            return "".join(f"[{i}]" for i in kept)
        # Non-numeric bracket content: author-year, "New Source", "Source 9", etc.
        dropped.append(body)
        return ""

    clean = _CITE_ANY.sub(repl, text)
    clean = re.sub(r"[ \t]{2,}", " ", clean)
    clean = re.sub(r"\s+([.,;:])", r"\1", clean)
    return clean.strip(), used, dropped


class Assistant:
    """Query -> hybrid retrieval -> rerank -> hosted LLM -> verified citations."""

    def __init__(self, cfg: Config, retriever: HybridRetriever | None = None):
        self.cfg = cfg
        # The response synthesizer builds its own PromptHelper from global
        # Settings, NOT from the LLM we pass — its default context window is a
        # tiny 3900 tokens, which makes CitationQueryEngine's refine loop
        # miscompute and fail ("available context size ... not non-negative")
        # whenever the packed sources approach that default. Point Settings at
        # the real window so the synthesizer budgets correctly.
        from llama_index.core import Settings

        self._max_output = min(4096, max(512, cfg.llm.num_ctx // 4))
        Settings.context_window = cfg.llm.num_ctx
        Settings.num_output = self._max_output

        self.retriever = retriever or self._build_retriever()
        self._engines: dict[tuple[str, str], CitationQueryEngine] = {}

    def _build_retriever(self) -> HybridRetriever:
        from qdrant_client import QdrantClient

        from .embed import Embedder

        return HybridRetriever(
            self.cfg,
            embedder=Embedder(self.cfg.embed_model),
            client=QdrantClient(url=self.cfg.qdrant_url, timeout=self.cfg.qdrant_timeout),
        )

    def _build_llm(self, model: str):
        """Construct the LLM for a given model string, honoring cfg.llm.provider."""
        c = self.cfg.llm
        if c.provider == "ollama":
            from llama_index.llms.ollama import Ollama

            return Ollama(
                model=model,
                base_url=c.ollama_url,
                request_timeout=c.request_timeout,
                context_window=c.num_ctx,
                temperature=c.temperature,
                additional_kwargs={"num_ctx": c.num_ctx},
            )

        # OpenRouter / any OpenAI-compatible host
        from llama_index.llms.openai_like import OpenAILike

        api_key = os.environ.get(c.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Environment variable {c.api_key_env} is not set — export your "
                f"{c.provider} API key before querying."
            )
        return OpenAILike(
            model=model,
            api_base=c.api_base,
            api_key=api_key,
            is_chat_model=True,
            context_window=c.num_ctx,
            max_tokens=self._max_output,
            temperature=c.temperature,
            timeout=c.request_timeout,
            max_retries=2,
        )

    def _engine(self, lang: str, model: str) -> CitationQueryEngine:
        key = (lang, model)
        if key not in self._engines:
            self._engines[key] = CitationQueryEngine.from_args(
                index=None,
                retriever=self.retriever,
                llm=self._build_llm(model),
                citation_chunk_size=self.cfg.llm.citation_chunk_size,
                citation_qa_template=_qa_template(lang),
                citation_refine_template=_refine_template(lang),
            )
        return self._engines[key]

    def retrieve(self, question: str, language: str = "auto"):
        """Run retrieval ONCE (embed + hybrid search + rerank + citation split).

        Returns (lang, query_bundle, citation_nodes). Retrieval is
        model-independent, so ``compare`` calls this once and reuses the nodes
        across every model — the expensive rerank pass no longer repeats per
        model.
        """
        from llama_index.core import QueryBundle

        lang = detect_language(question) if language == "auto" else language
        qb = QueryBundle(question)
        # any engine retrieves identically; use the default model's.
        nodes = self._engine(lang, self.cfg.llm.model).retrieve(qb)
        return lang, qb, nodes

    def generate(self, qb, nodes, lang: str, model: str) -> Answer:
        """Generate + verify an answer from already-retrieved nodes (no re-retrieval)."""
        response = self._engine(lang, model).synthesize(qb, nodes)
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
            model=model,
            dropped_citations=dropped,
            uncited=not used and bool(nodes),
            timing={"embed": t.embed, "search": t.search, "rerank": t.rerank},
        )

    def ask(self, question: str, language: str = "auto", model: str | None = None) -> Answer:
        model = model or self.cfg.llm.model
        lang, qb, nodes = self.retrieve(question, language)
        return self.generate(qb, nodes, lang, model)
