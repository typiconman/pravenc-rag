"""Query-stage helper: rebuild a chunk's parent section text from the source.

Because the index no longer stores ``parent_text`` on every chunk (it would
duplicate section text across all of a section's chunks), the query stage
reconstructs it on demand from the corpus checkout — which, since you maintain
pravenc-md, is always present on the box. Re-parsing is cached per document, so
a query touching several chunks of one article parses it once.

Usage (query stage), given a Qdrant hit's payload:

    hydrator = ParentHydrator(cfg.corpus_dir, cfg.section_types)
    context = hydrator.hydrate(hit.payload)   # full section text for the LLM
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .chunk import expandable, headword_lemma, siglum_pattern
from .ingest import parse_article


class ParentHydrator:
    def __init__(self, corpus_dir, section_types: dict[str, str] | None = None):
        self.corpus_dir = Path(corpus_dir)
        # section_types must match what was used at index time so section
        # indices line up; text and ordering are independent of it regardless.
        self._section_types = section_types

    @lru_cache(maxsize=1024)
    def _document(self, doc_id: str):
        return parse_article(self.corpus_dir / f"{doc_id}.md", self._section_types)

    def parent_text(self, doc_id: str, section_idx: int) -> str:
        doc = self._document(doc_id)
        if not (0 <= section_idx < len(doc.sections)):
            return ""
        section = doc.sections[section_idx]
        text = section.text
        # Expand the headword siglum so the LLM reads the term in full, matching
        # what the embedder and reranker see (КОНДАК body: "К." -> "кондак").
        # Bibliographic sections are skipped, same rule as indexing; grammatical
        # case is irrelevant to the model.
        pattern = siglum_pattern(doc.title)
        if pattern is not None and expandable(section):
            text = pattern.sub(headword_lemma(doc.title), text)
        return text

    def hydrate(self, payload: dict) -> str:
        """Return the full parent-section text for a retrieved chunk payload."""
        return self.parent_text(payload["doc_id"], int(payload["section_idx"]))
