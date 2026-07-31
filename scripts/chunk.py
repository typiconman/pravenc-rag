"""Section-aware, parent-child chunking.

Each section is the parent. Short sections become a single chunk; long ones are
packed paragraph-by-paragraph up to ``max_tokens`` (oversized paragraphs are
token-windowed with overlap). Every child chunk carries the full section text
as ``parent_text`` so generation sees coherent context, and a citation payload
built from the Russian-canonical metadata.
"""
from __future__ import annotations

from .models import Chunk, Document, Section


def _format_citation(doc: Document) -> str:
    parts = [doc.title, "// Православная энциклопедия"]
    if doc.volume is not None:
        parts.append(f"Т. {doc.volume}")
    if doc.page_numbers:
        parts.append(f"С. {doc.page_numbers}")
    tail = ". ".join(parts[1:])
    cite = f"{doc.title} {('// ' + tail) if tail else ''}".strip()
    if doc.source_url:
        cite = f"{cite} — {doc.source_url}"
    return cite


class Chunker:
    def __init__(self, model_name: str, max_tokens: int = 512, overlap_tokens: int = 64):
        from transformers import AutoTokenizer  # lazy: heavy import

        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.max_tokens = max_tokens
        self.overlap = overlap_tokens

    def _ntok(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))

    def _window(self, text: str) -> list[str]:
        ids = self.tok.encode(text, add_special_tokens=False)
        step = max(1, self.max_tokens - self.overlap)
        out = []
        for start in range(0, len(ids), step):
            out.append(self.tok.decode(ids[start : start + self.max_tokens]))
            if start + self.max_tokens >= len(ids):
                break
        return out

    def _split(self, text: str) -> list[str]:
        paras = [p.strip() for p in text.split("\n") if p.strip()]
        chunks: list[str] = []
        cur: list[str] = []
        cur_len = 0
        for p in paras:
            n = self._ntok(p)
            if n > self.max_tokens:
                if cur:
                    chunks.append(" ".join(cur))
                    cur, cur_len = [], 0
                chunks.extend(self._window(p))
                continue
            if cur and cur_len + n > self.max_tokens:
                chunks.append(" ".join(cur))
                cur, cur_len = [], 0
            cur.append(p)
            cur_len += n
        if cur:
            chunks.append(" ".join(cur))
        return chunks or [text]

    def _payload(self, doc: Document, section: Section, text: str, section_idx: int) -> dict:
        # Note: parent/section text is deliberately NOT stored here. It is
        # hydrated from the source file at query time (see hydrate.ParentHydrator),
        # Qdrant holds only vectors, citation metadata, and the child text.
        return {
            "doc_id": doc.id,
            "section_idx": section_idx,
            "article_title": doc.title,
            "authors": doc.authors,
            "section_author": section.author,
            "volume": doc.volume,
            "page_numbers": doc.page_numbers,
            "source_url": doc.source_url,
            "section_type": section.type,
            "heading": section.heading,
            "text": text,
            "citation": _format_citation(doc),
        }

    def chunk_document(self, doc: Document) -> list[Chunk]:
        out: list[Chunk] = []
        for s_idx, section in enumerate(doc.sections):
            if not section.text.strip():
                continue
            for c_idx, piece in enumerate(self._split(section.text)):
                out.append(
                    Chunk(
                        id=f"{doc.id}:{s_idx}:{c_idx}",
                        doc_id=doc.id,
                        text=piece,
                        parent_text=section.text,
                        section_type=section.type,
                        heading=section.heading,
                        payload=self._payload(doc, section, piece, s_idx),
                    )
                )
        return out
