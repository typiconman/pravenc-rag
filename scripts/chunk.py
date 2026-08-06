"""Section-aware, parent-child chunking, with retrieval-only text augmentation.

Each section is the parent. Short sections become a single chunk; long ones are
packed paragraph-by-paragraph up to ``max_tokens`` (oversized paragraphs are
token-windowed with overlap). Generation context is hydrated from the corpus at
query time, so the index stores only the child text plus citation metadata.

Two augmentations make the headword findable. The encyclopedia abbreviates each
article's headword to a siglum inside its own article ("КОНДАК" -> "К."), so the
defining term is nearly absent from the text that defines it — which is what
starves the sparse/lexical channel and hurts retrieval:

1. **Contextual prefix** — the article title (and section heading) are prepended
   to each chunk, so every chunk carries its headword.
2. **Siglum expansion** — the article's own siglum is replaced by the headword
   lemma. Case is deliberately ignored: retrieval needs the term *present*, not
   grammatical, which sidesteps Russian morphology entirely.

Both apply ONLY to ``Chunk.text`` (the string that gets embedded).
``payload["text"]`` keeps the original wording, so anything shown to a user or
scored by the reranker sees the encyclopedia as written.

Reference sections (Источники, Сочинения, Литература) are skipped for siglum
expansion: they are bibliographies that spell names out rather than using the
headword siglum, so expanding there would corrupt citations with false matches.
"""
from __future__ import annotations

import re

from .models import Chunk, Document, Section

CYR = "А-Яа-яЁё"

# Section types / headings whose bibliographic content must not be siglum-expanded.
NO_EXPAND_TYPES = {"sources", "literature", "works"}
NO_EXPAND_HEADING = re.compile(
    r"^\s*(источник|сочинен|литератур|библиогр|издани|публикац)", re.IGNORECASE
)

# Headword words that contribute no initial to a multi-word siglum.
SIGLUM_SKIP = {"и", "во", "в", "на", "с", "со", "от", "из", "к", "о", "об",
               "для", "при", "по", "за", "над", "под", "у", "не", "ни"}

_TITLE_TOKENS = re.compile(f"[{CYR}]+")


def _format_citation(doc: Document) -> str:
    """Russian bibliographic form: 'Title // Православная энциклопедия. Т. N. С. X-Y — URL'."""
    tail = ["Православная энциклопедия"]
    if doc.volume is not None:
        tail.append(f"Т. {doc.volume}")
    if doc.page_numbers:
        tail.append(f"С. {doc.page_numbers}")
    cite = f"{doc.title} // " + ". ".join(tail)
    if doc.source_url:
        cite = f"{cite} — {doc.source_url}"
    return cite


def siglum_pattern(title: str) -> re.Pattern | None:
    """Regex matching this article's own headword siglum in its body.

    'КОНДАК' -> matches a standalone 'К.'; 'АЛЕКСИЙ, ЧЕЛОВЕК БОЖИЙ' -> matches
    the dotted sequence 'А. ч. Б.'. Returns None if no siglum can be derived.

    Casing is written into the pattern per letter rather than using
    re.IGNORECASE, because the personal-initial guard below must stay
    case-sensitive: initials are uppercase ("С. А. Серафимову") while ordinary
    abbreviations are lowercase ("в 1890 г. А."), and an IGNORECASE guard would
    mistake the latter for the former and refuse to expand.
    """
    letters = [w[0].upper() for w in _TITLE_TOKENS.findall(title)
               if w.lower() not in SIGLUM_SKIP]
    if not letters:
        return None
    if len(letters) == 1:
        # Standalone uppercase letter + dot, not glued to other Cyrillic, and
        # not part of a personal-initial run ("С. А.", "К. В.") on either side.
        pat = (rf"(?<![{CYR}])(?<![А-ЯЁ]\.)(?<![А-ЯЁ]\.\s)"
               rf"{re.escape(letters[0])}\.(?!\s*[А-ЯЁ]\.)(?![{CYR}])")
    else:
        # Multi-word sigla mix case ("А. ч. Б.", "Л. а. П."), so accept either
        # for each letter explicitly.
        parts = [f"[{l}{l.lower()}]" for l in letters]
        pat = rf"(?<![{CYR}])" + r"\.\s*".join(parts) + r"\."
    return re.compile(pat)


def headword_lemma(title: str) -> str:
    """Embeddable full form of the headword: 'КОНДАК' -> 'кондак'.

    Nominative and lowercased on purpose — the goal is lexical presence for
    retrieval, not grammatical agreement with the surrounding sentence.
    """
    return re.sub(r"\s*,\s*", " ", title.strip()).lower()


def expandable(section: Section) -> bool:
    """False for bibliographic sections, which don't use the headword siglum."""
    if section.type in NO_EXPAND_TYPES:
        return False
    if section.heading and NO_EXPAND_HEADING.match(section.heading):
        return False
    return True


class Chunker:
    def __init__(
        self,
        model_name: str,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
        prefix_context: bool = True,
        expand_sigla: bool = True,
    ):
        from transformers import AutoTokenizer  # lazy: heavy import

        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.max_tokens = max_tokens
        self.overlap = overlap_tokens
        self.prefix_context = prefix_context
        self.expand_sigla = expand_sigla

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

    # --- retrieval-only augmentation ---------------------------------------
    def _expand(self, piece: str, section: Section,
                pattern: re.Pattern | None, lemma: str) -> str:
        """Siglum-expanded text; bibliographic sections are left untouched.

        This is the string stored in the payload (what the reranker scores) and,
        with a contextual header added, what gets embedded. The LLM sees the
        same expansion applied to the parent section by hydrate.ParentHydrator —
        so embedding, reranking and generation all read the headword in full,
        never the bare siglum. Grammatical case is irrelevant to all three.
        """
        if self.expand_sigla and pattern is not None and expandable(section):
            return pattern.sub(lemma, piece)
        return piece

    def _prefix(self, text: str, doc: Document, section: Section) -> str:
        """Prepend 'Title. Heading' — a contextual header, for the embedding only."""
        if not self.prefix_context:
            return text
        header = doc.title if not section.heading else f"{doc.title}. {section.heading}"
        return f"{header}\n\n{text}"

    def _payload(self, doc: Document, section: Section, text: str, section_idx: int) -> dict:
        # Parent/section text is hydrated from the source at query time
        # (hydrate.ParentHydrator), so Qdrant holds only vectors, citation
        # metadata, and this child text. "text" is the SIGLUM-EXPANDED child —
        # what the reranker scores. Citations use the separate metadata fields
        # (article_title, volume, page_numbers, source_url), which stay original,
        # so nothing user-facing is affected by the expansion.
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
        pattern = siglum_pattern(doc.title) if self.expand_sigla else None
        lemma = headword_lemma(doc.title)

        out: list[Chunk] = []
        for s_idx, section in enumerate(doc.sections):
            if not section.text.strip():
                continue
            for c_idx, piece in enumerate(self._split(section.text)):
                expanded = self._expand(piece, section, pattern, lemma)
                out.append(
                    Chunk(
                        id=f"{doc.id}:{s_idx}:{c_idx}",
                        doc_id=doc.id,
                        text=self._prefix(expanded, doc, section),  # embed: header + expansion
                        parent_text=section.text,
                        section_type=section.type,
                        heading=section.heading,
                        payload=self._payload(doc, section, expanded, s_idx),  # store: expansion
                    )
                )
        return out
