"""Parsed representations of a pravenc-md article.

The ingest stage turns each Markdown file into one ``Document`` (metadata +
ordered ``Section`` list + reference/media sidecars). The chunk stage turns a
``Document`` into embeddable ``Chunk`` objects that carry the citation payload.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Reference:
    """A cross-reference to another article.

    Resolved references (relative ``<id>.md`` links) carry ``target_id``.
    Unresolved references (``pravenc.ru/text/<slug>`` links, by title) carry
    ``slug`` and the visible ``anchor`` text.
    """

    anchor: str
    target_id: str | None = None
    slug: str | None = None

    @property
    def resolved(self) -> bool:
        return self.target_id is not None


@dataclass
class Media:
    kind: str            # "figure" | "glyph"
    url: str
    caption: str | None = None


@dataclass
class Section:
    type: str            # normalized tag: body | hymnography | ... | other
    heading: str | None  # raw Russian heading (None for the lead section)
    text: str
    author: str | None = None   # per-section signature, if detected


@dataclass
class Document:
    id: str              # filename stem == number in source_url
    title: str
    authors: list[str]
    volume: int | None
    page_numbers: str | None
    source_url: str
    downloaded_at: str | None
    sections: list[Section] = field(default_factory=list)
    refs_resolved: list[str] = field(default_factory=list)   # target article ids
    refs_unresolved: list[Reference] = field(default_factory=list)
    media: list[Media] = field(default_factory=list)


@dataclass
class Chunk:
    id: str              # f"{doc_id}:{section_idx}:{chunk_idx}"
    doc_id: str
    text: str            # the embedded child window
    parent_text: str     # full section text, returned as generation context
    section_type: str
    heading: str | None
    payload: dict        # everything downstream needs, incl. citation fields
