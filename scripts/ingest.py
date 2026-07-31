"""Parse a single pravenc-md Markdown file into a ``Document``.

Implements the ingest spec:
  * YAML frontmatter -> document metadata (title, authors, volume, pages, url).
  * Body split into sections on any heading; lead text before the first
    heading is the ``body`` section. Repeated headings are disambiguated.
  * Two classes of cross-reference: resolved (relative ``<id>.md``) and
    unresolved (``pravenc.ru/text/<slug>``, by title).
  * Two classes of image: captioned figures (kept as caption text) and
    ``/char/`` glyph images (replaced inline with a visible placeholder).
  * Per-section author signatures stripped from the text when detected.
"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlparse

import frontmatter
from markdown_it import MarkdownIt

from .models import Document, Media, Reference, Section

_md = MarkdownIt("commonmark")

GLYPH_PLACEHOLDER = "\u27e8glyph\u27e9"  # ⟨glyph⟩

# Deterministic Russian-heading -> section-type lookup. Extend from the audit
# command's output; anything unmapped falls through to "other".
DEFAULT_SECTION_TYPES = {
    "Гимнография": "hymnography",
    "Иконография": "iconography",
    "Источники": "sources",
    "Литература": "literature",
}

_WS = re.compile(r"[ \t\u00a0]+")
_MULTINL = re.compile(r"\n{3,}")


def _classify_link(href: str) -> tuple[str | None, str | None]:
    """Return ("resolved", article_id) | ("unresolved", slug) | (None, None)."""
    if href.endswith(".md"):
        stem = href.rsplit("/", 1)[-1][:-3]
        if stem.isdigit():
            return "resolved", stem
    if "pravenc.ru/text/" in href:
        seg = urlparse(href).path.rsplit("/", 1)[-1]
        if seg.endswith(".html"):
            seg = seg[:-5]
        slug = unquote(seg)
        if slug.isdigit():          # a numeric pravenc link is a resolved id
            return "resolved", slug
        return "unresolved", slug
    return None, None


def _walk_inline(children, parts, refs_resolved, refs_unresolved, media):
    """Flatten one inline token stream into readable text, harvesting refs/media."""
    link = None  # (kind, val, [anchor parts]) while inside a link
    for c in children:
        t = c.type
        if t == "link_open":
            kind, val = _classify_link(c.attrGet("href") or "")
            link = [kind, val, []]
        elif t == "link_close":
            if link is not None:
                kind, val, ap = link
                anchor = "".join(ap).strip()
                parts.extend(ap)                 # keep the visible term in text
                if kind == "resolved":
                    refs_resolved.append(val)
                elif kind == "unresolved":
                    refs_unresolved.append(Reference(anchor=anchor, slug=val))
                link = None
        elif t == "text":
            (link[2] if link else parts).append(c.content)
        elif t in ("softbreak", "hardbreak"):
            (link[2] if link else parts).append(" ")
        elif t == "code_inline":
            (link[2] if link else parts).append(c.content)
        elif t == "image":
            src = c.attrGet("src") or ""
            alt = (c.content or "").strip()
            if "/char/" in src:
                parts.append(GLYPH_PLACEHOLDER)  # non-Unicode glyph: keep a trace
                media.append(Media("glyph", src, None))
            else:
                if alt:
                    parts.append(f" [{alt}] ")   # figure caption is useful text
                media.append(Media("figure", src, alt or None))


def _normalize(text: str) -> str:
    text = _WS.sub(" ", text)
    text = _MULTINL.sub("\n\n", text)
    return text.strip()


def _extract_signature(text: str, known_authors: list[str]) -> tuple[str, str | None]:
    """Strip a trailing author-signature line if it matches known contributors."""
    lines = text.rstrip().split("\n")
    if not lines:
        return text, None
    last = lines[-1].strip()
    names = [n.strip() for n in last.split(",") if n.strip()]
    if (
        names
        and len(last) < 120
        and all(any(n == a or n in a for a in known_authors) for n in names)
    ):
        return "\n".join(lines[:-1]).strip(), last
    return text, None


def parse_article(path: str | Path, section_types: dict[str, str] | None = None) -> Document:
    section_types = section_types or DEFAULT_SECTION_TYPES
    path = Path(path)
    post = frontmatter.load(str(path))
    meta = post.metadata

    authors = [a.strip() for a in str(meta.get("author", "")).split(",") if a.strip()]
    volume = meta.get("volume")
    try:
        volume = int(volume) if volume not in (None, "") else None
    except (TypeError, ValueError):
        volume = None

    doc = Document(
        id=path.stem,
        title=str(meta.get("article_title", "")).strip(),
        authors=authors,
        volume=volume,
        page_numbers=(str(meta["page_numbers"]).strip() if meta.get("page_numbers") else None),
        source_url=str(meta.get("source_url", "")).strip(),
        downloaded_at=(str(meta["downloaded_at"]) if meta.get("downloaded_at") else None),
    )

    tokens = _md.parse(post.content)
    seen_headings: dict[str, int] = {}

    cur_heading: str | None = None
    cur_parts: list[str] = []
    in_heading = False

    def flush() -> None:
        text = _normalize("".join(cur_parts))
        if not text and cur_heading is None:
            return
        text, author = _extract_signature(text, doc.authors)
        heading = cur_heading
        if heading is not None:
            n = seen_headings.get(heading, 0) + 1
            seen_headings[heading] = n
            display = heading if n == 1 else f"{heading} ({n})"
        else:
            display = None
        stype = "body" if heading is None else section_types.get(heading, "other")
        doc.sections.append(Section(type=stype, heading=display, text=text, author=author))

    for tok in tokens:
        if tok.type == "heading_open":
            flush()
            cur_parts = []
            cur_heading = None
            in_heading = True
        elif tok.type == "heading_close":
            in_heading = False
        elif tok.type == "inline":
            if in_heading:
                cur_heading = tok.content.strip()
            else:
                _walk_inline(
                    tok.children or [],
                    cur_parts,
                    doc.refs_resolved,
                    doc.refs_unresolved,
                    doc.media,
                )
                cur_parts.append("\n")
    flush()

    # de-duplicate resolved edges, preserve order
    doc.refs_resolved = list(dict.fromkeys(doc.refs_resolved))
    return doc
