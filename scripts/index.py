"""Offline indexing entry point: repo -> parse -> chunk -> embed -> Qdrant.

    pravenc-index build                     # index the whole corpus
    pravenc-index build --recreate          # drop and rebuild the collection
    pravenc-index build --limit 50          # smoke-test on the first 50 files

    pravenc-index update updates/v2.4.txt   # apply a change manifest incrementally
"""
from __future__ import annotations

import uuid
from pathlib import Path

import typer

from .chunk import Chunker
from .config import Config
from .embed import Embedder
from .ingest import parse_article
from .store import Store

app = typer.Typer(add_completion=False)

# Fixed namespace so a chunk's string id maps to a stable Qdrant UUID across runs
# (upserts are then idempotent — re-indexing overwrites rather than duplicates).
_NS = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NS, chunk_id))


def _index_files(files, cfg, store, chunker, embedder, progress_every: int = 200) -> int:
    """Parse, chunk, embed and upsert a list of article files. Returns chunk count."""
    ids: list[str] = []
    texts: list[str] = []
    payloads: list[dict] = []
    n_chunks = 0

    def flush() -> None:
        nonlocal n_chunks
        if not texts:
            return
        dense, sparse = embedder.encode(texts, batch_size=cfg.batch_size)
        store.upsert(ids, dense, sparse, payloads)
        n_chunks += len(texts)
        ids.clear()
        texts.clear()
        payloads.clear()

    for n, f in enumerate(files, 1):
        doc = parse_article(f, cfg.section_types)
        for ch in chunker.chunk_document(doc):
            ids.append(_point_id(ch.id))
            texts.append(ch.text)
            payload = dict(ch.payload)
            payload["chunk_id"] = ch.id
            payloads.append(payload)
            if len(texts) >= cfg.batch_size:
                flush()
        if progress_every and n % progress_every == 0:
            typer.echo(f"  {n}/{len(files)} files, {n_chunks + len(texts)} chunks")
    flush()
    return n_chunks


def _parse_manifest(text: str) -> tuple[list[str], list[str]]:
    """Read `git diff --name-status` output -> (stems_to_index, stems_to_delete).

    Handles A/M/D and R.../C... (rename/copy carry two paths). Only articles/*.md
    lines are considered; anything else is ignored.
    """
    to_index: list[str] = []
    to_delete: list[str] = []
    for line in text.splitlines():
        line = line.rstrip()
        if not line:
            continue
        parts = line.split("\t")
        status = parts[0][:1]
        paths = [p for p in parts[1:] if p.endswith(".md")]
        if not paths:
            continue
        if status in ("A", "M"):
            stem = Path(paths[0]).stem
            to_delete.append(stem)   # M needs a clean slate; A is a harmless no-op
            to_index.append(stem)
        elif status == "D":
            to_delete.append(Path(paths[0]).stem)
        elif status in ("R", "C"):
            old, new = paths[0], paths[-1]
            if status == "R":
                to_delete.append(Path(old).stem)
            to_delete.append(Path(new).stem)
            to_index.append(Path(new).stem)
    return list(dict.fromkeys(to_index)), list(dict.fromkeys(to_delete))


@app.command()
def build(
    config: str = "config.yaml",
    recreate: bool = typer.Option(False, help="Drop and recreate the collection first."),
    limit: int = typer.Option(0, help="Index only the first N files (0 = all)."),
) -> None:
    cfg = Config.load(config)
    store = Store(cfg.qdrant_url, cfg.collection, cfg.dense_dim)
    store.ensure_collection(recreate=recreate)
    chunker = Chunker(cfg.embed_model, cfg.max_tokens, cfg.overlap_tokens)
    embedder = Embedder(cfg.embed_model)

    files = sorted(Path(cfg.corpus_dir).glob("*.md"))
    if limit:
        files = files[:limit]
    typer.echo(f"Indexing {len(files)} files -> collection '{cfg.collection}'")
    n_chunks = _index_files(files, cfg, store, chunker, embedder)
    typer.echo(f"Done: {n_chunks} chunks indexed.")


@app.command()
def update(
    manifest: str = typer.Argument(..., help="Path to a change manifest (git --name-status)."),
    config: str = "config.yaml",
) -> None:
    """Apply an incremental corpus update from a maintainer-emitted manifest.

    Deletes the points of every changed/removed/renamed article first (so
    shrunk or deleted sections leave nothing stale), then re-indexes the
    added/modified/new-name files.
    """
    cfg = Config.load(config)
    to_index, to_delete = _parse_manifest(Path(manifest).read_text(encoding="utf-8"))

    store = Store(cfg.qdrant_url, cfg.collection, cfg.dense_dim)
    store.ensure_collection(recreate=False)

    for stem in to_delete:
        store.delete_doc(stem)

    files = [Path(cfg.corpus_dir) / f"{s}.md" for s in to_index]
    missing = [f for f in files if not f.exists()]
    files = [f for f in files if f.exists()]
    if missing:
        typer.echo(f"  warning: {len(missing)} manifest files not found on disk (skipped)")

    n_chunks = 0
    if files:
        chunker = Chunker(cfg.embed_model, cfg.max_tokens, cfg.overlap_tokens)
        embedder = Embedder(cfg.embed_model)
        n_chunks = _index_files(files, cfg, store, chunker, embedder, progress_every=0)

    typer.echo(
        f"update: {len(to_delete)} docs cleared, {len(files)} re-indexed ({n_chunks} chunks)."
    )


if __name__ == "__main__":
    app()
