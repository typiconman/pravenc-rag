"""Corpus audit: survey frontmatter keys and section headings across all files.

Run this before committing to the section-type map — entries for places, texts,
and concepts use different section vocabularies than a saint's life, and the
output tells you which headings to add to ``section_types`` in config.yaml.

    pravenc-audit run
    pravenc-audit run --top 60
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import frontmatter
import typer

from .config import Config

app = typer.Typer(add_completion=False)
_HEADING = re.compile(r"^#{1,6}\s+(.*)$", re.M)


@app.command()
def run(config: str = "config.yaml", top: int = 40) -> None:
    cfg = Config.load(config)
    files = sorted(Path(cfg.corpus_dir).glob("*.md"))
    keys: Counter[str] = Counter()
    headings: Counter[str] = Counter()

    for f in files:
        post = frontmatter.load(str(f))
        keys.update(post.metadata.keys())
        headings.update(m.strip() for m in _HEADING.findall(post.content))

    typer.echo(f"{len(files)} files\n")
    typer.echo("== frontmatter keys (count) ==")
    for k, n in keys.most_common():
        typer.echo(f"{n:7d}  {k}")
    typer.echo(f"\n== headings — top {top} (count) ==")
    for h, n in headings.most_common(top):
        typer.echo(f"{n:7d}  {h}")
    typer.echo(f"\n{len(headings)} distinct headings total.")


if __name__ == "__main__":
    app()
