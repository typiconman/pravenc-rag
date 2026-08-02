"""Query CLI.

    pravenc-ask ui                                  # launch the web UI
    pravenc-ask q "Кто такой Алексий, человек Божий?"
    pravenc-ask q "Who was Alexius, Man of God?" --language en
    pravenc-ask check                               # verify Qdrant + Ollama are ready
"""
from __future__ import annotations

import typer

from .config import Config

app = typer.Typer(add_completion=False)


@app.command()
def ui(config: str = "config.yaml", port: int = 7860, share: bool = False) -> None:
    """Launch the Gradio interface."""
    from .app import build_ui

    build_ui(config).launch(server_port=port, share=share, inbrowser=True)


@app.command()
def q(
    question: str = typer.Argument(..., help="The question to ask."),
    language: str = typer.Option("auto", help="auto | ru | en"),
    config: str = "config.yaml",
) -> None:
    """Ask one question and print a cited answer."""
    from .generate import Assistant

    ans = Assistant(Config.load(config)).ask(question, language=language)
    typer.echo("\n" + ans.text + "\n")
    if ans.sources:
        typer.echo("Sources:")
        for s in ans.sources:
            loc = ", ".join(
                x for x in (f"т. {s.volume}" if s.volume is not None else "",
                            f"с. {s.pages}" if s.pages else "") if x
            )
            typer.echo(f"  [{s.n}] {s.title}" + (f" ({loc})" if loc else ""))
            typer.echo(f"       {s.url}")
    if ans.dropped_citations:
        typer.echo(f"\n! dropped invented citations: {sorted(set(ans.dropped_citations))}")
    if ans.uncited:
        typer.echo("\n! no citations returned — treat this answer with caution.")
    t = ans.timing
    typer.echo(
        f"\n({ans.language}; embed {t['embed']:.1f}s, search {t['search']:.2f}s, "
        f"rerank {t['rerank']:.1f}s)"
    )


@app.command()
def check(config: str = "config.yaml") -> None:
    """Confirm the index and the LLM are reachable before a first query."""
    import httpx
    from qdrant_client import QdrantClient

    cfg = Config.load(config)

    try:
        client = QdrantClient(url=cfg.qdrant_url)
        n = client.count(cfg.collection, exact=True).count
        typer.echo(f"Qdrant OK — collection '{cfg.collection}': {n} points")
    except Exception as e:  # noqa: BLE001
        typer.echo(f"Qdrant FAILED at {cfg.qdrant_url}: {e}")

    try:
        r = httpx.get(f"{cfg.llm.ollama_url}/api/tags", timeout=10)
        names = [m["name"] for m in r.json().get("models", [])]
        ok = any(n.startswith(cfg.llm.model.split(":")[0]) for n in names)
        typer.echo(f"Ollama OK — models: {', '.join(names) or '(none)'}")
        if not ok:
            typer.echo(f"  ! '{cfg.llm.model}' not pulled: ollama pull {cfg.llm.model}")
    except Exception as e:  # noqa: BLE001
        typer.echo(f"Ollama FAILED at {cfg.llm.ollama_url}: {e}")

    corpus = cfg.corpus_dir
    typer.echo(
        f"Corpus {'OK' if corpus.exists() else 'MISSING'} — {corpus} "
        f"({len(list(corpus.glob('*.md'))) if corpus.exists() else 0} articles)"
    )


if __name__ == "__main__":
    app()
