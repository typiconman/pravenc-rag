"""Query CLI.

    pravenc-ask ui                                   # launch the web UI
    pravenc-ask q "Кто такой Алексий, человек Божий?"
    pravenc-ask q "Who was Alexius?" --language en --model google/gemma-4-31b-it
    pravenc-ask q "..." --debug                       # narrate every pipeline step
    pravenc-ask compare "Did the canon replace the kontakion?"   # same Q, many models
    pravenc-ask retrieve "..."                        # ranked candidates, no LLM call
    pravenc-ask models --filter qwen                 # live catalog for your provider
    pravenc-ask check                                # Qdrant + LLM provider + corpus
"""
from __future__ import annotations

import typer

from .config import Config

app = typer.Typer(add_completion=False)


def _print_sources(ans) -> None:
    if not ans.sources:
        return
    typer.echo("Sources:")
    for s in ans.sources:
        loc = ", ".join(
            x for x in (f"т. {s.volume}" if s.volume is not None else "",
                        f"с. {s.pages}" if s.pages else "") if x
        )
        typer.echo(f"  [{s.n}] {s.title}" + (f" ({loc})" if loc else ""))
        typer.echo(f"       {s.url}")


def _print_answer(ans) -> None:
    typer.echo("\n" + ans.text + "\n")
    _print_sources(ans)
    if ans.dropped_citations:
        typer.echo(f"\n! stripped {len(ans.dropped_citations)} fabricated citation(s): "
                   f"{sorted(set(ans.dropped_citations))}")
    if ans.uncited:
        typer.echo("\n! no valid citations returned — treat this answer with caution.")
    t = ans.timing
    typer.echo(
        f"\n({ans.model}; {ans.language}; embed {t['embed']:.1f}s, "
        f"search {t['search']:.2f}s, rerank {t['rerank']:.1f}s, "
        f"generate {t.get('generate', 0.0):.1f}s)"
    )


@app.command()
def ui(config: str = "config.yaml", port: int = 7860, share: bool = False) -> None:
    """Launch the Gradio interface."""
    from .app import build_ui

    build_ui(config).launch(server_port=port, share=share, inbrowser=True)


@app.command()
def q(
    question: str = typer.Argument(..., help="The question to ask."),
    language: str = typer.Option(
        "auto", help="Answer language: auto (detect from the question's script), ru, or en. "
        "The query itself can be in any language regardless of this setting."
    ),
    model: str = typer.Option("", help="Override the configured model (a slug from `models`)."),
    debug: bool = typer.Option(
        False, "--debug", help="Print each pipeline step as it happens: hybrid search "
        "and rerank candidates (same format as `retrieve`), then the LLM call and "
        "citation verification."
    ),
    config: str = "config.yaml",
) -> None:
    """Ask one question and print a cited answer."""
    from .generate import Assistant

    cfg = Config.load(config)
    ans = Assistant(cfg).ask(question, language=language, model=model or None, debug=debug)
    _print_answer(ans)


@app.command()
def compare(
    question: str = typer.Argument(..., help="The question to run across several models."),
    language: str = typer.Option(
        "auto", help="Answer language: auto (detect from the question's script), ru, or en. "
        "The query itself can be in any language regardless of this setting."
    ),
    models: str = typer.Option("", help="Comma-separated slugs; defaults to config llm.models."),
    debug: bool = typer.Option(
        False, "--debug", help="Print each pipeline step as it happens: hybrid search "
        "and rerank candidates (same format as `retrieve`), then each model's LLM call "
        "and citation verification."
    ),
    config: str = "config.yaml",
) -> None:
    """Ask the SAME question across several models to compare their answers.

    Retrieval (embedding, hybrid search, reranking) runs ONCE and is reused for
    every model, so comparing N models costs one retrieval + N generations.
    """
    from .generate import Assistant

    cfg = Config.load(config)
    model_list = [m.strip() for m in models.split(",") if m.strip()] or cfg.llm.models
    a = Assistant(cfg)

    # Retrieve ONCE — embedding, hybrid search and reranking are
    # model-independent, so they must not repeat per model.
    lang, qb, nodes = a.retrieve(question, language=language, debug=debug)
    t = a.retriever.last_timing
    typer.echo(
        f"Comparing {len(model_list)} models on: {question!r}\n"
        f"(retrieved {len(nodes)} sources once — "
        f"embed {t.embed:.1f}s, search {t.search:.2f}s, rerank {t.rerank:.1f}s)"
    )
    for m in model_list:
        typer.echo("\n" + "=" * 72 + f"\nMODEL: {m}\n" + "=" * 72)
        try:
            ans = a.generate(qb, nodes, lang, m, debug=debug)
            typer.echo("\n" + ans.text + "\n")
            _print_sources(ans)
            flag = "  ⚠ SUSPECT" if ans.suspect else ""
            typer.echo(
                f"→ cited {len(ans.sources)} sources; "
                f"{len(ans.dropped_citations)} fabricated markers stripped{flag}"
            )
        except Exception as e:  # noqa: BLE001
            typer.echo(f"  FAILED: {e}")


@app.command()
def models(
    filter: str = typer.Option("", help="Case-insensitive substring filter (e.g. qwen, gemma)."),
    config: str = "config.yaml",
) -> None:
    """List model IDs actually available on the configured provider (live)."""
    import os

    import httpx

    cfg = Config.load(config)
    c = cfg.llm
    try:
        if c.provider == "ollama":
            r = httpx.get(f"{c.ollama_url}/api/tags", timeout=15)
            ids = [m["name"] for m in r.json().get("models", [])]
        else:
            key = os.environ.get(c.api_key_env, "")
            headers = {"Authorization": f"Bearer {key}"} if key else {}
            r = httpx.get(f"{c.api_base}/models", headers=headers, timeout=30)
            ids = [m["id"] for m in r.json().get("data", [])]
    except Exception as e:  # noqa: BLE001
        typer.echo(f"Could not fetch models from {c.provider}: {e}")
        raise typer.Exit(1)

    if filter:
        ids = [i for i in ids if filter.lower() in i.lower()]
    for i in sorted(ids):
        typer.echo(i)
    typer.echo(f"\n{len(ids)} models" + (f" matching '{filter}'" if filter else ""))


@app.command()
def retrieve(
    question: str = typer.Argument(..., help="Query to inspect (retrieval only, no LLM)."),
    rerank: bool = typer.Option(None, "--rerank/--no-rerank",
                                help="Override use_reranker for this run (loads the reranker)."),
    config: str = "config.yaml",
) -> None:
    """Show what retrieval surfaces for a query — ranked candidates, no generation.

    '>' marks the sections that would be sent to the LLM (top_n after dedup).
    The distinct-article summary is the quick tell: is the responsive article
    even in the pool, or is one broad article crowding it out?
    """
    from collections import Counter

    from qdrant_client import QdrantClient

    from .embed import Embedder
    from .retrieve import HybridRetriever, format_rows

    cfg = Config.load(config)
    r = HybridRetriever(
        cfg,
        embedder=Embedder(cfg.embed_model),
        client=QdrantClient(url=cfg.qdrant_url, timeout=cfg.qdrant_timeout),
    )
    rows = r.diagnose(question, rerank=rerank)
    if not rows:
        typer.echo("No candidates returned — is anything indexed and are section filters too strict?")
        raise typer.Exit()

    typer.echo("")
    for line in format_rows(rows, cfg.retrieval.top_n, cfg.retrieval.hybrid_limit):
        typer.echo(line)

    arts = Counter(row["article_title"] for row in rows)
    typer.echo(f"\nDistinct articles in pool ({len(arts)}):")
    for title, n in arts.most_common():
        typer.echo(f"  {n:>2}×  {title}")

    t = r.last_timing
    typer.echo(f"\n(embed {t.embed:.1f}s, search {t.search:.2f}s, rerank {t.rerank:.1f}s)")


@app.command()
def check(config: str = "config.yaml") -> None:
    """Confirm the index, the LLM provider, and the corpus are ready."""
    import os

    import httpx
    from qdrant_client import QdrantClient

    cfg = Config.load(config)

    try:
        client = QdrantClient(url=cfg.qdrant_url, timeout=cfg.qdrant_timeout)
        n = client.count(cfg.collection, exact=True).count
        typer.echo(f"Qdrant OK — collection '{cfg.collection}': {n} points")
    except Exception as e:  # noqa: BLE001
        typer.echo(f"Qdrant FAILED at {cfg.qdrant_url}: {e}")

    c = cfg.llm
    if c.provider == "ollama":
        try:
            r = httpx.get(f"{c.ollama_url}/api/tags", timeout=10)
            names = [m["name"] for m in r.json().get("models", [])]
            typer.echo(f"Ollama OK — models: {', '.join(names) or '(none)'}")
            if not any(n.startswith(c.model.split(':')[0]) for n in names):
                typer.echo(f"  ! '{c.model}' not pulled: ollama pull {c.model}")
        except Exception as e:  # noqa: BLE001
            typer.echo(f"Ollama FAILED at {c.ollama_url}: {e}")
    else:
        key = os.environ.get(c.api_key_env)
        if not key:
            typer.echo(f"LLM: {c.api_key_env} is not set — export your {c.provider} key.")
        else:
            try:
                r = httpx.get(f"{c.api_base}/models",
                              headers={"Authorization": f"Bearer {key}"}, timeout=20)
                ids = {m["id"] for m in r.json().get("data", [])}
                typer.echo(f"LLM OK — provider '{c.provider}', {len(ids)} models available")
                if c.model not in ids:
                    typer.echo(f"  ! active model '{c.model}' not in catalog — see `pravenc-ask models`")
                for m in c.models:
                    if m not in ids:
                        typer.echo(f"  ! candidate '{m}' not in catalog (slug may have changed)")
            except Exception as e:  # noqa: BLE001
                typer.echo(f"LLM check failed at {c.api_base}: {e}")

    corpus = cfg.corpus_dir
    typer.echo(
        f"Corpus {'OK' if corpus.exists() else 'MISSING'} — {corpus} "
        f"({len(list(corpus.glob('*.md'))) if corpus.exists() else 0} articles)"
    )


if __name__ == "__main__":
    app()
