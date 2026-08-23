"""Gradio UI for the pravenc research assistant.

    pravenc-ask ui            # then open http://localhost:7860

Models load lazily on the first question, so startup is instant and the first
query is the slow one.
"""
from __future__ import annotations

import atexit
import random
import re
import shutil
import tempfile
from pathlib import Path

import gradio as gr

from .config import Config
from .generate import Assistant

CSS = """
.answer-box { font-size: 1.02rem; line-height: 1.6; }
footer { display: none !important; }
"""

PROCESSING_MESSAGES = [
    "Theologizing with confidence…",
    "Pontificating…",
    "Conducting exegesis…",
    "Deducing boldly…",
    "Summoning the citations…",
]

# The filename stem in this repo is the article id and equals the number in
# source_url (see README); raw.githubusercontent.com serves the .md file
# directly rather than GitHub's rendered blob page, so the link downloads.
PRAVENC_MD_RAW_BASE = "https://raw.githubusercontent.com/slavonic/pravenc-md/main/articles"


def _format_sources(ans) -> str:
    if not ans.sources:
        return "_No sources cited._"
    lines = []
    for s in ans.sources:
        loc = []
        if s.volume is not None:
            loc.append(f"т. {s.volume}")
        if s.pages:
            loc.append(f"с. {s.pages}")
        loc_str = ", ".join(loc)
        head = f" — *{s.heading}*" if s.heading else ""
        score = f"  `{s.score:.3f}`" if s.score is not None else ""
        md_link = (
            f" · [↓ md]({PRAVENC_MD_RAW_BASE}/{s.doc_id}.md)" if s.doc_id else ""
        )
        lines.append(
            f"**[{s.n}]** [{s.title}]({s.url}){md_link}{head}"
            + (f"  \n<sub>{loc_str}</sub>" if loc_str else "")
            + score
        )
    return "\n\n".join(lines)


def _slugify(text: str, max_len: int = 60) -> str:
    text = re.sub(r"[^\w\s-]", "", text.strip().lower(), flags=re.UNICODE)
    text = re.sub(r"[\s_-]+", "-", text).strip("-")
    return text[:max_len].rstrip("-") or "answer"


def _build_report(question: str, ans) -> str:
    """The same answer + sources shown in the UI, as a standalone .md file."""
    t = ans.timing
    lines = [
        f"# {question}",
        "",
        f"*{ans.model} · {ans.language} · embed {t['embed']:.1f}s · "
        f"search {t['search']:.2f}s · rerank {t['rerank']:.1f}s · "
        f"generate {t.get('generate', 0.0):.1f}s*",
        "",
        ans.text,
        "",
        "## Sources",
        "",
        _format_sources(ans),
    ]
    if ans.dropped_citations:
        lines += [
            "",
            f"*Stripped {len(ans.dropped_citations)} fabricated citation(s): "
            f"{sorted(set(ans.dropped_citations))}*",
        ]
    if ans.uncited:
        lines += ["", "*No valid citations were returned — treat this answer with caution.*"]
    return "\n".join(lines) + "\n"


_REPORT_TEMP_DIRS: set[Path] = set()


def _cleanup_report_temp_dirs() -> None:
    """Backstop for whatever's still around when the process exits.

    Steady-state cleanup happens in `answer()` (each new report deletes the
    previous one), but the very last report written in a session, and any
    left behind by an ungraceful shutdown, wouldn't otherwise be removed.
    """
    for d in list(_REPORT_TEMP_DIRS):
        shutil.rmtree(d, ignore_errors=True)
    _REPORT_TEMP_DIRS.clear()


atexit.register(_cleanup_report_temp_dirs)


def _write_report(question: str, ans) -> str:
    # A fresh temp dir per answer, so the filename can be the question slug
    # (Gradio serves the file under its actual basename) instead of a random
    # tempfile name.
    tmp_dir = Path(tempfile.mkdtemp(prefix="pravenc-ask-"))
    _REPORT_TEMP_DIRS.add(tmp_dir)
    path = tmp_dir / f"{_slugify(question)}.md"
    path.write_text(_build_report(question, ans), encoding="utf-8")
    return str(path)


def build_ui(config_path: str = "config.yaml") -> gr.Blocks:
    cfg = Config.load(config_path)
    state = {"assistant": None, "last_report_dir": None}

    def assistant() -> Assistant:
        if state["assistant"] is None:
            state["assistant"] = Assistant(cfg)
        return state["assistant"]

    def write_report(question: str, ans) -> str:
        """Write the new report, then drop the previous one.

        Bounds disk use to at most one lingering report while the server is
        running (the last one gets caught by the atexit cleanup instead).
        """
        path = _write_report(question, ans)
        prev = state["last_report_dir"]
        if prev is not None:
            shutil.rmtree(prev, ignore_errors=True)
            _REPORT_TEMP_DIRS.discard(prev)
        state["last_report_dir"] = Path(path).parent
        return path

    def answer(question: str, language: str, model: str, top_n: int,
               use_reranker: bool, include_refs: bool):
        """Generator so we can show a pending state, then the result.

        Yielding twice (rather than relying on Gradio's implicit per-output
        loading indicator) is deliberate: it's the only way to guarantee a
        pending message appears exactly once, on every submission including
        the first, and that the Ask button is reliably disabled/re-enabled
        around the actual work rather than around Gradio's own queue
        bookkeeping.
        """
        if not question.strip():
            yield "", "", "", gr.update(interactive=True), gr.update(visible=False)
            return

        # First yield: show pending state immediately, lock the button, and
        # hide any download link left over from a previous answer.
        yield ("", "", random.choice(PROCESSING_MESSAGES),
               gr.update(interactive=False), gr.update(visible=False))

        try:
            a = assistant()
            # live knobs — applied per query without rebuilding the engine
            a.cfg.retrieval.top_n = int(top_n)
            a.cfg.retrieval.use_reranker = bool(use_reranker)
            a.cfg.retrieval.exclude_section_types = (
                [] if include_refs else ["sources", "literature"]
            )

            ans = a.ask(question, language=language, model=model or None)

            notes = []
            t = ans.timing
            notes.append(
                f"**{ans.model}** · {ans.language} · embed {t['embed']:.1f}s · "
                f"search {t['search']:.2f}s · rerank {t['rerank']:.1f}s"
            )
            if ans.dropped_citations:
                notes.append(
                    f"⚠️ stripped {len(ans.dropped_citations)} fabricated citation(s): "
                    f"{sorted(set(ans.dropped_citations))}"
                )
            if ans.uncited:
                notes.append("⚠️ no valid citations — treat with caution.")
            report_path = write_report(question, ans)
            # Second yield: final result, button unlocked, download ready.
            yield (ans.text, _format_sources(ans), " · ".join(notes),
                   gr.update(interactive=True), gr.update(value=report_path, visible=True))
        except Exception as e:  # noqa: BLE001
            yield "", "", f"⚠️ Error: {e}", gr.update(interactive=True), gr.update(visible=False)

    with gr.Blocks(title="Православная энциклопедия — research assistant", css=CSS) as demo:
        gr.Markdown(
            "## Православная энциклопедия — research assistant\n"
            "Ask in Russian or English. Answers are grounded in the encyclopedia "
            "corpus and cite the articles they draw on."
        )
        with gr.Row():
            with gr.Column(scale=3):
                question = gr.Textbox(
                    label="Question / Вопрос",
                    placeholder="Кто такой Алексий, человек Божий?",
                    lines=3,
                    autofocus=True,
                )
                submit = gr.Button("Ask", variant="primary")
            with gr.Column(scale=1):
                model = gr.Dropdown(
                    choices=cfg.llm.models,
                    value=cfg.llm.model if cfg.llm.model in cfg.llm.models
                          else (cfg.llm.models[0] if cfg.llm.models else cfg.llm.model),
                    label="Model", allow_custom_value=True,
                )
                language = gr.Radio(
                    ["auto", "ru", "en"], value="auto", label="Answer language"
                )
                top_n = gr.Slider(
                    1, 10, value=cfg.retrieval.top_n, step=1,
                    label="Sections given to the model",
                )
                use_reranker = gr.Checkbox(
                    value=cfg.retrieval.use_reranker,
                    label="Rerank (better, much slower on CPU)",
                )
                include_refs = gr.Checkbox(
                    value=False, label="Include bibliography sections",
                )

        status = gr.Markdown()
        with gr.Row():
            with gr.Column(scale=3):
                out = gr.Markdown(label="Answer", elem_classes="answer-box")
            with gr.Column(scale=2):
                gr.Markdown("#### Sources")
                srcs = gr.Markdown()

        download = gr.DownloadButton("⬇ Download answer (.md)", visible=False)

        inputs = [question, language, model, top_n, use_reranker, include_refs]
        outputs = [out, srcs, status, submit, download]
        # show_progress="hidden": we render our own pending state and
        # drive the button's disabled/enabled state explicitly above, so
        # Gradio's own per-output loading indicator would just be a second,
        # independently-timed "processing" signal — the exact duplication/
        # inconsistency this replaces.
        submit.click(answer, inputs=inputs, outputs=outputs, show_progress="hidden")
        question.submit(answer, inputs=inputs, outputs=outputs, show_progress="hidden")

    return demo
