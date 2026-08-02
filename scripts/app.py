"""Gradio UI for the pravenc research assistant.

    pravenc-ask ui            # then open http://localhost:7860

Models load lazily on the first question, so startup is instant and the first
query is the slow one.
"""
from __future__ import annotations

import gradio as gr

from .config import Config
from .generate import Assistant

CSS = """
.answer-box { font-size: 1.02rem; line-height: 1.6; }
footer { display: none !important; }
"""

EXAMPLES = [
    ["Кто такой Алексий, человек Божий?", "auto"],
    ["What is the hymnography of Alexius, Man of God?", "auto"],
    ["Какие источники сообщают о сирийской версии жития?", "auto"],
]


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
        lines.append(
            f"**[{s.n}]** [{s.title}]({s.url}){head}"
            + (f"  \n<sub>{loc_str}</sub>" if loc_str else "")
            + score
        )
    return "\n\n".join(lines)


def build_ui(config_path: str = "config.yaml") -> gr.Blocks:
    cfg = Config.load(config_path)
    state = {"assistant": None}

    def assistant() -> Assistant:
        if state["assistant"] is None:
            state["assistant"] = Assistant(cfg)
        return state["assistant"]

    def answer(question: str, language: str, top_n: int, use_reranker: bool,
               include_refs: bool):
        if not question.strip():
            return "", "", ""
        a = assistant()
        # live knobs — applied per query without rebuilding the engine
        a.cfg.retrieval.top_n = int(top_n)
        a.cfg.retrieval.use_reranker = bool(use_reranker)
        a.cfg.retrieval.exclude_section_types = (
            [] if include_refs else ["sources", "literature"]
        )

        ans = a.ask(question, language=language)

        notes = []
        t = ans.timing
        notes.append(
            f"language: **{ans.language}** · embed {t['embed']:.1f}s · "
            f"search {t['search']:.2f}s · rerank {t['rerank']:.1f}s"
        )
        if ans.dropped_citations:
            notes.append(
                f"⚠️ dropped {len(ans.dropped_citations)} invented citation marker(s): "
                f"{sorted(set(ans.dropped_citations))}"
            )
        if ans.uncited:
            notes.append("⚠️ the model returned no citations — treat with caution.")
        return ans.text, _format_sources(ans), " · ".join(notes)

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

        gr.Examples(examples=EXAMPLES, inputs=[question, language])

        inputs = [question, language, top_n, use_reranker, include_refs]
        submit.click(answer, inputs=inputs, outputs=[out, srcs, status])
        question.submit(answer, inputs=inputs, outputs=[out, srcs, status])

    return demo
