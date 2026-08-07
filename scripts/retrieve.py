"""Hybrid retrieval + reranking, exposed as a LlamaIndex retriever.

Pipeline per query:
  1. Encode the question with BGE-M3 (dense + sparse) — cross-lingual, so an
     English question matches Russian passages.
  2. Qdrant hybrid search: dense and sparse prefetch fused server-side with RRF.
  3. Rerank the top candidates with BGE-reranker-v2-m3 (cross-encoder).
  4. Collapse chunks to their parent sections, hydrate the full section text
     from the corpus, and emit LlamaIndex nodes carrying citation metadata.

Step 3 is by far the slowest part on CPU (a cross-encoder pass per candidate).
Tune ``retrieval.rerank_candidates`` down, or set ``use_reranker: false``, if
queries feel sluggish.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from qdrant_client import models

from .config import Config
from .hydrate import ParentHydrator


@dataclass
class Timing:
    embed: float = 0.0
    search: float = 0.0
    rerank: float = 0.0


def _rows(scored: list[tuple], score_kind: str, top_n: int, mark_selection: bool = True) -> list[dict]:
    """Build report rows from a (point, score) list.

    When ``mark_selection`` is true, rows are also flagged with the same
    (doc_id, section) dedup + top_n cutoff ``_retrieve`` applies — i.e. which
    ones would actually reach the LLM. Pass false for a listing that isn't the
    final order (e.g. pre-rerank hybrid results), where "selected" would be
    misleading.
    """
    rows: list[dict] = []
    seen: set[tuple[str, int]] = set()
    n_selected = 0
    for rank, (point, score) in enumerate(scored, 1):
        p = point.payload or {}
        key = (p.get("doc_id", ""), int(p.get("section_idx", 0)))
        selected = False
        if mark_selection and key not in seen:
            seen.add(key)
            if n_selected < top_n:
                selected = True
                n_selected += 1
        rows.append(
            {
                "rank": rank,
                "score": float(score) if score is not None else None,
                "score_kind": score_kind,
                "doc_id": p.get("doc_id"),
                "article_title": p.get("article_title") or "",
                "heading": p.get("heading"),
                "section_type": p.get("section_type"),
                "selected": selected,
            }
        )
    return rows


def format_rows(rows: list[dict], top_n: int, hybrid_limit: int, header: bool = True) -> list[str]:
    """Render candidate rows as plain text lines — shared by `pravenc-ask retrieve`
    and the `--debug` flag, so both report retrieval the same way."""
    if not rows:
        return ["No candidates returned — is anything indexed and are section filters too strict?"]
    lines: list[str] = []
    if header:
        kind = rows[0]["score_kind"]
        lines.append(
            f"{len(rows)} candidates ({kind} score); '>' = sent to LLM "
            f"(top_n={top_n}, hybrid_limit={hybrid_limit})"
        )
        lines.append("")
    for row in rows:
        mark = ">" if row["selected"] else " "
        score = f"{row['score']:.4f}" if row["score"] is not None else "   -  "
        head = f" · {row['heading']}" if row["heading"] else ""
        lines.append(
            f"{mark} {row['rank']:>2}. {score}  {row['article_title']}{head}"
            f"  [{row['section_type']}]"
        )
    return lines


class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        cfg: Config,
        embedder,
        client,
        hydrator: ParentHydrator | None = None,
        reranker=None,
        debug: bool = False,
    ):
        self.cfg = cfg
        self.embedder = embedder
        self.client = client
        self.hydrator = hydrator or ParentHydrator(cfg.corpus_dir, cfg.section_types)
        self._reranker = reranker
        self.last_timing = Timing()
        self.debug = debug
        super().__init__()

    # --- lazily constructed reranker (heavy import) -------------------------
    @property
    def reranker(self):
        if self._reranker is None and self.cfg.retrieval.use_reranker:
            from FlagEmbedding import FlagReranker

            self._reranker = FlagReranker(
                self.cfg.retrieval.reranker_model, use_fp16=True
            )
        return self._reranker

    def _filter(self) -> models.Filter | None:
        excluded = self.cfg.retrieval.exclude_section_types
        if not excluded:
            return None
        return models.Filter(
            must_not=[
                models.FieldCondition(
                    key="section_type", match=models.MatchAny(any=list(excluded))
                )
            ]
        )

    def _search(self, question: str):
        r = self.cfg.retrieval

        t0 = time.perf_counter()
        dense, sparse = self.embedder.encode([question], batch_size=1)
        lw = sparse[0]
        self.last_timing.embed = time.perf_counter() - t0

        t0 = time.perf_counter()
        res = self.client.query_points(
            self.cfg.collection,
            prefetch=[
                models.Prefetch(
                    query=dense[0].tolist(), using="dense", limit=r.hybrid_limit,
                    filter=self._filter(),
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=[int(k) for k in lw.keys()],
                        values=[float(v) for v in lw.values()],
                    ),
                    using="sparse", limit=r.hybrid_limit,
                    filter=self._filter(),
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=r.hybrid_limit,
            with_payload=True,
        )
        self.last_timing.search = time.perf_counter() - t0
        points = res.points

        if self.debug:
            # Not yet reranked, so "selected" would misrepresent the final
            # order UNLESS reranking is off, in which case this IS the order.
            rows = _rows([(p, p.score) for p in points], "hybrid", r.top_n,
                         mark_selection=not r.use_reranker)
            for line in format_rows(rows, r.top_n, r.hybrid_limit):
                print(f"[retrieve] {line}")
            print(f"[retrieve] embed {self.last_timing.embed:.2f}s, "
                  f"search {self.last_timing.search:.2f}s")

        return points

    def _rerank(self, question: str, points):
        r = self.cfg.retrieval
        self.last_timing.rerank = 0.0
        if not r.use_reranker or not points:
            if self.debug and not r.use_reranker:
                print("[rerank] skipped (use_reranker=false); hybrid order stands")
            return [(p, p.score) for p in points]

        candidates = points[: r.rerank_candidates]
        pairs = [[question, p.payload.get("text", "")] for p in candidates]
        t0 = time.perf_counter()
        scores = self.reranker.compute_score(pairs, normalize=True)
        self.last_timing.rerank = time.perf_counter() - t0
        if not isinstance(scores, list):
            scores = [scores]
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        if self.debug:
            rows = _rows(scored, "rerank", r.top_n, mark_selection=True)
            for line in format_rows(rows, r.top_n, r.hybrid_limit):
                print(f"[rerank] {line}")
            print(f"[rerank] {r.reranker_model} in {self.last_timing.rerank:.2f}s")

        return scored

    def diagnose(self, question: str, rerank: bool | None = None) -> list[dict]:
        """Return the ranked candidate pool for a query — no LLM call.

        Mirrors the real selection (hybrid RRF search, optional rerank, then the
        same (doc_id, section) dedup and top_n cutoff `_retrieve` applies) but
        keeps every candidate and flags which ones would actually reach the LLM.
        Use it to see whether the responsive articles are even in the pool, or
        whether one broad article is crowding them out.
        """
        if rerank is not None:
            self.cfg.retrieval.use_reranker = rerank
        points = self._search(question)
        scored = self._rerank(question, points)
        used_rr = bool(self.cfg.retrieval.use_reranker and points)
        return _rows(scored, "rerank" if used_rr else "hybrid", self.cfg.retrieval.top_n)

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        question = query_bundle.query_str
        points = self._search(question)
        scored = self._rerank(question, points)

        # Collapse to parent sections: several chunks can share one section, and
        # feeding the same section text twice wastes context and skews citations.
        out: list[NodeWithScore] = []
        seen: set[tuple[str, int]] = set()
        for point, score in scored:
            p = point.payload or {}
            key = (p.get("doc_id", ""), int(p.get("section_idx", 0)))
            if key in seen:
                continue
            seen.add(key)

            text = self.hydrator.hydrate(p) or p.get("text", "")
            node = TextNode(
                text=text,
                id_=p.get("chunk_id", f"{key[0]}:{key[1]}"),
                metadata={
                    "doc_id": p.get("doc_id"),
                    "article_title": p.get("article_title"),
                    "heading": p.get("heading"),
                    "section_type": p.get("section_type"),
                    "volume": p.get("volume"),
                    "page_numbers": p.get("page_numbers"),
                    "source_url": p.get("source_url"),
                    "citation": p.get("citation"),
                    "section_author": p.get("section_author"),
                },
                excluded_llm_metadata_keys=list(
                    ["doc_id", "section_type", "source_url", "citation",
                     "volume", "page_numbers", "section_author"]
                ),
            )
            out.append(NodeWithScore(node=node, score=float(score)))
            if len(out) >= self.cfg.retrieval.top_n:
                break

        if self.debug:
            print(f"[retrieve] {len(out)} section(s) selected for generation")

        return out
