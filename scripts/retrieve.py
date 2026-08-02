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


class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        cfg: Config,
        embedder,
        client,
        hydrator: ParentHydrator | None = None,
        reranker=None,
    ):
        self.cfg = cfg
        self.embedder = embedder
        self.client = client
        self.hydrator = hydrator or ParentHydrator(cfg.corpus_dir, cfg.section_types)
        self._reranker = reranker
        self.last_timing = Timing()
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
        return res.points

    def _rerank(self, question: str, points):
        r = self.cfg.retrieval
        self.last_timing.rerank = 0.0
        if not r.use_reranker or not points:
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
        return scored

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
        return out
