"""Typed configuration loaded from ``config.yaml``."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

@dataclass
class RetrievalConfig:
    hybrid_limit: int = 40          # candidates pulled from Qdrant (fast)
    rerank_candidates: int = 20     # how many of those the cross-encoder scores (slow on CPU)
    top_n: int = 5                  # sections finally handed to the LLM
    use_reranker: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    exclude_section_types: list[str] = field(default_factory=lambda: ["sources", "literature"])

@dataclass
class LLMConfig:
    model: str = "gemma3:b"
    ollama_url: str = "http://localhost:11434"
    num_ctx: int = 16384            # MUST exceed the retrieved context; Ollama's default truncates
    temperature: float = 0.2
    request_timeout: float = 900.0
    citation_chunk_size: int = 1024

@dataclass
class Config:
    corpus_dir: Path
    qdrant_url: str
    collection: str
    embed_model: str
    dense_dim: int
    batch_size: int
    max_tokens: int
    overlap_tokens: int
    section_types: dict[str, str] = field(default_factory=dict)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    @classmethod
    def load(cls, path: str | Path = "config.yaml") -> "Config":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        chunking = data.get("chunking", {})
        r = data.get("retrieval", {})
        l = data.get("llm", {})
        return cls(
            corpus_dir=Path(data["corpus_dir"]).expanduser(),
            qdrant_url=data.get("qdrant_url", "http://localhost:6333"),
            collection=data.get("collection", "pravenc"),
            embed_model=data.get("embed_model", "BAAI/bge-m3"),
            dense_dim=int(data.get("dense_dim", 1024)),
            batch_size=int(data.get("batch_size", 16)),
            max_tokens=int(chunking.get("max_tokens", 512)),
            overlap_tokens=int(chunking.get("overlap_tokens", 64)),
            section_types=data.get("section_types") or {},
            retrieval=RetrievalConfig(
                hybrid_limit=int(r.get("hybrid_limit", 40)),
                rerank_candidates=int(r.get("rerank_candidates", 20)),
                top_n=int(r.get("top_n", 5)),
                use_reranker=bool(r.get("use_reranker", True)),
                reranker_model=r.get("reranker_model", "BAAI/bge-reranker-v2-m3"),
                exclude_section_types=list(
                    r.get("exclude_section_types", ["sources", "literature"])
                ),
            ),
            llm=LLMConfig(
                model=l.get("model", "gemma3:12b"),
                ollama_url=l.get("ollama_url", "http://localhost:11434"),
                num_ctx=int(l.get("num_ctx", 16384)),
                temperature=float(l.get("temperature", 0.2)),
                request_timeout=float(l.get("request_timeout", 900.0)),
                citation_chunk_size=int(l.get("citation_chunk_size", 1024)),
            ),
        )
