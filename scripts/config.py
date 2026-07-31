"""Typed configuration loaded from ``config.yaml``."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


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

    @classmethod
    def load(cls, path: str | Path = "config.yaml") -> "Config":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        chunking = data.get("chunking", {})
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
        )
