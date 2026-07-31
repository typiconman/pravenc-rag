"""BGE-M3 embedding: dense + sparse (lexical) in a single pass.

``encode`` returns dense vectors (numpy) and sparse lexical weights (one
``{token_id: weight}`` dict per text) ready to convert into Qdrant sparse
vectors. ColBERT/multi-vector output is disabled — hybrid dense+sparse is what
the retrieval design uses.
"""
from __future__ import annotations


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = True):
        from FlagEmbedding import BGEM3FlagModel  # lazy: pulls torch

        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)

    def encode(self, texts: list[str], batch_size: int = 16):
        out = self.model.encode(
            texts,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        # dense_vecs: (n, 1024) float array; lexical_weights: list[dict[str, float]]
        return out["dense_vecs"], out["lexical_weights"]
