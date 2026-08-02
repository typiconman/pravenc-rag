"""Qdrant collection with named dense + sparse vectors, plus upsert.

The collection holds one point per child chunk: a ``dense`` vector (cosine) and
a ``sparse`` vector (BGE-M3 lexical weights) so the query side can run hybrid
search with server-side fusion. Payload indexes on ``doc_id``, ``section_type``
and ``volume`` support metadata filtering (e.g. excluding reference sections).
"""
from __future__ import annotations

from qdrant_client import QdrantClient, models


class Store:
    def __init__(self, url: str, collection: str, dense_dim: int = 1024, timeout: float = 30.0):
        self.client = QdrantClient(url=url, timeout=timeout)
        self.collection = collection
        self.dense_dim = dense_dim

    def ensure_collection(self, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if exists:
            return
        self.client.create_collection(
            self.collection,
            vectors_config={
                "dense": models.VectorParams(
                    size=self.dense_dim, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )
        for field, schema in (
            ("doc_id", models.PayloadSchemaType.KEYWORD),
            ("section_type", models.PayloadSchemaType.KEYWORD),
            ("volume", models.PayloadSchemaType.INTEGER),
        ):
            self.client.create_payload_index(self.collection, field, schema)

    def delete_doc(self, doc_id: str) -> None:
        """Remove every point belonging to one article (all its chunks).

        Used on update before re-inserting a modified article, so sections that
        shrank or were deleted don't leave orphaned chunks behind, and for
        deletions and the old side of a rename.
        """
        self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id", match=models.MatchValue(value=doc_id)
                        )
                    ]
                )
            ),
        )

    def upsert(self, ids, dense, sparse, payloads) -> None:
        points = []
        for i, pid in enumerate(ids):
            lw = sparse[i]
            points.append(
                models.PointStruct(
                    id=pid,
                    vector={
                        "dense": dense[i].tolist(),
                        "sparse": models.SparseVector(
                            indices=[int(k) for k in lw.keys()],
                            values=[float(v) for v in lw.values()],
                        ),
                    },
                    payload=payloads[i],
                )
            )
        self.client.upsert(self.collection, points=points)
