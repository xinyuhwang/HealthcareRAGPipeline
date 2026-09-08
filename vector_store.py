"""
Vector stores: index vectors, answer nearest-neighbour queries.

Neither implementation owns an embedding model, and neither accepts raw text at
query time - see `contracts.py` for why that boundary is drawn here.

- `NumpyVectorStore`  - exact brute-force cosine search via one matmul. Correct
  and dependency-free; O(N) per query, which is fine at this corpus size and is
  the honest baseline against which an ANN index should be compared.
- `MilvusVectorStore` - real Milvus through `pymilvus`. Points at a local
  Milvus Lite file by default, or at a Milvus server/cluster URI in production.
  Same class either way; only the URI changes.

`MilvusVectorStore` needs `pip install "pymilvus[milvus_lite]"`. It is not
imported unless used, so the rest of the project runs without it.
"""

from typing import Sequence

import numpy as np

from contracts import Candidate


def _as_candidates(ids, texts, scores) -> list[Candidate]:
    candidates = []
    for rank, (doc_id, text, score) in enumerate(zip(ids, texts, scores), start=1):
        candidates.append(
            Candidate(
                id=doc_id,
                text=text,
                score=float(score),
                retrieval_score=float(score),
                retrieval_rank=rank,
            )
        )
    return candidates


class NumpyVectorStore:
    """Exact cosine similarity over an in-memory matrix of normalized vectors."""

    name = "numpy-exact"

    def __init__(self):
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._matrix: np.ndarray | None = None

    def add(self, ids: Sequence[str], texts: Sequence[str], vectors: np.ndarray) -> None:
        if len(ids) != len(texts) or len(ids) != vectors.shape[0]:
            raise ValueError("ids, texts and vectors must be the same length")
        self._ids.extend(ids)
        self._texts.extend(texts)
        block = np.asarray(vectors, dtype=np.float32)
        self._matrix = block if self._matrix is None else np.vstack([self._matrix, block])

    def search(self, query_vector: np.ndarray, top_k: int) -> list[Candidate]:
        if self._matrix is None:
            return []
        k = max(0, min(top_k, len(self._ids)))
        if k == 0:
            return []

        sims = self._matrix @ np.asarray(query_vector, dtype=np.float32)
        # argpartition finds the top-k without sorting all N, then sort just those.
        top_unsorted = np.argpartition(-sims, k - 1)[:k]
        top_idx = top_unsorted[np.argsort(-sims[top_unsorted])]

        return _as_candidates(
            [self._ids[i] for i in top_idx],
            [self._texts[i] for i in top_idx],
            [sims[i] for i in top_idx],
        )

    def __len__(self):
        return len(self._ids)


class MilvusVectorStore:
    """Real Milvus-backed store (Milvus Lite file, or a server URI).

    Milvus's primary key is an integer here, with the human-readable document id
    carried as a payload field. `metric_type="COSINE"` matches the L2-normalized
    vectors the embedders emit.
    """

    def __init__(
        self,
        dimension: int,
        uri: str = "milvus_healthcare.db",
        collection_name: str = "healthcare_docs",
        token: str | None = None,
        recreate: bool = True,
    ):
        from pymilvus import MilvusClient

        self.name = f"milvus[{uri}]"
        self.collection_name = collection_name
        self.dimension = dimension
        self.client = MilvusClient(uri=uri, token=token) if token else MilvusClient(uri=uri)

        if recreate and self.client.has_collection(collection_name):
            self.client.drop_collection(collection_name)
        if not self.client.has_collection(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                dimension=dimension,
                metric_type="COSINE",
                auto_id=False,
            )
        self._next_pk = 0

    def add(self, ids: Sequence[str], texts: Sequence[str], vectors: np.ndarray) -> None:
        rows = []
        for doc_id, text, vector in zip(ids, texts, vectors):
            rows.append(
                {
                    "id": self._next_pk,
                    "vector": np.asarray(vector, dtype=np.float32).tolist(),
                    "doc_id": doc_id,
                    "text": text,
                }
            )
            self._next_pk += 1
        if rows:
            self.client.insert(collection_name=self.collection_name, data=rows)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[Candidate]:
        if top_k <= 0:
            return []
        results = self.client.search(
            collection_name=self.collection_name,
            data=[np.asarray(query_vector, dtype=np.float32).tolist()],
            limit=top_k,
            output_fields=["doc_id", "text"],
        )
        hits = results[0] if results else []
        ids, texts, scores = [], [], []
        for hit in hits:
            entity = hit.get("entity", hit)
            ids.append(entity["doc_id"])
            texts.append(entity["text"])
            # COSINE distance in Milvus is the similarity itself (higher is better).
            scores.append(hit["distance"])
        return _as_candidates(ids, texts, scores)

    def close(self):
        self.client.close()
