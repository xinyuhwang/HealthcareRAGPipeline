"""
Component contracts for the pipeline.

This file exists so the responsibility boundaries are stated in one place
instead of being implied by whatever the concrete classes happen to do.

The boundary that matters most
-----------------------------
An earlier version of this project exposed:

    Embedder.embed_query(text) -> vector
    VectorStore.search(query_text, top_k)      # <- takes TEXT

That is incoherent: if the store accepts text, then the store owns an embedder
and the embedding stage is not a separate component at all. (In that version
the store literally held a reference to the embedder and reached into its
document matrix, so the two objects were one object with two names.)

The contract here instead is:

    vector      = embedder.embed_query(text)          # text  -> vector
    candidates  = store.search(vector, top_k)         # vector -> documents

A `VectorStore` never sees raw text at query time and owns no model. It stores
vectors and answers nearest-neighbour questions about them - exactly what
Milvus, FAISS, or pgvector do. This matters beyond tidiness: it is what makes
`MilvusVectorStore` a drop-in for `NumpyVectorStore`, since a real Milvus
server cannot embed text for you.

`Retriever` is the component that owns the two-step composition, which is why
it is a separate contract from `VectorStore`. A `Retriever` maps text to
candidates by whatever internal means it likes - `DenseRetriever` embeds and
then searches a vector store; `Bm25Retriever` scores an inverted index and
never touches a vector store at all. Because both satisfy the same contract,
the pipeline and the evaluation harness are indifferent to which is installed.
"""

from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence, runtime_checkable

import numpy as np


@dataclass
class Candidate:
    """One retrieved document, carried through every stage of the pipeline.

    `score` is always the score from the stage that produced this candidate,
    on that stage's own scale. Scores from different stages are NOT comparable
    (cosine similarity, BM25, and cross-encoder logits have unrelated ranges),
    so each stage records its own field and nothing is silently overwritten.
    """

    id: str
    text: str
    score: float
    retrieval_score: Optional[float] = None
    retrieval_rank: Optional[int] = None
    rerank_score: Optional[float] = None
    rerank_rank: Optional[int] = None
    meta: dict = field(default_factory=dict)

    def as_dict(self):
        return {
            "id": self.id,
            "text": self.text,
            "score": self.score,
            "retrieval_score": self.retrieval_score,
            "retrieval_rank": self.retrieval_rank,
            "rerank_score": self.rerank_score,
            "rerank_rank": self.rerank_rank,
        }


@runtime_checkable
class Embedder(Protocol):
    """Maps text to fixed-width vectors. Owns a model; owns no index."""

    name: str
    dimension: int

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, dimension) L2-normalized matrix, one row per text."""

    def embed_query(self, text: str) -> np.ndarray:
        """Return a single (dimension,) L2-normalized vector."""


@runtime_checkable
class VectorStore(Protocol):
    """Stores vectors and answers nearest-neighbour queries. Owns no model."""

    name: str

    def add(self, ids: Sequence[str], texts: Sequence[str], vectors: np.ndarray) -> None:
        """Index `vectors`, keyed by `ids`, with `texts` retained for payload."""

    def search(self, query_vector: np.ndarray, top_k: int) -> list[Candidate]:
        """Return the `top_k` nearest documents to an already-embedded query."""


@runtime_checkable
class Retriever(Protocol):
    """First stage: text in, ranked candidates out. Owns how that happens."""

    name: str

    def retrieve(self, query_text: str, top_k: int) -> list[Candidate]:
        ...


@runtime_checkable
class Reranker(Protocol):
    """Second stage: re-scores an existing candidate set. Never adds documents.

    A reranker cannot improve what the first stage failed to retrieve - it can
    only reorder. That is precisely why `retrieval_k` must exceed `context_k`
    for a reranker to have any effect on what the LLM sees.
    """

    name: str

    def rerank(self, query_text: str, candidates: list[Candidate]) -> list[Candidate]:
        ...


@runtime_checkable
class LLM(Protocol):
    """Final stage: grounded answer generation over the selected context."""

    name: str

    def generate(self, query_text: str, context: list[Candidate]) -> dict:
        """Return {answer, cited_ids, abstained, ...}."""
