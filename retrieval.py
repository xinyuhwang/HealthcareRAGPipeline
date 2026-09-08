"""
Retrievers (first stage) and rerankers (second stage).

Terminology note - BM25 is not a neural reranker
------------------------------------------------
BM25 is a lexical retrieval/scoring function. It can legitimately be *used* as
a second-stage scorer over a candidate set, and that is what `Bm25Reranker`
does, but calling it "the reranker" invites the obvious objection: it is not
doing what a cross-encoder does. A cross-encoder jointly encodes the
(query, passage) pair with full cross-attention and can resolve paraphrase and
negation; BM25 counts weighted term matches and cannot.

So this module names the two things separately and the experiment reports them
as separate arms:

    Bm25Reranker          -> "lexical second-stage reranking baseline"
    CrossEncoderReranker  -> "cross-encoder reranker"

Reporting the lexical arm is worthwhile precisely because it is cheap: it sets
the bar the neural reranker has to clear to justify its cost. Hiding it would
also hide the more interesting result, which is how much of the reranking gain
is available for microseconds instead of milliseconds.

One BM25 index, two uses
------------------------
`Bm25Index` computes document frequencies over the *whole corpus* once. Both
the first-stage `Bm25Retriever` and the second-stage `Bm25Reranker` score
against those global statistics. An earlier version computed IDF over only the
top-k candidate set, which made a document's score depend on how many other
documents happened to be retrieved alongside it - so the same passage scored
differently at k=3 and k=10, and the "reranker" was partly measuring k.
"""

import re
from typing import Sequence

import numpy as np

from contracts import Candidate

_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------
class Bm25Index:
    """Okapi BM25 with corpus-level document frequencies."""

    def __init__(self, doc_ids: Sequence[str], doc_texts: Sequence[str], k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids = list(doc_ids)
        self.doc_texts = list(doc_texts)
        self.doc_tokens = [tokenize(t) for t in doc_texts]
        self.doc_lengths = np.array([max(len(t), 1) for t in self.doc_tokens], dtype=np.float32)
        self.avg_length = float(self.doc_lengths.mean()) if len(self.doc_lengths) else 1.0
        self.n_docs = len(self.doc_ids)

        self.term_frequencies: list[dict[str, int]] = []
        document_frequency: dict[str, int] = {}
        for tokens in self.doc_tokens:
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            self.term_frequencies.append(counts)
            for token in counts:
                document_frequency[token] = document_frequency.get(token, 0) + 1

        self.idf = {
            token: float(np.log(1 + (self.n_docs - df + 0.5) / (df + 0.5)))
            for token, df in document_frequency.items()
        }
        self._index_by_id = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}

    def score_document(self, query_tokens: Sequence[str], doc_index: int) -> float:
        counts = self.term_frequencies[doc_index]
        doc_length = self.doc_lengths[doc_index]
        length_norm = self.k1 * (1 - self.b + self.b * doc_length / self.avg_length)
        score = 0.0
        for token in query_tokens:
            frequency = counts.get(token)
            if not frequency:
                continue
            score += self.idf.get(token, 0.0) * (frequency * (self.k1 + 1)) / (frequency + length_norm)
        return float(score)

    def score_all(self, query_text: str) -> np.ndarray:
        query_tokens = tokenize(query_text)
        return np.array(
            [self.score_document(query_tokens, i) for i in range(self.n_docs)],
            dtype=np.float32,
        )

    def index_of(self, doc_id: str) -> int | None:
        """Row index for a document id, or None if it is not in this index."""
        return self._index_by_id.get(doc_id)

    def score_by_id(self, query_text: str, doc_id: str) -> float:
        return self.score_document(tokenize(query_text), self._index_by_id[doc_id])


# ---------------------------------------------------------------------------
# First stage: retrievers
# ---------------------------------------------------------------------------
class DenseRetriever:
    """Embed the query, then search the vector store.

    This class owns the two-step composition that used to be smeared across the
    embedder and the store. The store never sees text; the embedder never sees
    the index.
    """

    def __init__(self, embedder, store, corpus=None):
        self.embedder = embedder
        self.store = store
        self.name = f"dense({embedder.name})"
        if corpus is not None:
            self.index_corpus(corpus)

    def index_corpus(self, corpus) -> None:
        ids = [d["id"] for d in corpus]
        texts = [d["text"] for d in corpus]
        vectors = self.embedder.embed_documents(texts)
        self.store.add(ids, texts, vectors)

    def embed_query(self, query_text: str) -> np.ndarray:
        return self.embedder.embed_query(query_text)

    def search_vector(self, query_vector: np.ndarray, top_k: int) -> list[Candidate]:
        return self.store.search(query_vector, top_k)

    def retrieve(self, query_text: str, top_k: int) -> list[Candidate]:
        query_vector = self.embedder.embed_query(query_text)
        return self.store.search(query_vector, top_k)


class Bm25Retriever:
    """First-stage lexical retrieval over the full corpus (no vectors at all)."""

    name = "bm25-lexical"

    def __init__(self, corpus):
        self.index = Bm25Index([d["id"] for d in corpus], [d["text"] for d in corpus])

    def retrieve(self, query_text: str, top_k: int) -> list[Candidate]:
        scores = self.index.score_all(query_text)
        k = max(0, min(top_k, len(scores)))
        if k == 0:
            return []
        top_unsorted = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_unsorted[np.argsort(-scores[top_unsorted])]
        return [
            Candidate(
                id=self.index.doc_ids[i],
                text=self.index.doc_texts[i],
                score=float(scores[i]),
                retrieval_score=float(scores[i]),
                retrieval_rank=rank,
            )
            for rank, i in enumerate(top_idx, start=1)
        ]


# ---------------------------------------------------------------------------
# Second stage: rerankers
# ---------------------------------------------------------------------------
def _apply_ranking(candidates: list[Candidate], scores: Sequence[float]) -> list[Candidate]:
    for candidate, score in zip(candidates, scores):
        candidate.rerank_score = float(score)
    ordered = sorted(candidates, key=lambda c: c.rerank_score, reverse=True)
    for rank, candidate in enumerate(ordered, start=1):
        candidate.rerank_rank = rank
        candidate.score = candidate.rerank_score
    return ordered


class IdentityReranker:
    """No second stage. Keeps first-stage order; used as the ablation control."""

    name = "none"

    def rerank(self, query_text: str, candidates: list[Candidate]) -> list[Candidate]:
        for candidate in candidates:
            candidate.rerank_score = candidate.retrieval_score
            candidate.rerank_rank = candidate.retrieval_rank
        return list(candidates)


class Bm25Reranker:
    """Lexical second-stage reranking baseline.

    Scores the candidate set with BM25 using corpus-level statistics. Deliberately
    *not* described as equivalent to a neural reranker - it is the cheap lower
    bound on what second-stage scoring can achieve.
    """

    name = "bm25-lexical-second-stage"

    def __init__(self, corpus=None, index: Bm25Index | None = None):
        if index is None:
            if corpus is None:
                raise ValueError("Bm25Reranker needs either a corpus or a prebuilt index")
            index = Bm25Index([d["id"] for d in corpus], [d["text"] for d in corpus])
        self.index = index

    def rerank(self, query_text: str, candidates: list[Candidate]) -> list[Candidate]:
        if not candidates:
            return []
        query_tokens = tokenize(query_text)
        scores = []
        for candidate in candidates:
            row = self.index.index_of(candidate.id)
            # A candidate from outside this index (e.g. a differently-chunked
            # corpus) scores 0 rather than raising - the reranker must not be
            # the thing that fails a query.
            scores.append(
                self.index.score_document(query_tokens, row) if row is not None else 0.0
            )
        return _apply_ranking(list(candidates), scores)


class CrossEncoderReranker:
    """Cross-encoder reranker (joint query-passage encoding).

    Unlike BM25, this model sees the query and passage together and attends
    across them, which is what lets it handle paraphrase and negation. The cost
    is one forward pass per candidate - so latency is linear in `retrieval_k`,
    which is the tradeoff the ablation is designed to expose.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", batch_size: int = 32):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.name = f"cross-encoder[{model_name.split('/')[-1]}]"
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name)

    def rerank(self, query_text: str, candidates: list[Candidate]) -> list[Candidate]:
        if not candidates:
            return []
        pairs = [(query_text, c.text) for c in candidates]
        scores = self.model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )
        return _apply_ranking(list(candidates), scores)
