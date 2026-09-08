"""
The RAG pipeline: retrieve -> (rerank) -> select context -> generate.

Two design points this file exists to get right.

1. `retrieval_k` and `context_k` are different numbers
-----------------------------------------------------
    retrieval_k : how many candidates the first stage fetches
    context_k   : how many documents actually reach the LLM's context window

Collapsing them into one `top_k` silently disables the second stage. If you
retrieve k documents, rerank those same k, and then evaluate at k, reordering
cannot change the set - recall@k and precision@k are mathematically identical
with the reranker on or off. (The previous version of this project did exactly
that, which is why its reranker "results" were unfalsifiable.)

The reranker earns its keep only in the gap between the two numbers: retrieve
20 cheaply, then spend the neural model on picking the best 5. Widening
`retrieval_k` raises the ceiling on what the reranker can find; `context_k`
controls how much of the LLM's context - and how much of the prompt bill - is
spent, and how much irrelevant text the model has to ignore.

2. Timing is per stage, and stages do not time themselves
---------------------------------------------------------
Each component does its job; the pipeline owns the clock. `total_ms` is the sum
of the stages that actually ran, and each stage appears exactly once, so the
number means the same thing whether or not generation is enabled.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from contracts import Candidate


@dataclass
class RunResult:
    """Everything one query produced, including why it may have declined."""

    query: str
    retrieved: list[Candidate] = field(default_factory=list)
    reranked: list[Candidate] = field(default_factory=list)
    context: list[Candidate] = field(default_factory=list)
    generation: Optional[dict] = None
    timings_ms: dict = field(default_factory=dict)
    retrieval_k: int = 0
    context_k: int = 0
    warnings: list[str] = field(default_factory=list)
    degraded: bool = False

    @property
    def context_ids(self) -> list[str]:
        return [c.id for c in self.context]

    @property
    def ranked_ids(self) -> list[str]:
        """Final ranking (post-rerank), used for retrieval metrics."""
        return [c.id for c in (self.reranked or self.retrieved)]

    @property
    def top_score(self) -> Optional[float]:
        ranked = self.reranked or self.retrieved
        return ranked[0].score if ranked else None


class RAGPipeline:
    """Composes a retriever, an optional reranker, and an optional LLM.

    Every component is injected. The pipeline knows the *shape* of the flow and
    nothing about dense vectors, BM25, or any particular model - which is what
    lets the experiment swap architectures without touching this file.
    """

    def __init__(
        self,
        retriever,
        reranker=None,
        llm=None,
        retrieval_k: int = 20,
        context_k: int = 5,
        abstain_below: Optional[float] = None,
    ):
        if context_k > retrieval_k:
            raise ValueError(
                f"context_k ({context_k}) cannot exceed retrieval_k ({retrieval_k}); "
                "the second stage can only reorder what the first stage fetched"
            )
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.retrieval_k = retrieval_k
        self.context_k = context_k
        # Score below which the pipeline declines rather than grounding an
        # answer in a weak match. Scale is stage-dependent, so this is
        # calibrated per architecture (see evaluation.calibrate_abstention).
        self.abstain_below = abstain_below

    @property
    def name(self) -> str:
        reranker_name = self.reranker.name if self.reranker else "none"
        return f"{self.retriever.name} + rerank:{reranker_name}"

    def run(
        self,
        query_text: str,
        retrieval_k: Optional[int] = None,
        context_k: Optional[int] = None,
        generate: bool = False,
    ) -> RunResult:
        retrieval_k = self.retrieval_k if retrieval_k is None else retrieval_k
        context_k = self.context_k if context_k is None else context_k
        if context_k > retrieval_k:
            raise ValueError("context_k cannot exceed retrieval_k")

        result = RunResult(query=query_text, retrieval_k=retrieval_k, context_k=context_k)

        # --- stage 1: retrieve -------------------------------------------
        started = time.perf_counter()
        try:
            result.retrieved = self.retriever.retrieve(query_text, retrieval_k)
        except Exception as exc:  # a dead index must not take the caller down
            result.warnings.append(f"retrieval failed: {type(exc).__name__}: {exc}")
            result.degraded = True
            result.retrieved = []
        result.timings_ms["retrieve"] = (time.perf_counter() - started) * 1000

        if not result.retrieved:
            result.warnings.append("no documents retrieved")
            if generate:
                result.generation = self._abstain("no documents were retrieved")
                result.timings_ms["generate"] = 0.0
            result.timings_ms["total"] = sum(
                v for k, v in result.timings_ms.items() if k != "total"
            )
            return result

        # --- stage 2: rerank ---------------------------------------------
        started = time.perf_counter()
        if self.reranker is not None:
            try:
                result.reranked = self.reranker.rerank(query_text, result.retrieved)
            except Exception as exc:
                # Degrade to first-stage order rather than failing the query:
                # a stale first-stage ranking beats no answer.
                result.warnings.append(f"rerank failed, using first-stage order: {exc}")
                result.degraded = True
                result.reranked = list(result.retrieved)
        else:
            result.reranked = list(result.retrieved)
        result.timings_ms["rerank"] = (time.perf_counter() - started) * 1000

        # --- context selection -------------------------------------------
        result.context = result.reranked[:context_k]

        # --- stage 3: generate -------------------------------------------
        if generate:
            started = time.perf_counter()
            result.generation = self._generate(query_text, result)
            result.timings_ms["generate"] = (time.perf_counter() - started) * 1000

        result.timings_ms["total"] = sum(
            v for k, v in result.timings_ms.items() if k != "total"
        )
        return result

    # ------------------------------------------------------------------
    def _generate(self, query_text: str, result: RunResult) -> dict:
        if self.llm is None:
            return self._abstain("no LLM configured")

        top_score = result.context[0].score if result.context else None
        if (
            self.abstain_below is not None
            and top_score is not None
            and top_score < self.abstain_below
        ):
            # Retrieval confidence gate: refuse before spending a generation
            # call on context we already believe is off-topic.
            result.warnings.append(
                f"top score {top_score:.3f} below abstention threshold {self.abstain_below:.3f}"
            )
            return self._abstain(
                "no sufficiently relevant passage was retrieved", gated=True
            )

        try:
            return self.llm.generate(query_text, result.context)
        except Exception as exc:
            result.warnings.append(f"generation failed: {type(exc).__name__}: {exc}")
            result.degraded = True
            return self._abstain(f"generation error: {type(exc).__name__}")

    @staticmethod
    def _abstain(reason: str, gated: bool = False) -> dict:
        return {
            "answer": (
                "I don't have a relevant source for that in the knowledge base, "
                "so I can't answer it."
            ),
            "cited_ids": [],
            "abstained": True,
            "abstain_reason": reason,
            "retrieval_gated": gated,
        }


def build_pipeline(
    corpus,
    retriever: str = "dense",
    reranker: str = "cross-encoder",
    llm=None,
    retrieval_k: int = 20,
    context_k: int = 5,
    embedder=None,
    store=None,
    abstain_below: Optional[float] = None,
) -> RAGPipeline:
    """Convenience factory for the named architectures used in the experiment.

    `retriever`: "dense" | "bm25"
    `reranker` : "none" | "bm25" | "cross-encoder"
    """
    from retrieval import (
        Bm25Reranker,
        Bm25Retriever,
        CrossEncoderReranker,
        DenseRetriever,
        IdentityReranker,
    )

    if retriever == "dense":
        if embedder is None:
            from embedding import DenseEmbedder

            embedder = DenseEmbedder()
        if store is None:
            from vector_store import NumpyVectorStore

            store = NumpyVectorStore()
        first_stage = DenseRetriever(embedder, store, corpus=corpus)
    elif retriever == "bm25":
        first_stage = Bm25Retriever(corpus)
    else:
        raise ValueError(f"unknown retriever {retriever!r}")

    if reranker in (None, "none"):
        second_stage = IdentityReranker()
    elif reranker == "bm25":
        second_stage = Bm25Reranker(corpus=corpus)
    elif reranker == "cross-encoder":
        second_stage = CrossEncoderReranker()
    else:
        raise ValueError(f"unknown reranker {reranker!r}")

    return RAGPipeline(
        retriever=first_stage,
        reranker=second_stage,
        llm=llm,
        retrieval_k=retrieval_k,
        context_k=context_k,
        abstain_below=abstain_below,
    )
