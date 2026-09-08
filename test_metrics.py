"""
Hand-checked tests for the metrics and the pipeline contracts.

Every headline number in the README is produced by `metrics.py`, so these are
worked examples with values computed by hand rather than snapshots of whatever
the code currently returns. Run with:

    python3 test_metrics.py
"""

import math

import metrics as M
from contracts import Candidate
from pipeline import RAGPipeline
from retrieval import Bm25Index, Bm25Retriever, IdentityReranker

RANKED = ["A", "B", "C", "D", "E"]
GRADED = {"C": 2, "A": 1}  # A is partially relevant, C directly answers


def approx(value, expected, tol=1e-9):
    assert abs(value - expected) < tol, f"expected {expected}, got {value}"


def test_precision_recall():
    # Binary relevance = gain >= 1, so both A and C count.
    approx(M.precision_at_k(RANKED, GRADED, 1), 1 / 1)   # A is relevant
    approx(M.precision_at_k(RANKED, GRADED, 3), 2 / 3)   # A and C in top 3
    approx(M.precision_at_k(RANKED, GRADED, 5), 2 / 5)
    approx(M.recall_at_k(RANKED, GRADED, 1), 1 / 2)      # 1 of 2 relevant found
    approx(M.recall_at_k(RANKED, GRADED, 3), 2 / 2)
    # Precision@k is bounded by n_relevant/k - the reason it is a poor headline
    # metric on this eval set rather than a retriever-quality signal.
    approx(M.precision_at_k(RANKED, {"A": 2}, 5), 1 / 5)


def test_hit_and_primary_hit():
    approx(M.hit_at_k(RANKED, GRADED, 1), 1.0)
    # A (gain 1) is at rank 1, but the directly-answering passage C is not in
    # the top 1 - primary_hit@1 must not be fooled by adjacent context.
    approx(M.primary_hit_at_k(RANKED, GRADED, 1), 0.0)
    approx(M.primary_hit_at_k(RANKED, GRADED, 3), 1.0)


def test_reciprocal_rank():
    approx(M.reciprocal_rank(RANKED, GRADED), 1.0)              # A at rank 1
    approx(M.reciprocal_rank(RANKED, {"C": 2}), 1 / 3)          # C at rank 3
    approx(M.reciprocal_rank(RANKED, {"Z": 2}), 0.0)            # not retrieved
    # Beyond the cutoff counts as a miss.
    approx(M.reciprocal_rank(RANKED, {"E": 2}, k=3), 0.0)
    approx(M.reciprocal_rank(RANKED, {"E": 2}, k=5), 1 / 5)


def test_dcg_and_ndcg():
    # DCG@3 = gain(A)/log2(2) + gain(C)/log2(4)
    #       = (2**1 - 1)/1 + (2**2 - 1)/2 = 1 + 1.5 = 2.5
    approx(M.dcg_at_k(RANKED, GRADED, 3), 2.5)
    # Ideal ordering is C then A: (2**2 - 1)/1 + (2**1 - 1)/log2(3)
    ideal = 3.0 + 1.0 / math.log2(3)
    approx(M.ndcg_at_k(RANKED, GRADED, 3), 2.5 / ideal)
    # A perfect ranking scores exactly 1.0.
    approx(M.ndcg_at_k(["C", "A"], GRADED, 3), 1.0)
    # Graded labels are the point: same binary hits, better ordering, higher NDCG.
    assert M.ndcg_at_k(["C", "A"], GRADED, 5) > M.ndcg_at_k(["A", "C"], GRADED, 5)


def test_unanswerable_metrics_are_nan():
    # Recall is undefined with no relevant document; it must be NaN rather than
    # a silent 0.0 that would drag the mean down for a correct abstention.
    assert math.isnan(M.recall_at_k(RANKED, {}, 5))
    assert math.isnan(M.ndcg_at_k(RANKED, {}, 5))
    assert math.isnan(M.reciprocal_rank(RANKED, {}))


def test_aggregate_ignores_nan():
    scores = [{"recall@5": 1.0}, {"recall@5": float("nan")}, {"recall@5": 0.0}]
    approx(M.aggregate(scores)["recall@5"], 0.5)


def test_paired_bootstrap_detects_no_difference():
    values = [0.5] * 40
    mean_diff, low, high, _ = M.paired_bootstrap(values, values, n_resamples=500)
    approx(mean_diff, 0.0)
    approx(low, 0.0)
    approx(high, 0.0)


def test_bm25_idf_is_corpus_level():
    """A document's score must not depend on which other docs were retrieved.

    This is the bug the old candidate-set-local BM25 had: identical passage,
    identical query, different score depending on `top_k`.
    """
    corpus = [
        {"id": "X1", "text": "metformin is first-line therapy for type 2 diabetes"},
        {"id": "X2", "text": "insulin is first-line therapy for type 1 diabetes"},
        {"id": "X3", "text": "hand hygiene prevents infection in clinical settings"},
    ]
    index = Bm25Index([d["id"] for d in corpus], [d["text"] for d in corpus])
    score_alone = index.score_by_id("metformin diabetes", "X1")
    # Same index, same document, same query -> identical score regardless of
    # how many candidates a caller happens to pass to the reranker.
    from retrieval import Bm25Reranker

    reranker = Bm25Reranker(index=index)
    candidates_small = [Candidate(id="X1", text=corpus[0]["text"], score=0.0)]
    candidates_large = [
        Candidate(id=d["id"], text=d["text"], score=0.0) for d in corpus
    ]
    small = reranker.rerank("metformin diabetes", candidates_small)[0].rerank_score
    large = next(
        c for c in reranker.rerank("metformin diabetes", candidates_large) if c.id == "X1"
    ).rerank_score
    approx(small, large)
    approx(small, score_alone)


def test_pipeline_rejects_context_k_above_retrieval_k():
    corpus = [{"id": "X1", "topic": "t", "text": "aspirin prevents stroke recurrence"}]
    retriever = Bm25Retriever(corpus)
    try:
        RAGPipeline(retriever, IdentityReranker(), retrieval_k=3, context_k=5)
    except ValueError:
        pass
    else:
        raise AssertionError("context_k > retrieval_k must be rejected")


def test_pipeline_handles_empty_and_failing_components():
    class DeadRetriever:
        name = "dead"

        def retrieve(self, query_text, top_k):
            raise RuntimeError("index unavailable")

    class DeadReranker:
        name = "dead-reranker"

        def rerank(self, query_text, candidates):
            raise RuntimeError("model unavailable")

    # A dead first stage degrades to an abstention, not a crash.
    result = RAGPipeline(DeadRetriever(), retrieval_k=5, context_k=3).run(
        "anything", generate=True
    )
    assert result.degraded and result.generation["abstained"]
    assert "retrieval failed" in result.warnings[0]

    # A dead reranker degrades to first-stage order and still answers.
    corpus = [
        {"id": "X1", "topic": "t", "text": "aspirin prevents stroke recurrence"},
        {"id": "X2", "topic": "t", "text": "statins lower LDL cholesterol"},
    ]
    pipeline = RAGPipeline(
        Bm25Retriever(corpus), DeadReranker(), retrieval_k=2, context_k=1
    )
    result = pipeline.run("aspirin stroke")
    assert result.degraded
    assert result.context_ids == ["X1"], result.context_ids


def test_abstention_gate_fires_below_threshold():
    from generation import ExtractiveLLM

    corpus = [{"id": "X1", "topic": "t", "text": "aspirin prevents stroke recurrence"}]
    pipeline = RAGPipeline(
        Bm25Retriever(corpus),
        IdentityReranker(),
        llm=ExtractiveLLM(),
        retrieval_k=1,
        context_k=1,
        abstain_below=1e6,  # nothing can clear this
    )
    result = pipeline.run("aspirin stroke", generate=True)
    assert result.generation["abstained"]
    assert result.generation["retrieval_gated"]
    assert result.generation["cited_ids"] == []


def test_extractive_llm_cites_from_context():
    from generation import ExtractiveLLM

    context = [Candidate(id="X1", text="Metformin is first-line for type 2 diabetes.", score=5.0)]
    out = ExtractiveLLM().generate("first-line for type 2 diabetes", context)
    assert out["cited_ids"] == ["X1"]
    assert not out["abstained"]
    assert "X1" in out["answer"]


def test_groundedness_catches_unsupported_claims():
    from evaluation import groundedness

    passage = "Metformin is first-line pharmacologic therapy for type 2 diabetes."
    approx(groundedness("Metformin is first-line therapy.", [passage]), 1.0)
    # A dose that appears nowhere in the cited passage drags the score down.
    assert groundedness("Take 850 milligrams of ibuprofen twice daily.", [passage]) < 0.3


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
