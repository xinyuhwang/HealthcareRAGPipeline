"""
Evaluation harness: retrieval quality, latency, and answer quality.

Three things are measured separately because they fail separately:

- `evaluate_retrieval` - did the right passage get found and ranked highly?
- `measure_latency`    - what did each stage cost, with warmup excluded?
- `evaluate_answers`   - did the generation stage cite honestly, and did it
                         decline when it had no source?

Slicing by query kind
---------------------
A single mean over a mixed query set hides the finding. The eval set tags each
query (`direct`, `paraphrase`, `distractor`, `multi_doc`), and results are
reported per slice as well as overall - because the whole argument for dense
retrieval lives in the `paraphrase` slice, and the whole argument for a
cross-encoder lives in the `distractor` slice.
"""

import re
from typing import Optional, Sequence

import numpy as np
import pandas as pd

import metrics as M


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def evaluate_retrieval(
    pipeline,
    queries,
    ks: Sequence[int] = M.DEFAULT_KS,
    retrieval_k: Optional[int] = None,
    context_k: Optional[int] = None,
    arch_name: Optional[str] = None,
) -> pd.DataFrame:
    """Per-query retrieval metrics over the pipeline's final ranking.

    Metrics are computed on the post-rerank ranking, so a reranker can change
    them - but only where `retrieval_k > context_k`, since @k for k <=
    retrieval_k is what actually reaches the context window.
    """
    rows = []
    for query in queries:
        if not query["answerable"]:
            continue
        result = pipeline.run(
            query["query"], retrieval_k=retrieval_k, context_k=context_k
        )
        scores = M.evaluate_ranking(result.ranked_ids, query["relevant"], ks=ks)

        row = {
            "architecture": arch_name or pipeline.name,
            "query": query["query"],
            "kind": query["kind"],
            "retrieval_k": result.retrieval_k,
            "context_k": result.context_k,
            "n_relevant": len(query["relevant"]),
            "top1_id": result.ranked_ids[0] if result.ranked_ids else None,
            "top_score": result.top_score,
            # Recall restricted to what the LLM actually receives. This is the
            # number that changes when you tune context_k, and the one that
            # bounds answer quality - anything not here cannot be cited.
            "context_recall": M.recall_at_k(
                result.context_ids, query["relevant"], result.context_k
            ),
            "context_primary_hit": M.primary_hit_at_k(
                result.context_ids, query["relevant"], result.context_k
            ),
        }
        row.update(scores)
        rows.append(row)
    return pd.DataFrame(rows)


METRIC_COLUMNS = [
    "recall@1",
    "recall@3",
    "recall@5",
    "recall@10",
    "primary_hit@1",
    "mrr@10",
    "ndcg@5",
    "ndcg@10",
    "precision@5",
    "context_recall",
    "context_primary_hit",
]


def summarize(per_query: pd.DataFrame, columns: Sequence[str] = METRIC_COLUMNS) -> pd.Series:
    present = [c for c in columns if c in per_query.columns]
    return per_query[present].mean(numeric_only=True)


def summarize_by_kind(per_query: pd.DataFrame, columns: Sequence[str] = METRIC_COLUMNS) -> pd.DataFrame:
    present = [c for c in columns if c in per_query.columns]
    return per_query.groupby("kind")[present].mean(numeric_only=True)


def compare_arms(per_query_a: pd.DataFrame, per_query_b: pd.DataFrame, metric: str,
                 n_resamples: int = 10_000) -> dict:
    """Paired bootstrap between two arms evaluated on the same queries."""
    merged = per_query_a.merge(
        per_query_b, on="query", suffixes=("_a", "_b"), validate="one_to_one"
    )
    mean_diff, low, high, favor = M.paired_bootstrap(
        merged[f"{metric}_a"].to_numpy(),
        merged[f"{metric}_b"].to_numpy(),
        n_resamples=n_resamples,
    )
    return {
        "metric": metric,
        "mean_a": float(merged[f"{metric}_a"].mean()),
        "mean_b": float(merged[f"{metric}_b"].mean()),
        "difference": mean_diff,
        "ci_low": low,
        "ci_high": high,
        "p_favors_a": favor,
        "significant": bool(low > 0 or high < 0),
    }


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------
def measure_latency(
    pipeline,
    queries,
    repeats: int = 5,
    warmup: int = 3,
    retrieval_k: Optional[int] = None,
    context_k: Optional[int] = None,
) -> dict:
    """Per-stage latency, in milliseconds, with model warmup discarded.

    Warmup matters here: the first call to a torch model pays lazy kernel
    initialization and can be an order of magnitude slower than steady state,
    which would otherwise land entirely in whichever arm ran first.
    """
    texts = [q["query"] for q in queries]
    for text in texts[: max(warmup, 1)]:
        pipeline.run(text, retrieval_k=retrieval_k, context_k=context_k)

    samples = {"retrieve": [], "rerank": [], "total": []}
    for _ in range(repeats):
        for text in texts:
            result = pipeline.run(text, retrieval_k=retrieval_k, context_k=context_k)
            for stage in samples:
                if stage in result.timings_ms:
                    samples[stage].append(result.timings_ms[stage])

    out = {}
    for stage, values in samples.items():
        if not values:
            continue
        out[f"{stage}_mean_ms"] = float(np.mean(values))
        out[f"{stage}_p50_ms"] = float(np.percentile(values, 50))
        out[f"{stage}_p95_ms"] = float(np.percentile(values, 95))
    out["n_samples"] = len(samples["total"])
    return out


# ---------------------------------------------------------------------------
# Abstention calibration
# ---------------------------------------------------------------------------
def split_queries(queries, seed: int = 0, calibration_fraction: float = 0.5):
    """Stratified split into calibration and test halves.

    The abstention threshold is a fitted parameter, so it must not be tuned on
    the queries it is then scored on. Stratifying by `kind` keeps the
    unanswerable queries balanced across both halves.
    """
    rng = np.random.default_rng(seed)
    calibration, test = [], []
    by_kind: dict[str, list] = {}
    for query in queries:
        by_kind.setdefault(query["kind"], []).append(query)

    for kind_queries in by_kind.values():
        order = rng.permutation(len(kind_queries))
        cut = int(round(len(kind_queries) * calibration_fraction))
        for position, index in enumerate(order):
            (calibration if position < cut else test).append(kind_queries[index])
    return calibration, test


def score_distribution(pipeline, queries, retrieval_k=None, context_k=None) -> pd.DataFrame:
    """Top-context score for each query, tagged with whether it is answerable."""
    rows = []
    for query in queries:
        result = pipeline.run(query["query"], retrieval_k=retrieval_k, context_k=context_k)
        rows.append(
            {
                "query": query["query"],
                "kind": query["kind"],
                "answerable": query["answerable"],
                "top_score": result.top_score,
            }
        )
    return pd.DataFrame(rows)


def calibrate_abstention(pipeline, calibration_queries, retrieval_k=None, context_k=None) -> dict:
    """Pick the retrieval-score threshold that best separates answerable from not.

    Chooses the threshold maximizing balanced accuracy, so the 10 unanswerable
    queries are not swamped by the 85 answerable ones. Score scales differ by
    architecture (cosine vs BM25 vs cross-encoder logits), which is exactly why
    this is calibrated per pipeline instead of hard-coded.
    """
    distribution = score_distribution(
        pipeline, calibration_queries, retrieval_k=retrieval_k, context_k=context_k
    )
    distribution = distribution.dropna(subset=["top_score"])
    answerable = distribution[distribution["answerable"]]["top_score"].to_numpy()
    unanswerable = distribution[~distribution["answerable"]]["top_score"].to_numpy()

    if len(answerable) == 0 or len(unanswerable) == 0:
        return {"threshold": None, "balanced_accuracy": float("nan")}

    candidates = np.unique(np.concatenate([answerable, unanswerable]))
    midpoints = np.concatenate(
        [[candidates[0] - 1e-6], (candidates[:-1] + candidates[1:]) / 2, [candidates[-1] + 1e-6]]
    )

    best = {"threshold": None, "balanced_accuracy": -1.0}
    for threshold in midpoints:
        # Answer when score >= threshold; abstain below it.
        true_positive_rate = float((answerable >= threshold).mean())
        true_negative_rate = float((unanswerable < threshold).mean())
        balanced = (true_positive_rate + true_negative_rate) / 2
        if balanced > best["balanced_accuracy"]:
            best = {
                "threshold": float(threshold),
                "balanced_accuracy": balanced,
                "answer_rate_answerable": true_positive_rate,
                "abstain_rate_unanswerable": true_negative_rate,
            }
    best["n_calibration"] = int(len(distribution))
    return best


# ---------------------------------------------------------------------------
# Answer quality
# ---------------------------------------------------------------------------
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "is", "are", "was",
    "were", "be", "with", "on", "at", "by", "it", "that", "this", "as", "from",
    "not", "no", "any", "can", "may", "should", "have", "has", "than", "then",
    "there", "their", "you", "your", "i", "we", "but", "if", "so", "do", "does",
}


def _content_tokens(text: str) -> set:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(token) > 2 and token not in _STOPWORDS
    }


def groundedness(answer: str, cited_texts: Sequence[str]) -> float:
    """Share of the answer's content tokens that appear in its cited passages.

    A cheap, deterministic faithfulness proxy - not a substitute for an LLM
    judge or human review, but it catches the failure that matters most here:
    an answer asserting specifics (a drug, a dose, a threshold) that appear
    nowhere in the passage it claims to be citing.
    """
    answer_tokens = _content_tokens(answer)
    if not answer_tokens:
        return float("nan")
    supported = set()
    for text in cited_texts:
        supported |= _content_tokens(text)
    return len(answer_tokens & supported) / len(answer_tokens)


def evaluate_answers(
    pipeline,
    queries,
    retrieval_k: Optional[int] = None,
    context_k: Optional[int] = None,
    arch_name: Optional[str] = None,
) -> pd.DataFrame:
    """Per-query answer-level evaluation, including abstention behaviour.

    Columns:
      abstained            - did the system decline?
      should_abstain       - was declining the correct behaviour?
      citation_valid       - every cited id was actually in the context window
      citation_correct     - at least one cited id is a gold gain-2 passage
      groundedness         - token-level support of the answer by its citations
      answered_correctly   - answered, cited validly, and cited the right passage
    """
    corpus_text = {}
    rows = []
    for query in queries:
        result = pipeline.run(
            query["query"],
            retrieval_k=retrieval_k,
            context_k=context_k,
            generate=True,
        )
        generation = result.generation or {}
        cited_ids = list(generation.get("cited_ids") or [])
        context_ids = set(result.context_ids)
        for candidate in result.context:
            corpus_text[candidate.id] = candidate.text

        abstained = bool(generation.get("abstained"))
        should_abstain = not query["answerable"]
        primary = {d for d, g in query["relevant"].items() if g == 2}

        citation_valid = all(cited in context_ids for cited in cited_ids) if cited_ids else None
        citation_correct = (
            bool(primary & set(cited_ids)) if (cited_ids and primary) else None
        )
        cited_texts = [corpus_text.get(cited, "") for cited in cited_ids]

        rows.append(
            {
                "architecture": arch_name or pipeline.name,
                "query": query["query"],
                "kind": query["kind"],
                "answerable": query["answerable"],
                "abstained": abstained,
                "should_abstain": should_abstain,
                "abstain_reason": generation.get("abstain_reason"),
                "n_cited": len(cited_ids),
                "cited_ids": ",".join(cited_ids),
                "citation_valid": citation_valid,
                "citation_correct": citation_correct,
                "groundedness": (
                    groundedness(generation.get("answer", ""), cited_texts)
                    if cited_ids and not abstained
                    else float("nan")
                ),
                "answered_correctly": (
                    (not abstained)
                    and bool(citation_valid)
                    and bool(citation_correct)
                    if query["answerable"]
                    else abstained
                ),
                "answer": generation.get("answer", ""),
                "degraded": result.degraded,
                "warnings": "; ".join(result.warnings),
            }
        )
    return pd.DataFrame(rows)


def summarize_answers(per_query: pd.DataFrame) -> dict:
    """Headline answer-quality numbers, keeping answerable and unanswerable apart."""
    answerable = per_query[per_query["answerable"]]
    unanswerable = per_query[~per_query["answerable"]]

    def _mean(frame, column):
        values = frame[column].dropna()
        return float(values.mean()) if len(values) else float("nan")

    out = {
        "n_answerable": int(len(answerable)),
        "n_unanswerable": int(len(unanswerable)),
        # On answerable queries: did it answer, cite legally, and cite correctly?
        "answer_rate": 1.0 - _mean(answerable, "abstained"),
        "citation_validity": _mean(answerable, "citation_valid"),
        "citation_accuracy": _mean(answerable, "citation_correct"),
        "groundedness": _mean(answerable, "groundedness"),
        "answered_correctly": _mean(answerable, "answered_correctly"),
        # On unanswerable queries: abstention is the only correct behaviour.
        "abstention_recall": _mean(unanswerable, "abstained"),
        # False abstention: declined a question it did have a source for.
        "false_abstention_rate": _mean(answerable, "abstained"),
    }
    hallucination_risk = unanswerable[~unanswerable["abstained"]]
    out["unsupported_answer_rate"] = (
        float(len(hallucination_risk) / len(unanswerable)) if len(unanswerable) else float("nan")
    )
    return out
