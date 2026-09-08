"""
Retrieval metrics.

Every function takes a ranked list of document ids (best first) and the graded
relevance mapping for that query (`{doc_id: gain}`), so the same code path
serves both the binary metrics and NDCG.

Why these metrics
-----------------
Precision@k alone is misleading on this eval set. Most queries have one or two
relevant passages, so precision@10 is arithmetically capped near 0.1 no matter
how good the retriever is - a "precision drop" from k=3 to k=10 measures the
label distribution, not the retriever. The metrics below separate the questions
that actually matter:

- Recall@k     : did the relevant material reach the context window at all?
- Recall@1     : is the single best passage already on top? (the strictest
                 useful signal, and the one a reranker is supposed to move)
- MRR          : how far down does the user/LLM have to read to hit something
                 relevant? Rewards putting one good passage first.
- NDCG@k       : rank-weighted and grade-aware - retrieving the passage that
                 *directly answers* the question beats retrieving one that is
                 merely adjacent. This is why the labels are graded 2/1.
- Precision@k  : still reported, but read as a context-efficiency measure (how
                 much of the LLM's context is wasted), not as retriever skill.
"""

import math

import numpy as np

from data import RELEVANT_THRESHOLD


def _relevant_ids(relevance, threshold=RELEVANT_THRESHOLD):
    return {doc_id for doc_id, gain in relevance.items() if gain >= threshold}


def precision_at_k(ranked_ids, relevance, k):
    """Fraction of the top-k slots filled with relevant documents."""
    if k <= 0:
        return 0.0
    relevant = _relevant_ids(relevance)
    hits = sum(1 for doc_id in ranked_ids[:k] if doc_id in relevant)
    return hits / k


def recall_at_k(ranked_ids, relevance, k):
    """Fraction of all relevant documents that appear in the top-k."""
    relevant = _relevant_ids(relevance)
    if not relevant:
        return float("nan")  # undefined; caller should exclude unanswerable queries
    hits = sum(1 for doc_id in ranked_ids[:k] if doc_id in relevant)
    return hits / len(relevant)


def hit_at_k(ranked_ids, relevance, k):
    """1.0 if at least one relevant document is in the top-k, else 0.0."""
    relevant = _relevant_ids(relevance)
    if not relevant:
        return float("nan")
    return 1.0 if any(doc_id in relevant for doc_id in ranked_ids[:k]) else 0.0


def primary_hit_at_k(ranked_ids, relevance, k):
    """1.0 if a *directly answering* (gain-2) passage is in the top-k.

    Stricter than hit_at_k: retrieving merely-adjacent context does not count.
    """
    primary = _relevant_ids(relevance, threshold=2)
    if not primary:
        return float("nan")
    return 1.0 if any(doc_id in primary for doc_id in ranked_ids[:k]) else 0.0


def reciprocal_rank(ranked_ids, relevance, k=None):
    """1 / rank of the first relevant document (0.0 if none within k)."""
    relevant = _relevant_ids(relevance)
    if not relevant:
        return float("nan")
    candidates = ranked_ids if k is None else ranked_ids[:k]
    for rank, doc_id in enumerate(candidates, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def dcg_at_k(ranked_ids, relevance, k):
    """Discounted cumulative gain with exponential gains (2**g - 1)."""
    total = 0.0
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        gain = relevance.get(doc_id, 0)
        if gain:
            total += (2 ** gain - 1) / math.log2(rank + 1)
    return total


def ndcg_at_k(ranked_ids, relevance, k):
    """DCG normalized by the best achievable ordering of the labeled documents."""
    if not relevance:
        return float("nan")
    ideal_order = sorted(relevance, key=lambda d: relevance[d], reverse=True)
    ideal = dcg_at_k(ideal_order, relevance, k)
    if ideal == 0:
        return float("nan")
    return dcg_at_k(ranked_ids, relevance, k) / ideal


# ---------------------------------------------------------------------------
# Per-query metric bundle
# ---------------------------------------------------------------------------
DEFAULT_KS = (1, 3, 5, 10)


def evaluate_ranking(ranked_ids, relevance, ks=DEFAULT_KS, mrr_cutoff=10):
    """All retrieval metrics for a single query's ranking."""
    scores = {}
    for k in ks:
        scores[f"recall@{k}"] = recall_at_k(ranked_ids, relevance, k)
        scores[f"precision@{k}"] = precision_at_k(ranked_ids, relevance, k)
        scores[f"ndcg@{k}"] = ndcg_at_k(ranked_ids, relevance, k)
        scores[f"hit@{k}"] = hit_at_k(ranked_ids, relevance, k)
        scores[f"primary_hit@{k}"] = primary_hit_at_k(ranked_ids, relevance, k)
    scores[f"mrr@{mrr_cutoff}"] = reciprocal_rank(ranked_ids, relevance, mrr_cutoff)
    return scores


def aggregate(per_query_scores):
    """Macro-average each metric across queries, ignoring NaNs."""
    if not per_query_scores:
        return {}
    keys = per_query_scores[0].keys()
    out = {}
    for key in keys:
        values = [s[key] for s in per_query_scores if not math.isnan(s[key])]
        out[key] = float(np.mean(values)) if values else float("nan")
    return out


# ---------------------------------------------------------------------------
# Significance
# ---------------------------------------------------------------------------
def paired_bootstrap(values_a, values_b, n_resamples=10_000, seed=0):
    """Paired bootstrap on the mean difference (a - b) over the same queries.

    With ~85 queries, a difference of a couple of points is well inside noise.
    Reporting the interval keeps the ablation honest about which gaps are real.
    Returns (mean_difference, ci_low, ci_high, share_of_resamples_favoring_a).
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired bootstrap needs one value per query in both arms")

    mask = ~(np.isnan(a) | np.isnan(b))
    diff = a[mask] - b[mask]
    if diff.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_resamples, diff.size))
    means = diff[idx].mean(axis=1)
    return (
        float(diff.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
        float((means > 0).mean()),
    )
