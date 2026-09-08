"""
Experiment driver: architecture comparison, ablations, latency, answer quality.

Run everything:

    python3 experiment.py

Faster subset (skips the retrieval_k sweep and the answer evaluation):

    python3 experiment.py --quick

Use a real Milvus Lite collection instead of the in-memory numpy store:

    python3 experiment.py --milvus

Outputs land in `results/` as CSVs, plus `results.csv` at the repo root as the
headline architecture table.
"""

import argparse
import os
import sys

import pandas as pd

import evaluation as E
import metrics as M
from data import CORPUS, QUERIES, answerable_queries, validate
from generation import default_llm
from pipeline import RAGPipeline
from retrieval import (
    Bm25Reranker,
    Bm25Retriever,
    CrossEncoderReranker,
    DenseRetriever,
    IdentityReranker,
)

RETRIEVAL_K = 20
CONTEXT_K = 5
RETRIEVAL_K_SWEEP = (5, 10, 20, 50)
CONTEXT_K_SWEEP = (1, 3, 5, 10)


# ---------------------------------------------------------------------------
# Component construction (built once and shared across arms)
# ---------------------------------------------------------------------------
def build_components(use_milvus: bool = False, embedding_model: str = "all-MiniLM-L6-v2"):
    from embedding import DenseEmbedder, TfidfEmbedder
    from vector_store import NumpyVectorStore

    print("Loading components ...", flush=True)
    dense_embedder = DenseEmbedder(embedding_model)

    if use_milvus:
        from vector_store import MilvusVectorStore

        store = MilvusVectorStore(dimension=dense_embedder.dimension)
        print(f"  vector store: {store.name}")
    else:
        store = NumpyVectorStore()
        print(f"  vector store: {store.name}")

    components = {
        "dense_retriever": DenseRetriever(dense_embedder, store, corpus=CORPUS),
        "bm25_retriever": Bm25Retriever(CORPUS),
        "tfidf_retriever": DenseRetriever(
            TfidfEmbedder([d["text"] for d in CORPUS]), NumpyVectorStore(), corpus=CORPUS
        ),
        "cross_encoder": CrossEncoderReranker(),
        "bm25_reranker": Bm25Reranker(corpus=CORPUS),
        "identity": IdentityReranker(),
    }
    print(f"  dense embedder: {dense_embedder.name} ({dense_embedder.dimension}-d)")
    print(f"  reranker: {components['cross_encoder'].name}")
    return components


def build_arms(components, retrieval_k=RETRIEVAL_K, context_k=CONTEXT_K):
    """The three architectures under comparison, plus the lexical-rerank ablation.

    Naming is deliberate: BM25 in second position is a *lexical second-stage
    reranking baseline*, not a peer of the cross-encoder. It is included because
    it is nearly free, and it sets the bar the neural reranker must clear.
    """
    def make(retriever, reranker):
        return RAGPipeline(
            retriever=retriever,
            reranker=reranker,
            retrieval_k=retrieval_k,
            context_k=context_k,
        )

    return {
        "A. Lexical retrieval (BM25 only)": make(
            components["bm25_retriever"], components["identity"]
        ),
        "B. Dense retrieval (no reranker)": make(
            components["dense_retriever"], components["identity"]
        ),
        "C. Dense + cross-encoder reranker": make(
            components["dense_retriever"], components["cross_encoder"]
        ),
        "D. Dense + lexical second-stage rerank (baseline)": make(
            components["dense_retriever"], components["bm25_reranker"]
        ),
    }


# ---------------------------------------------------------------------------
# 1. Architecture comparison
# ---------------------------------------------------------------------------
def run_architecture_comparison(arms, queries, retrieval_k=RETRIEVAL_K, context_k=CONTEXT_K):
    per_query_frames = {}
    summary_rows = []

    for name, pipeline in arms.items():
        print(f"  evaluating: {name}", flush=True)
        per_query = E.evaluate_retrieval(
            pipeline,
            queries,
            retrieval_k=retrieval_k,
            context_k=context_k,
            arch_name=name,
        )
        per_query_frames[name] = per_query
        row = {"architecture": name}
        row.update(E.summarize(per_query).to_dict())
        summary_rows.append(row)

    return pd.DataFrame(summary_rows), per_query_frames


def run_significance_tests(per_query_frames):
    """Paired bootstrap on the comparisons the architecture claims depend on."""
    names = list(per_query_frames)
    lexical, dense, cross_encoder, lexical_rerank = names
    comparisons = [
        ("dense vs lexical retrieval", dense, lexical),
        ("cross-encoder vs no reranker", cross_encoder, dense),
        ("cross-encoder vs lexical second stage", cross_encoder, lexical_rerank),
        ("lexical second stage vs no reranker", lexical_rerank, dense),
    ]
    rows = []
    for label, arm_a, arm_b in comparisons:
        # primary_hit@1 is included because it is where a reranker is supposed
        # to act: promoting the directly-answering passage to rank 1. recall@1
        # alone dilutes that with partially-relevant (gain-1) passages.
        for metric in ("recall@1", "primary_hit@1", "ndcg@5", "mrr@10", "context_recall"):
            result = E.compare_arms(
                per_query_frames[arm_a], per_query_frames[arm_b], metric
            )
            result["comparison"] = label
            result["arm_a"] = arm_a
            result["arm_b"] = arm_b
            rows.append(result)
    return pd.DataFrame(rows)[
        ["comparison", "metric", "mean_a", "mean_b", "difference",
         "ci_low", "ci_high", "significant", "arm_a", "arm_b"]
    ]


# ---------------------------------------------------------------------------
# 2. Ablations
# ---------------------------------------------------------------------------
def run_retrieval_k_ablation(components, queries, sweep=RETRIEVAL_K_SWEEP, context_k=CONTEXT_K):
    """Widen the first stage while holding the LLM's context window fixed.

    This is the ablation the old harness could not express: with
    retrieval_k == context_k the reranker cannot alter the context set at all,
    so every point at k=5 with context_k=5 is expected to collapse onto the
    no-reranker arm. The gap that opens as retrieval_k grows *is* the
    reranker's contribution.
    """
    rows = []
    reranker_arms = {
        "none": components["identity"],
        "lexical second stage (BM25)": components["bm25_reranker"],
        "cross-encoder": components["cross_encoder"],
    }
    for reranker_name, reranker in reranker_arms.items():
        for retrieval_k in sweep:
            pipeline = RAGPipeline(
                retriever=components["dense_retriever"],
                reranker=reranker,
                retrieval_k=retrieval_k,
                context_k=context_k,
            )
            per_query = E.evaluate_retrieval(
                pipeline, queries, retrieval_k=retrieval_k, context_k=context_k
            )
            latency = E.measure_latency(
                pipeline, queries[:20], repeats=2, warmup=2,
                retrieval_k=retrieval_k, context_k=context_k,
            )
            row = {
                "reranker": reranker_name,
                "retrieval_k": retrieval_k,
                "context_k": context_k,
            }
            row.update(E.summarize(per_query).to_dict())
            row["rerank_mean_ms"] = latency.get("rerank_mean_ms")
            row["total_mean_ms"] = latency.get("total_mean_ms")
            row["total_p95_ms"] = latency.get("total_p95_ms")
            rows.append(row)
            print(
                f"  retrieval_k={retrieval_k:>2} context_k={context_k} "
                f"reranker={reranker_name:<28} "
                f"context_recall={row['context_recall']:.3f} "
                f"recall@1={row['recall@1']:.3f} "
                f"total={row['total_mean_ms']:.1f}ms",
                flush=True,
            )
    return pd.DataFrame(rows)


def run_context_k_ablation(components, queries, sweep=CONTEXT_K_SWEEP, retrieval_k=RETRIEVAL_K):
    """Hold the first stage fixed and vary how much context the LLM receives.

    Separating this from retrieval_k is the point: context_k trades recall of
    relevant material against the share of the prompt spent on irrelevant text
    (precision@context_k), which is a cost and distraction question, not a
    retrieval-quality one.
    """
    rows = []
    for context_k in sweep:
        pipeline = RAGPipeline(
            retriever=components["dense_retriever"],
            reranker=components["cross_encoder"],
            retrieval_k=retrieval_k,
            context_k=context_k,
        )
        # Include context_k itself in the metric cutoffs so precision@context_k
        # is available for every point in the sweep.
        per_query = E.evaluate_retrieval(
            pipeline,
            queries,
            ks=tuple(sorted(set(M.DEFAULT_KS) | {context_k})),
            retrieval_k=retrieval_k,
            context_k=context_k,
        )
        row = {"retrieval_k": retrieval_k, "context_k": context_k}
        row.update(E.summarize(per_query).to_dict())
        # Share of the LLM's context slots occupied by relevant passages -
        # a prompt-cost measure, not a retriever-quality one.
        row["context_precision"] = float(per_query[f"precision@{context_k}"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def run_embedder_ablation(components, queries, retrieval_k=RETRIEVAL_K, context_k=CONTEXT_K):
    """TF-IDF vs a dense bi-encoder behind an otherwise identical pipeline.

    Returns the summary table plus the per-query frames, because the
    interesting question is not just which first stage wins but whether the
    cross-encoder's value depends on how good the first stage already is.
    """
    rows = []
    per_query_frames = {}
    for label, retriever in (
        ("tfidf vectors", components["tfidf_retriever"]),
        ("dense bi-encoder", components["dense_retriever"]),
    ):
        for reranker_label, reranker in (
            ("none", components["identity"]),
            ("cross-encoder", components["cross_encoder"]),
        ):
            pipeline = RAGPipeline(
                retriever=retriever,
                reranker=reranker,
                retrieval_k=retrieval_k,
                context_k=context_k,
            )
            per_query = E.evaluate_retrieval(
                pipeline, queries, retrieval_k=retrieval_k, context_k=context_k
            )
            per_query_frames[f"{label} + {reranker_label}"] = per_query
            row = {"first_stage_vectors": label, "reranker": reranker_label}
            row.update(E.summarize(per_query).to_dict())
            rows.append(row)
    return pd.DataFrame(rows), per_query_frames


def run_reranker_value_tests(embedder_frames):
    """Does the cross-encoder's value depend on first-stage quality?

    Same reranker, same k, two first stages of different strength. If the gain
    is large behind TF-IDF and negligible behind the dense retriever, then
    "add a reranker" is not architecture-independent advice.
    """
    pairs = [
        ("weak first stage (tfidf)", "tfidf vectors + cross-encoder", "tfidf vectors + none"),
        ("strong first stage (dense)", "dense bi-encoder + cross-encoder", "dense bi-encoder + none"),
    ]
    rows = []
    for label, arm_a, arm_b in pairs:
        for metric in ("recall@1", "primary_hit@1", "ndcg@5", "context_recall"):
            result = E.compare_arms(embedder_frames[arm_a], embedder_frames[arm_b], metric)
            result["comparison"] = f"cross-encoder gain, {label}"
            rows.append(result)
    return pd.DataFrame(rows)[
        ["comparison", "metric", "mean_a", "mean_b", "difference",
         "ci_low", "ci_high", "significant"]
    ]


# ---------------------------------------------------------------------------
# 3. Latency
# ---------------------------------------------------------------------------
def run_latency(arms, queries, retrieval_k=RETRIEVAL_K, context_k=CONTEXT_K, repeats=3):
    rows = []
    for name, pipeline in arms.items():
        stats = E.measure_latency(
            pipeline, queries, repeats=repeats, warmup=3,
            retrieval_k=retrieval_k, context_k=context_k,
        )
        stats["architecture"] = name
        rows.append(stats)
    columns = [
        "architecture", "retrieve_mean_ms", "rerank_mean_ms",
        "total_mean_ms", "total_p50_ms", "total_p95_ms", "n_samples",
    ]
    frame = pd.DataFrame(rows)
    return frame[[c for c in columns if c in frame.columns]]


# ---------------------------------------------------------------------------
# 4. Answer quality + abstention
# ---------------------------------------------------------------------------
def run_answer_evaluation(components, retrieval_k=RETRIEVAL_K, context_k=CONTEXT_K, seed=0):
    """Calibrate the abstention gate on one half of the queries, score the other.

    The threshold is a fitted parameter, so calibrating and scoring on the same
    queries would report the fit, not the behaviour.
    """
    calibration, test = E.split_queries(QUERIES, seed=seed)

    ungated = RAGPipeline(
        retriever=components["dense_retriever"],
        reranker=components["cross_encoder"],
        retrieval_k=retrieval_k,
        context_k=context_k,
    )
    calibrated = E.calibrate_abstention(ungated, calibration)
    print(
        f"  calibrated abstention threshold: {calibrated['threshold']:.3f} "
        f"(balanced accuracy {calibrated['balanced_accuracy']:.3f} "
        f"on {calibrated['n_calibration']} held-out-from-test queries)",
        flush=True,
    )

    llm = default_llm()
    print(f"  generator: {llm.name}", flush=True)

    rows, per_query_frames = [], {}
    for label, threshold in (
        ("no abstention gate", None),
        (f"abstention gate @ {calibrated['threshold']:.3f}", calibrated["threshold"]),
    ):
        pipeline = RAGPipeline(
            retriever=components["dense_retriever"],
            reranker=components["cross_encoder"],
            llm=llm,
            retrieval_k=retrieval_k,
            context_k=context_k,
            abstain_below=threshold,
        )
        per_query = E.evaluate_answers(pipeline, test, arch_name=label)
        per_query_frames[label] = per_query
        summary = {"configuration": label, "threshold": threshold, "generator": llm.name}
        summary.update(E.summarize_answers(per_query))
        rows.append(summary)

    return pd.DataFrame(rows), pd.concat(per_query_frames.values(), ignore_index=True), calibrated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="skip the retrieval_k sweep and answer evaluation")
    parser.add_argument("--milvus", action="store_true",
                        help="use a real Milvus Lite collection as the vector store")
    parser.add_argument("--skip-answer-eval", action="store_true")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--retrieval-k", type=int, default=RETRIEVAL_K)
    parser.add_argument("--context-k", type=int, default=CONTEXT_K)
    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.float_format", lambda v: f"{v:0.3f}")

    stats = validate()
    print("=" * 78)
    print("EVAL SET")
    print("=" * 78)
    print(f"  {stats['documents']} documents / {stats['topics']} topics")
    print(f"  {stats['queries']} queries ({stats['answerable']} answerable, "
          f"{stats['unanswerable']} abstention tests), {stats['labels']} graded labels")

    components = build_components(use_milvus=args.milvus)
    queries = answerable_queries()
    arms = build_arms(components, args.retrieval_k, args.context_k)

    print()
    print("=" * 78)
    print(f"1. ARCHITECTURE COMPARISON  (retrieval_k={args.retrieval_k}, "
          f"context_k={args.context_k})")
    print("=" * 78)
    summary, per_query_frames = run_architecture_comparison(
        arms, queries, args.retrieval_k, args.context_k
    )
    headline = ["architecture", "recall@1", "recall@3", "recall@5", "recall@10",
                "primary_hit@1", "mrr@10", "ndcg@5", "ndcg@10", "context_recall"]
    print()
    print(summary[headline].to_string(index=False))
    summary.to_csv(f"{args.output_dir}/architecture_comparison.csv", index=False)
    summary.to_csv("results.csv", index=False)

    all_per_query = pd.concat(per_query_frames.values(), ignore_index=True)
    all_per_query.to_csv(f"{args.output_dir}/per_query_retrieval.csv", index=False)

    print()
    print("-" * 78)
    print("1b. BY QUERY KIND  (ndcg@5)")
    print("-" * 78)
    by_kind = all_per_query.pivot_table(
        index="kind", columns="architecture", values="ndcg@5", aggfunc="mean"
    )
    print(by_kind.to_string())
    by_kind.to_csv(f"{args.output_dir}/by_query_kind_ndcg5.csv")

    kind_recall1 = all_per_query.pivot_table(
        index="kind", columns="architecture", values="recall@1", aggfunc="mean"
    )
    print()
    print("1c. BY QUERY KIND  (recall@1)")
    print(kind_recall1.to_string())
    kind_recall1.to_csv(f"{args.output_dir}/by_query_kind_recall1.csv")

    print()
    print("=" * 78)
    print("2. SIGNIFICANCE (paired bootstrap, 95% CI on the mean difference)")
    print("=" * 78)
    significance = run_significance_tests(per_query_frames)
    print(significance.drop(columns=["arm_a", "arm_b"]).to_string(index=False))
    significance.to_csv(f"{args.output_dir}/significance.csv", index=False)

    print()
    print("=" * 78)
    print("3. LATENCY")
    print("=" * 78)
    latency = run_latency(arms, queries, args.retrieval_k, args.context_k)
    print(latency.to_string(index=False))
    latency.to_csv(f"{args.output_dir}/latency.csv", index=False)

    print()
    print("=" * 78)
    print("4. ABLATION: retrieval_k vs context_k")
    print("=" * 78)
    sweep = RETRIEVAL_K_SWEEP if not args.quick else (5, 20)
    k_ablation = run_retrieval_k_ablation(
        components, queries, sweep=sweep, context_k=args.context_k
    )
    k_ablation.to_csv(f"{args.output_dir}/ablation_retrieval_k.csv", index=False)
    print()
    print(k_ablation[["reranker", "retrieval_k", "context_k", "context_recall",
                      "recall@1", "ndcg@5", "mrr@10", "rerank_mean_ms",
                      "total_mean_ms"]].to_string(index=False))

    print()
    print("-" * 78)
    print("4b. ABLATION: context_k (first stage fixed)")
    print("-" * 78)
    context_ablation = run_context_k_ablation(
        components, queries, retrieval_k=args.retrieval_k
    )
    print(context_ablation[["retrieval_k", "context_k", "context_recall",
                            "context_precision", "recall@1", "ndcg@5"]].to_string(index=False))
    context_ablation.to_csv(f"{args.output_dir}/ablation_context_k.csv", index=False)

    print()
    print("-" * 78)
    print("4c. ABLATION: first-stage vectors (TF-IDF vs dense)")
    print("-" * 78)
    embedder_ablation, embedder_frames = run_embedder_ablation(
        components, queries, args.retrieval_k, args.context_k
    )
    print(embedder_ablation[["first_stage_vectors", "reranker", "recall@1",
                             "primary_hit@1", "ndcg@5", "mrr@10",
                             "context_recall"]].to_string(index=False))
    embedder_ablation.to_csv(f"{args.output_dir}/ablation_embedder.csv", index=False)

    print()
    print("-" * 78)
    print("4d. Does the cross-encoder's value depend on first-stage quality?")
    print("-" * 78)
    reranker_value = run_reranker_value_tests(embedder_frames)
    print(reranker_value.to_string(index=False))
    reranker_value.to_csv(f"{args.output_dir}/reranker_value_by_first_stage.csv", index=False)

    if not (args.quick or args.skip_answer_eval):
        print()
        print("=" * 78)
        print("5. ANSWER QUALITY + ABSTENTION")
        print("=" * 78)
        answer_summary, answer_per_query, calibrated = run_answer_evaluation(
            components, args.retrieval_k, args.context_k
        )
        columns = ["configuration", "threshold", "answer_rate", "citation_validity",
                   "citation_accuracy", "groundedness", "answered_correctly",
                   "abstention_recall", "false_abstention_rate",
                   "unsupported_answer_rate"]
        print()
        print(answer_summary[columns].to_string(index=False))
        answer_summary.to_csv(f"{args.output_dir}/answer_quality.csv", index=False)
        answer_per_query.to_csv(f"{args.output_dir}/per_query_answers.csv", index=False)

    print()
    print(f"Wrote CSVs to {args.output_dir}/ and results.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
