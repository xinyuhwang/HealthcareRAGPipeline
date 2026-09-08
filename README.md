# Healthcare RAG Pipeline: Retrieval Experiment

A small, evaluated retrieval-augmented generation (RAG) pipeline over a
synthetic healthcare corpus:

```
query -> embedding -> Milvus -> top-k documents -> reranker -> LLM
```

Built to compare retrieval quality (precision/recall) and latency at
`top_k = 3, 5, 10`, with real measured numbers in `retrieval_experiment.ipynb`.

---

## 1. Overview

This project implements and evaluates a two-stage retrieval + generation
pipeline for answering healthcare questions from a small knowledge base:

1. A query is embedded into a vector space.
2. The vector store (Milvus, or a local stand-in — see [Section 15](#15-limitations--safety-considerations)) returns the top-k
   most similar documents.
3. A reranker re-scores those top-k candidates for finer-grained relevance.
4. An LLM generates a grounded answer from the reranked context, citing the
   source document.

The deliverable is a runnable, reproducible experiment
(`retrieval_experiment.ipynb`) with real precision/recall/latency numbers at
three values of `top_k`, plus this write-up of the design and its tradeoffs.

## 2. Problem Statement

Question-answering over specialized domains like healthcare has two
properties that make naive "stuff the whole corpus into the prompt"
approaches unworkable and naive "just embed and retrieve" approaches risky:

- **Corpora are too large (or too sensitive) to put in-context.** Clinical
  guidelines, patient education material, and institutional protocols can
  run to thousands of documents; only a handful are relevant to any given
  question.
- **Wrong or ungrounded answers carry real cost.** An LLM answering from
  parametric memory alone can produce a fluent, plausible, and _wrong_
  clinical statement. Retrieval grounds the answer in a specific, citable
  source — but only if the retrieval step itself reliably surfaces the
  right source, which is exactly what this experiment measures.

The core question this project answers: **for a given corpus and query
distribution, what value of `top_k` gives the best retrieval quality for
the least latency cost**, and how does a reranking stage change that
answer?

## 3. Goals & Requirements

**Goals**

- Implement the full pipeline shape (`query -> embedding -> Milvus -> top-k
-> reranker -> LLM`) end-to-end and runnably.
- Evaluate retrieval quality quantitatively (precision@k, recall@k) against
  hand-labeled ground truth, not by eyeballing results.
- Measure latency per stage and end-to-end, not just anecdotally.
- Use a healthcare-flavored corpus without any real licensing or patient
  privacy risk.

**Functional requirements**

- Given a natural-language query, return ranked candidate documents and a
  generated answer that cites its source.
- Support re-running the same evaluation harness against different `top_k`
  values and different component implementations.

**Non-functional requirements**

- Retrieval latency should be a small fraction of total pipeline latency
  once a real LLM generation call is added.
- The corpus and eval set should be reviewable for licensing/privacy risk
  at a glance (i.e., synthetic and clearly labeled as such).

See the [design doc](docs/design_doc.pdf) for the full requirements
derivation and the criteria used to arrive at them.

## 4. System Architecture

```
┌────────┐   ┌───────────┐   ┌───────────────┐   ┌──────────┐   ┌───────────┐   ┌─────┐
│ Query  │──▶│ Embedder  │──▶│ Vector Store  │──▶│ top-k    │──▶│ Reranker  │──▶│ LLM │
│ (text) │   │ (dense    │   │ (Milvus:      │   │ candidate│   │ (cross-   │   │     │
│        │   │ vector)   │   │ ANN search)   │   │ docs     │   │ encoder)  │   │     │
└────────┘   └───────────┘   └───────────────┘   └──────────┘   └───────────┘   └─────┘
```

Each stage is written behind a small, stable interface so that swapping the
underlying implementation (e.g. TF-IDF → a real embedding model) never
requires changing the pipeline or evaluation code:

```python
Embedder.embed_query(text) -> vector
VectorStore.search(query_text, top_k) -> [{id, text, score}, ...]
Reranker.rerank(query_text, candidates) -> [{..., rerank_score}, ...]
LLM.generate(query_text, ranked_docs) -> {answer, context_used}
```

Full architecture rationale, alternatives considered, and the production
infrastructure design (Milvus deployment topology, indexing strategy,
observability, failure modes) are in
[`docs/design_doc.pdf`](docs/design_doc.pdf).

## 8. Data Processing

**Corpus** (`data.py :: CORPUS`): 44 short, fully synthetic passages across
15 topic areas — diabetes, hypertension, asthma, infection/antibiotics,
pediatric fever, vaccination, mental health screening, pregnancy, cardiac
risk, stroke, nutrition, kidney, and thyroid. Each passage has a stable
`id` and `topic` field.

**Evaluation queries** (`data.py :: QUERIES`): 35 natural-language questions,
each hand-labeled with the `id`(s) of the passage(s) considered relevant.
Ground truth was written _before_ the retriever was built or tuned, to
avoid unconsciously designing the retriever around the eval set.

**Why synthetic data:** healthcare text often carries licensing restrictions
(professional-society guidelines) or patient-privacy risk (real records).
Building and validating the pipeline against a synthetic corpus first lets
the architecture and evaluation harness be proven out before any licensing
or privacy review is needed. Swapping in a real, properly licensed and
de-identified corpus only requires changing the corpus loader — nothing
else in the pipeline.

## 9. Evaluation

**Metrics**

- `Precision@k` = (# relevant docs in top-k) / k
- `Recall@k` = (# relevant docs in top-k) / (total # relevant docs for that
  query)
- Latency, measured separately for the retrieve step, the rerank step, and
  end-to-end, averaged over 20 repeated runs per query to reduce timing
  jitter, with mean and p95 reported.

**Method:** for each `top_k ∈ {3, 5, 10}`, run every query through the full
pipeline, compute precision/recall against the labeled ground truth, and
record per-stage timings. See `experiment.py` for the harness and
`retrieval_experiment.ipynb` for the executed run.

## 12. Technology Stack

| Component    | Prototype (this repo, offline sandbox)                                     | Production target                                                                  |
| ------------ | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Embedding    | scikit-learn `TfidfVectorizer`                                             | `sentence-transformers` or a hosted embedding API                                  |
| Vector store | In-memory cosine similarity (`MilvusLiteStub`, mirrors the `pymilvus` API) | Milvus / Milvus Lite via `pymilvus`                                                |
| Reranker     | From-scratch BM25                                                          | Cross-encoder (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) or a hosted rerank API |
| LLM          | Template-based extractive stub                                             | Claude / GPT API call over retrieved context                                       |
| Eval / glue  | Python, `numpy`, `pandas`, `matplotlib`                                    | Same                                                                               |

**Why the stand-ins:** this repo was built in a network-isolated environment
that couldn't install `pymilvus`/`sentence-transformers` or call a hosted
API. Every stand-in implements the exact interface its production
counterpart would expose (see [Section 4](#4-system-architecture)), so
swapping to real infrastructure is a component substitution, not a rewrite.
Details and a step-by-step migration path are in the
[design doc](docs/design_doc.pdf), Section 10 (Roadmap).

## 13. Project Structure

```
.
├── data.py                       # synthetic corpus + labeled eval queries
├── pipeline.py                   # Embedder, VectorStore, Reranker, LLM, RAGPipeline
├── experiment.py                 # precision/recall/latency evaluation harness
├── retrieval_experiment.ipynb    # executed notebook with real results
├── results.csv                   # top_k=3/5/10 metrics, machine-readable
└── docs/
    └── design_doc.pdf            # design process, architecture, infra plan
```

## 14. Running the Pipeline

```bash
pip install numpy pandas scikit-learn matplotlib

# Run the full precision/recall/latency experiment and print a summary table
python3 experiment.py

# Or step through interactively
jupyter notebook retrieval_experiment.ipynb
```

To try a single query end-to-end:

```python
from data import CORPUS
from pipeline import RAGPipeline

pipeline = RAGPipeline(CORPUS)
result = pipeline.run("What medication is first-line for type 2 diabetes?",
                       top_k=5, use_reranker=True, use_llm=True)
print(result["generation"]["answer"])
```

## 15. Limitations & Safety Considerations

- **Small corpus inflates recall.** With only 44 documents, most queries
  have one obviously-matching passage; recall will not hold at real-world
  scale (10³–10⁶ documents) where many partially-relevant passages compete.
  Re-measure on the real target corpus before trusting these numbers in
  production.
- **TF-IDF is lexical, not semantic.** It cannot match synonyms or
  paraphrases (e.g. "sugar levels" vs. "glucose") the way a real dense
  embedding model would — expect this to be the single biggest quality
  jump from prototype to production.
- **BM25 reranking is still lexical**, not a substitute for a neural
  cross-encoder's semantic disambiguation.
- **No real patient data anywhere in this repo.** The corpus is entirely
  synthetic and written from scratch; nothing is copied from a licensed or
  copyrighted source. Before connecting a real clinical corpus or real
  patient queries, a licensing review (for guideline text) and a privacy/
  security review (encryption, access control, audit logging, and — if
  applicable — HIPAA compliance) are required prerequisites, not optional
  hardening.
- **The LLM stage is a template stub**, not a real generation call, in this
  environment. A production deployment must add a prompt contract that
  requires citing the source document and refusing to answer when no
  relevant document is retrieved, to avoid fabricated clinical claims.
- **Single-relevant-doc bias in the eval set**: most eval queries have
  exactly one labeled relevant document. Real query distributions have more
  partial and multi-source relevance; the eval set should grow accordingly.

## 16. Future Improvements

1. Swap TF-IDF for a real embedding model; re-run this same harness and
   compare precision/recall deltas directly against the baseline below.
2. Stand up Milvus (Lite for dev, clustered for staging/production);
   migrate `VectorStore` to `pymilvus.MilvusClient` with no change to
   callers.
3. Swap BM25 for a cross-encoder reranker, and re-measure the latency/
   quality tradeoff at `k = {3, 5, 10}` again — a neural reranker's cost
   profile differs from BM25's.
4. Connect a real LLM API with a citation-required prompt contract; add a
   faithfulness/hallucination evaluation (does the generated answer
   actually match its cited passage).
5. Add hybrid (lexical + dense) retrieval and query rewriting.
6. Grow the corpus and eval set together, including multi-label and
   graded-relevance queries, and re-validate the optimal `top_k`.
7. Add retrieval-quality regression tests to CI, gating any embedding model
   or corpus re-index change.

## 17. Results

Measured by actually executing `experiment.py` (35 queries, 20 timing
repeats per query per `top_k`):

| top_k | Mean Precision@k | Mean Recall@k | Mean retrieve latency | Mean rerank latency | Mean total latency | p95 total latency |
| ----- | ---------------- | ------------- | --------------------- | ------------------- | ------------------ | ----------------- |
| 3     | 0.333            | 0.960         | 0.40 ms               | 0.06 ms             | 0.46 ms            | 0.53 ms           |
| 5     | 0.206            | 0.967         | 0.47 ms               | 0.09 ms             | 0.57 ms            | 0.88 ms           |
| 10    | 0.103            | 0.967         | 0.41 ms               | 0.13 ms             | 0.54 ms            | 0.64 ms           |

**Reading the numbers:** recall is already ~0.96 at `k=3` and barely moves
by `k=10` — for this corpus, the correct document is almost always found
within the first few results, so raising `k` mostly adds irrelevant
documents rather than finding more relevant ones (which is exactly why
precision falls roughly as `1/k`). Retrieval latency is close to flat
across `k` (dominated by a single similarity computation, not the top-k
selection); rerank latency grows roughly linearly with `k`, since BM25
scores every candidate. **Conclusion for this corpus/pipeline:** `top_k=3`
gives nearly the same recall as `top_k=10` at about a third of the rerank
cost and much better precision. This should be re-validated whenever the
corpus or the embedding/reranker models change — see
[Limitations](#15-limitations--safety-considerations).

Full per-query results, the worked single-query example (showing the
reranker correcting a lexical false-positive), and the results plot are in
`retrieval_experiment.ipynb`.
