# Healthcare RAG Pipeline: Retrieval Architecture Comparison

A small, honestly-evaluated retrieval-augmented generation pipeline over a
synthetic healthcare corpus. It compares three retrieval architectures, ablates
the components that are usually asserted rather than measured, and evaluates the
generation stage on citation honesty and abstention.

```text
query -> embedder -> vector store -> retrieval_k candidates
                                          |
                                    second stage (rerank)
                                          |
                                    context_k passages -> LLM -> cited answer
```

Every component is real: dense retrieval via `sentence-transformers`
(all-MiniLM-L6-v2), a real cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`), and
an optional real Milvus Lite collection. Measured results are in
[retrieval_experiment.ipynb](retrieval_experiment.ipynb) and
[results/](results/).

---

## 1. Headline results

127 documents, 85 answerable eval queries with graded relevance, plus 10
unanswerable queries for abstention testing. `retrieval_k=20`, `context_k=5`.

| Architecture | Recall@1 | Recall@3 | Recall@5 | Recall@10 | primary_hit@1 | MRR@10 | NDCG@5 | NDCG@10 | context_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **A.** Lexical retrieval (BM25 only) | 0.418 | 0.572 | 0.618 | 0.689 | 0.612 | 0.709 | 0.647 | 0.668 | 0.618 |
| **B.** Dense retrieval (no reranker) | 0.558 | 0.819 | 0.890 | 0.930 | 0.800 | 0.922 | 0.875 | 0.888 | 0.890 |
| **C.** Dense + cross-encoder reranker | **0.568** | 0.807 | 0.880 | 0.926 | **0.847** | **0.924** | **0.879** | **0.893** | 0.880 |
| **D.** Dense + lexical second-stage rerank | 0.444 | 0.614 | 0.668 | 0.813 | 0.647 | 0.768 | 0.696 | 0.739 | 0.668 |

Paired bootstrap over the same queries (95% CI on the mean difference):

| Comparison | Metric | Difference | 95% CI | Significant |
| --- | --- | --- | --- | --- |
| Dense vs lexical retrieval | Recall@1 | +0.140 | [+0.058, +0.224] | **yes** |
| Dense vs lexical retrieval | NDCG@5 | +0.229 | [+0.159, +0.304] | **yes** |
| Cross-encoder vs no reranker | Recall@1 | +0.010 | [-0.043, +0.063] | no |
| Cross-encoder vs no reranker | primary_hit@1 | +0.047 | [-0.047, +0.141] | no |
| Cross-encoder vs lexical second stage | NDCG@5 | +0.183 | [+0.128, +0.243] | **yes** |
| **Lexical second stage vs no reranker** | Recall@1 | **-0.113** | [-0.195, -0.035] | **yes (worse)** |

**Three conclusions, stated at the strength the evidence supports:**

1. **Dense retrieval is the upgrade that matters**, and its advantage is
   concentrated where real users live — paraphrased, lay-vocabulary questions
   (NDCG@5 0.403 lexical → 0.827 dense on that slice; see §5).
2. **BM25 in second position is a net negative here.** Measured against the same
   dense first stage, it loses 0.11 recall@1 and 0.22 context_recall, and it
   degrades further as the candidate pool widens. It is reported as a *lexical
   second-stage reranking baseline*, which is what it is.
3. **The cross-encoder's gain over no reranker is not significant on this
   corpus.** It wins every point estimate, and it fixes specific distractor
   cases, but the dense retriever already lands a relevant passage in the top 5
   on ~89% of queries, so there is little left to reorder. Its value turns out to
   depend on first-stage quality (§6) — which is a more useful finding than a
   manufactured win.

## 2. Problem statement

Question answering over clinical reference material has two properties that make
both "stuff the corpus in the prompt" and "just embed and retrieve" unworkable:

- **Corpora are too large or too sensitive to put in context.** Only a handful of
  passages are relevant to any question.
- **A wrong answer carries real cost.** An LLM answering from parametric memory
  produces fluent, plausible, wrong clinical statements. Retrieval grounds the
  answer in a citable source — but only if retrieval reliably surfaces the right
  source, and only if the system declines when it doesn't.

So this project measures three things separately, because they fail separately:
which architecture retrieves the right passage, what each stage costs, and
whether the generator cites honestly and abstains when it has no source.

## 3. Evaluation set design

A retrieval eval is only as informative as its hard negatives. Both the corpus
and the query set are built to be discriminative rather than flattering.

**Corpus** ([data.py](data.py) `CORPUS`): 127 synthetic passages over 14 topic
areas, built with deliberate **distractor families** — clusters of passages that
share a query's surface vocabulary while answering a different question. For
"first-line drug for type 2 diabetes" the corpus also contains first-line therapy
for type **1** diabetes, second-line therapy **after** metformin, and first-line
therapy for **gestational** diabetes. Keyword overlap cannot separate these.

**Queries** (`QUERIES`): 95 questions, each labeled with **graded** relevance
(`2` = directly answers, `1` = useful supporting context), tagged by kind:

| Kind | n | What it isolates |
| --- | --- | --- |
| `direct` | 30 | question phrased close to the passage's wording |
| `paraphrase` | 25 | lay vocabulary with little lexical overlap ("water pill", "puffer") — separates dense from lexical |
| `distractor` | 20 | a near-miss passage shares more surface vocabulary than the correct one — separates a reranker from raw recall |
| `multi_doc` | 10 | several passages contribute, with graded gains |
| `unanswerable` | 10 | no passage answers it — abstention test, excluded from retrieval metrics |

`python3 data.py` self-validates the set: every label points at a real passage,
every answerable query has a gain-2 passage, every unanswerable query has none.

**Why synthetic:** real guideline text carries licensing restrictions and real
records carry patient-privacy risk. Proving the architecture and harness on
synthetic data first means the licensing and privacy review gates a corpus swap,
not the whole project. Swapping in a licensed, de-identified corpus changes the
corpus loader and nothing else.

## 4. Metrics

| Metric | What it answers |
| --- | --- |
| `Recall@k` | did the relevant material reach the candidate set at all? |
| `Recall@1` | is a relevant passage already on top? |
| `primary_hit@1` | is a **directly-answering** (gain-2) passage at rank 1? Adjacent context earns no credit — this is where a reranker should show up |
| `MRR@10` | how far down before the first relevant hit |
| `NDCG@5/@10` | rank-weighted **and** grade-aware — the reason labels are graded |
| `Precision@k` | reported as **context efficiency**, not retriever skill (see below) |
| `context_recall` | recall restricted to the `context_k` passages the LLM actually receives — anything outside it cannot be cited, so this bounds answer quality |

**On precision@k.** With one or two relevant passages per query, precision@10 is
arithmetically capped near 0.1 no matter how good the retriever is. A "precision
drop" from k=3 to k=10 measures the label distribution, not retrieval quality.
The earlier version of this project reported exactly that fall (0.333 → 0.103)
as a finding; it was an artifact. Precision is kept here only as a prompt-cost
measure: how many context slots earn their tokens.

Metric implementations are checked against hand-computed values in
[test_metrics.py](test_metrics.py) (13 tests, `python3 test_metrics.py`).

## 5. Where each component earns its cost

NDCG@5 by query kind:

| Kind | A. lexical | B. dense | C. dense + cross-encoder | D. dense + lexical 2nd stage |
| --- | --- | --- | --- | --- |
| `direct` | 0.860 | 0.931 | **0.938** | 0.879 |
| `distractor` | 0.742 | **0.909** | 0.898 | 0.753 |
| `paraphrase` | 0.403 | 0.827 | **0.836** | 0.519 |
| `multi_doc` | 0.424 | 0.761 | **0.772** | 0.472 |

The `paraphrase` slice is the entire case for dense retrieval: lexical scores
0.403 there against 0.827 for dense, while on `direct` queries the gap is
0.860 vs 0.931. Users do not phrase questions in the corpus's vocabulary.

## 6. Ablations

### 6a. `retrieval_k` vs `context_k` — the fix that makes reranking measurable

If you retrieve `k`, rerank those same `k`, and evaluate at `k`, reordering
cannot change the set: recall@k and precision@k are **mathematically identical**
with the reranker on or off. The previous version of this project did exactly
that, which made its reranker claims unfalsifiable — every reported number was
the same with the reranker disabled.

A second stage only acts in the gap between the two numbers. Holding
`context_k=5` and widening `retrieval_k`:

| Reranker | retrieval_k | context_recall | Recall@1 | primary_hit@1 | rerank ms | total ms |
| --- | --- | --- | --- | --- | --- | --- |
| none | 5 | 0.890 | 0.558 | 0.800 | 0.003 | 18.9 |
| none | 20 | 0.890 | 0.558 | 0.800 | 0.005 | 17.2 |
| none | 50 | 0.890 | 0.558 | 0.800 | 0.010 | 16.8 |
| lexical (BM25) | 5 | 0.890 | 0.503 | 0.729 | 0.069 | 17.9 |
| lexical (BM25) | 20 | 0.668 | 0.444 | 0.647 | 0.156 | 17.0 |
| lexical (BM25) | 50 | 0.645 | 0.418 | 0.612 | 0.294 | 20.0 |
| cross-encoder | 5 | 0.890 | **0.589** | **0.894** | 23.5 | 40.9 |
| cross-encoder | 20 | 0.880 | 0.568 | 0.847 | 47.9 | 67.4 |
| cross-encoder | 50 | 0.878 | 0.568 | 0.847 | 69.5 | 86.4 |

- At `retrieval_k = context_k = 5`, all three arms sit at **0.890 context_recall**
  — identical, as the math requires. Any reranker "gain" measured there is an
  artifact.
- Widening the first stage does nothing for the no-reranker arm (flat 0.890): its
  top 5 is its top 5.
- The **lexical second stage degrades monotonically** (0.890 → 0.645) as the pool
  widens. A wider pool hands BM25 more lexically-similar distractors to promote,
  and it promotes them. This is the mechanism, made visible.
- The **cross-encoder holds** quality as the pool widens (0.890 → 0.878) while
  its cost grows with the candidate count (23 → 69 ms). On this corpus it is
  buying robustness that isn't needed.

### 6b. `context_k` — the prompt-cost knob

`retrieval_k=20` fixed, dense + cross-encoder:

| context_k | context_recall | context_precision | Recall@1 | NDCG@5 |
| --- | --- | --- | --- | --- |
| 1 | 0.568 | 0.882 | 0.568 | 0.879 |
| 3 | 0.807 | 0.459 | 0.568 | 0.879 |
| 5 | 0.880 | 0.311 | 0.568 | 0.879 |
| 10 | 0.926 | 0.166 | 0.568 | 0.879 |

Citable coverage rises 0.568 → 0.926 while precision falls 0.882 → 0.166 — past
`context_k=5`, five of every six passages in the prompt are irrelevant. Note
Recall@1 and NDCG@5 are **flat**: `context_k` is a cost-and-distraction dial, not
a retrieval-quality one. `context_k=3–5` is the defensible range here.

### 6c. Does the reranker's value depend on the first stage?

Same reranker, same `k`, two first stages of different strength:

| First stage | Reranker | Recall@1 | primary_hit@1 | NDCG@5 | context_recall |
| --- | --- | --- | --- | --- | --- |
| TF-IDF | none | 0.435 | 0.635 | 0.660 | 0.634 |
| TF-IDF | cross-encoder | 0.535 | 0.800 | 0.808 | 0.778 |
| dense | none | 0.558 | 0.800 | 0.875 | 0.890 |
| dense | cross-encoder | 0.568 | 0.847 | 0.879 | 0.880 |

| Cross-encoder gain | Metric | Difference | 95% CI | Significant |
| --- | --- | --- | --- | --- |
| behind weak first stage (TF-IDF) | NDCG@5 | +0.148 | [+0.096, +0.205] | **yes** |
| behind weak first stage (TF-IDF) | primary_hit@1 | +0.165 | [+0.094, +0.247] | **yes** |
| behind strong first stage (dense) | NDCG@5 | +0.004 | [-0.035, +0.040] | no |
| behind strong first stage (dense) | primary_hit@1 | +0.047 | [-0.047, +0.141] | no |

**A reranker substitutes for first-stage quality; it does not compound with it.**
"Add a cross-encoder" is not architecture-independent advice. Also worth noting:
TF-IDF + cross-encoder (NDCG@5 0.808) nearly reaches dense-only (0.875) — if a
dense model were unavailable, reranking a lexical first stage recovers most of
the quality.

## 7. Latency

`retrieval_k=20`, `context_k=5`, CPU, warmup discarded, 255 samples per arm:

| Architecture | retrieve mean | rerank mean | total mean | total p50 | total p95 |
| --- | --- | --- | --- | --- | --- |
| A. Lexical (BM25 only) | 0.30 ms | 0.00 ms | 0.30 ms | 0.30 ms | 0.37 ms |
| B. Dense (no reranker) | 19.2 ms | 0.01 ms | 19.2 ms | 17.8 ms | 27.0 ms |
| C. Dense + cross-encoder | 18.5 ms | 52.2 ms | 70.7 ms | 66.5 ms | 100.5 ms |
| D. Dense + lexical 2nd stage | 17.6 ms | 0.15 ms | 17.8 ms | 17.4 ms | 22.4 ms |

The cross-encoder roughly triples end-to-end latency for a gain §1 could not
distinguish from zero here. Warmup matters: the first call into a torch model
pays lazy kernel initialization and can be an order of magnitude slower than
steady state, which would otherwise be charged to whichever arm ran first.

These are CPU numbers on one machine and are the least portable results in this
README — absolute values move ±20% between runs and a GPU changes the
cross-encoder trade entirely. The relative shape (lexical ≈ free, dense ~18 ms,
cross-encoder linear in `retrieval_k`) is the part that transfers.

## 8. Generation: citation contract and abstention

The generation stage is where a retrieval miss becomes a fabricated clinical
claim, so the contract is enforced at three points rather than trusted once:

1. **Prompt** — every claim must come from the supplied context and carry a
   passage citation; refusal is required when the context lacks the answer.
2. **Structure** — the response is parsed into `{answer, cited_ids, abstained}`,
   so citations are a field rather than a string the reader must trust.
3. **Verification** — the harness checks each citation against the context window
   and the gold labels. A model citing a document id it was never shown is
   caught here, which is the failure a citation-shaped prompt quietly invites.

**Abstention is calibrated, not guessed.** The retrieval-score threshold is a
fitted parameter, so it is fitted on one stratified half of the queries (47
queries, balanced accuracy 0.905) and scored on the held-out other half (43
answerable + 5 unanswerable). Score scales differ per architecture (cosine
similarity, BM25, cross-encoder logits), so it is calibrated per pipeline rather
than hard-coded.

| Configuration | answer rate | citation validity | citation accuracy | groundedness | abstention recall | false abstention | unsupported answers |
| --- | --- | --- | --- | --- | --- | --- | --- |
| No abstention gate | 1.000 | 1.000 | 0.930 | 0.955 | 0.000 | 0.000 | **1.000** |
| Calibrated gate | 0.767 | 1.000 | **1.000** | 0.954 | **1.000** | 0.233 | **0.000** |

Without a gate the system answers **every** unanswerable question, each grounded
in a passage that does not address it. With the gate, unsupported answers go to
zero and abstention recall to 100%, paid for with a 23% false-abstention rate. In
a clinical reference setting that is the right side of the trade: a refusal costs
a click; a confidently wrong threshold does not.

**Which LLM ran:** these numbers come from `generation.ExtractiveLLM`, a
deterministic quoter that follows the same citation contract, because no
Anthropic credentials were available in this environment.
`generation.AnthropicLLM` is the real path (model `claude-opus-5`, citation
contract in the system prompt, refusal handling) and is used automatically when
`ANTHROPIC_API_KEY` or an `ant auth login` profile is present. The extractive
baseline is trivially faithful, so **treat its groundedness figure as a floor** —
what it genuinely measures is whether retrieval handed over the right passage and
whether the gate fired correctly.

## 9. Failure handling

| Failure | Behaviour |
| --- | --- |
| First stage raises | Query returns `degraded=True` with an abstention rather than propagating the exception |
| Reranker raises | Degrades to first-stage ordering and still answers (a stale ranking beats no answer) |
| Zero candidates retrieved | Abstains, records `no documents retrieved` |
| Top score below threshold | Abstains **before** spending a generation call |
| LLM raises | Abstains, records the error type |
| `context_k > retrieval_k` | Rejected at construction — the second stage cannot reorder what was never fetched |

Each path is covered in [test_metrics.py](test_metrics.py).

## 10. Architecture and responsibility boundaries

[contracts.py](contracts.py) states every interface in one place:

```python
Embedder.embed_documents(texts) -> (n, d) matrix     # owns a model, no index
Embedder.embed_query(text)      -> (d,) vector
VectorStore.add(ids, texts, vectors)                 # owns an index, no model
VectorStore.search(query_vector, top_k) -> [Candidate]
Retriever.retrieve(query_text, top_k)   -> [Candidate]   # owns the composition
Reranker.rerank(query_text, candidates) -> [Candidate]   # reorders, never adds
LLM.generate(query_text, context)       -> {answer, cited_ids, abstained}
```

**The boundary that was wrong before.** The earlier version exposed
`VectorStore.search(query_text, top_k)` — a store that accepts *text*. That
implies the store owns an embedder, and in that version it literally did: it held
a reference to the `Embedder` and read the embedder's document matrix, so the two
objects were one object with two names. A real Milvus server cannot embed text
for you, so that interface could never have been swapped for the thing it stood
in for. Now the composition is explicit and lives in `DenseRetriever`:

```python
query_vector = embedder.embed_query(query_text)   # text   -> vector
candidates   = store.search(query_vector, top_k)  # vector -> documents
```

Which is what makes `MilvusVectorStore` a genuine drop-in for
`NumpyVectorStore` — verified: **identical top-10 rankings on 25/25 queries.**

## 11. Technology stack

| Component | This repo | Production notes |
| --- | --- | --- |
| Embedding | `sentence-transformers` all-MiniLM-L6-v2 (384-d), real | Swap model name; consider a domain-tuned or larger biomedical encoder |
| Vector store | `NumpyVectorStore` (exact) or real Milvus Lite via `pymilvus` | Same `MilvusVectorStore` class against a server/cluster URI |
| Lexical baseline | BM25 from scratch, corpus-level IDF | Fine as-is, or Elasticsearch/OpenSearch for hybrid |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2`, real | Larger cross-encoder or a hosted rerank API; GPU changes the latency calculus |
| LLM | `AnthropicLLM` (`claude-opus-5`) when credentials exist; deterministic extractive baseline otherwise | Add response caching and per-tenant rate limits |
| Eval | numpy, pandas, matplotlib, paired bootstrap | Wire the same harness into CI as a regression gate |

The exact-search numpy store is not a placeholder for Milvus — it is the correct
brute-force baseline at 127 documents (O(N) per query, ~0.3 ms) and the reference
an ANN index should be validated against, which is exactly how it is used in §10.

## 12. Project structure

```text
.
├── data.py                     # corpus + graded eval set, self-validating
├── contracts.py                # every component interface in one place
├── embedding.py                # TfidfEmbedder, DenseEmbedder
├── vector_store.py             # NumpyVectorStore, MilvusVectorStore
├── retrieval.py                # Bm25Index/Retriever, rerankers (identity/BM25/cross-encoder)
├── pipeline.py                 # RAGPipeline: retrieval_k, context_k, failure handling
├── generation.py               # citation contract, AnthropicLLM, ExtractiveLLM
├── metrics.py                  # recall/precision/MRR/NDCG, paired bootstrap
├── evaluation.py               # retrieval, latency, abstention calibration, answer eval
├── experiment.py               # driver for all comparisons and ablations
├── test_metrics.py             # hand-checked metric + contract tests
├── retrieval_experiment.ipynb  # executed notebook with all results
├── results.csv                 # headline architecture table
└── results/                    # all result CSVs
```

## 13. Running it

```bash
pip install -r requirements.txt        # first dense/cross-encoder run downloads ~90 MB of models

python3 data.py                        # validate the eval set
python3 test_metrics.py                # 13 metric + contract tests
python3 experiment.py                  # full comparison + ablations (~5 min, CPU)
python3 experiment.py --quick          # skip the retrieval_k sweep and answer eval
jupyter notebook retrieval_experiment.ipynb
```

Optional extras, each degrading gracefully when absent:

```bash
pip install "pymilvus[milvus_lite]>=2.4.2" && python3 experiment.py --milvus
pip install anthropic && export ANTHROPIC_API_KEY=...   # or: ant auth login
```

A single query end to end:

```python
from data import CORPUS
from generation import default_llm
from pipeline import build_pipeline

pipeline = build_pipeline(
    CORPUS,
    retriever="dense",
    reranker="cross-encoder",
    llm=default_llm(),
    retrieval_k=20,   # candidates fetched by the first stage
    context_k=5,      # passages that reach the LLM
)
result = pipeline.run("Which water pill is used to bring blood pressure down?", generate=True)
print(result.context_ids)                    # ['HT06', ...]
print(result.generation["answer"])
print(result.generation["cited_ids"], result.generation["abstained"])
```

## 14. Limitations and safety

- **127 documents is small, and this bounds the reranking conclusion.** Dense
  recall@5 of ~0.89 leaves little headroom, which is *why* the cross-encoder
  cannot show a significant gain. At 10³–10⁶ documents the first stage misses
  more and a reranker has more to fix. Re-measure before generalizing §1(3).
- **n=85 answerable queries** limits the resolvable effect size to roughly ±0.05.
  Smaller differences are reported with confidence intervals, not as findings.
- **The corpus is synthetic** and written to resemble patient-education register.
  It is not clinical guidance. Real guideline text is longer and more hedged and
  needs chunking, which introduces retrieval failure modes this corpus cannot
  exhibit.
- **Labels are single-annotator.** Graded relevance is a judgment call and no
  inter-annotator agreement was measured.
- **No real LLM call in the published run** (no credentials), so faithfulness
  numbers come from the extractive baseline and are a floor, not a measurement of
  a language model.
- **Groundedness is a token-overlap proxy**, not an LLM judge or human review. It
  catches an answer asserting specifics absent from its cited passage; it does
  not catch a fluent misreading of a passage that shares its vocabulary.
- **No real patient data anywhere in this repo**, and nothing copied from a
  licensed or copyrighted source. Before connecting a real clinical corpus or
  real patient queries, a licensing review (guideline text) and a
  privacy/security review (encryption, access control, audit logging, and HIPAA
  compliance where applicable) are prerequisites, not optional hardening.
- **Abstention rests on 10 unanswerable queries** — 5 for calibration and 5 for
  scoring — which is enough to demonstrate the mechanism and far too few to fix a
  production threshold. A 5-query denominator means the reported 100% abstention
  recall has a wide interval; widen that slice before trusting the operating
  point.

## 15. What I'd do next

1. **Scale the corpus 100×** (chunked, licensed guideline text) and re-run this
   harness unchanged. The reranking conclusion is the one most likely to flip,
   and the harness is built so that re-running it is the whole experiment.
2. **Hybrid retrieval** (dense + BM25 fused with reciprocal rank fusion). The
   per-kind breakdown in §5 suggests the two arms fail on different queries —
   lexical wins nothing overall but is competitive on `direct` — which is
   precisely the profile that fusion exploits.
3. **A real LLM faithfulness eval**: run `AnthropicLLM` over the same queries and
   grade with an LLM judge plus a human-reviewed subset, replacing the
   token-overlap proxy.
4. **Query rewriting** for the `paraphrase` slice, and measure whether it adds
   anything on top of a dense retriever that already handles paraphrase well.
5. **CI regression gate**: fail the build if recall@1 or context_recall drops more
   than the bootstrap CI width on any embedding-model or corpus re-index change.
6. **Widen the abstention slice** and calibrate the threshold against a
   cost-weighted objective rather than balanced accuracy, since a false answer
   and a false refusal are not equally expensive.
