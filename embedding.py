"""
Embedders: text -> L2-normalized vectors.

Two implementations, both real and both runnable:

- `DenseEmbedder`  - `sentence-transformers` bi-encoder (default:
  all-MiniLM-L6-v2, 384-d). This is the production-shaped option and the one
  that can match paraphrases.
- `TfidfEmbedder`  - sparse lexical vectors projected into the same interface.
  Kept deliberately as the *baseline*, not as a stand-in: the experiment needs
  a lexical arm to quantify what the dense model actually buys.

Vectors are L2-normalized on the way out, so cosine similarity reduces to an
inner product. That is what lets `NumpyVectorStore` use a single matmul and
what makes Milvus's `COSINE` metric equivalent to `IP` on the same data.
"""

from typing import Sequence

import numpy as np


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class TfidfEmbedder:
    """TF-IDF lexical baseline embedder.

    The vocabulary is a property of the embedder, so it is fitted at
    construction time from the corpus it will encode. Queries are then
    projected into that same fitted space.
    """

    name = "tfidf"

    def __init__(self, fit_texts: Sequence[str], ngram_range=(1, 2)):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range, stop_words="english"
        )
        self.vectorizer.fit(fit_texts)
        self.dimension = len(self.vectorizer.vocabulary_)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        matrix = self.vectorizer.transform(texts).toarray().astype(np.float32)
        return _l2_normalize(matrix)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class DenseEmbedder:
    """Dense bi-encoder embedder backed by `sentence-transformers`.

    The model is loaded once and reused. `embed_documents` batches, which is
    why it is a separate method from `embed_query` rather than a loop over it -
    that difference is the entire reason document encoding is a build-time cost
    and query encoding is a request-time cost.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.name = f"dense[{model_name}]"
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)
        self.dimension = int(self.model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        vector = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector[0].astype(np.float32)
