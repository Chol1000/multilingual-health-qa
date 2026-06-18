"""Retrieval utilities: TF-IDF, dense bi-encoder, cross-encoder reranker, and hybrid retrieval."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


class TFIDFRetriever:
    """Character n-gram TF-IDF nearest-neighbour retrieval over a training corpus.

    Parameters
    ----------
    per_language : bool
        When True, builds a separate index per language and falls back to
        the global index for unseen languages.
    """

    def __init__(self, per_language: bool = False):
        self.per_language = per_language
        self.models: dict = {}

    def _build(self, questions: list[str], answers: list[str]) -> dict:
        vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=200_000,
            lowercase=False,
            min_df=1,
        )
        X = vec.fit_transform(questions)
        nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(X)
        return {"vec": vec, "nn": nn, "ans": np.array(answers, dtype=object)}

    def fit(self, df: pd.DataFrame) -> "TFIDFRetriever":
        if self.per_language:
            for lang, sub in df.groupby("language"):
                if len(sub) >= 2:
                    self.models[lang] = self._build(
                        sub["input"].tolist(), sub["output"].tolist()
                    )
        self.models["__all__"] = self._build(
            df["input"].tolist(), df["output"].tolist()
        )
        return self

    def predict(self, df: pd.DataFrame) -> list[str]:
        out = []
        for _, row in df.iterrows():
            key = (
                row["language"]
                if self.per_language and row["language"] in self.models
                else "__all__"
            )
            m = self.models[key]
            _, idx = m["nn"].kneighbors(m["vec"].transform([row["input"]]))
            out.append(str(m["ans"][idx[0][0]]))
        return out


class TFIDFRetrieverK3(TFIDFRetriever):
    """TF-IDF retriever that returns top-3 candidates (for RAG context concatenation)."""

    def _build(self, questions: list[str], answers: list[str]) -> dict:
        vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=200_000,
            lowercase=False,
            min_df=1,
        )
        X = vec.fit_transform(questions)
        k = min(3, len(questions))
        nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(X)
        return {"vec": vec, "nn": nn, "ans": np.array(answers, dtype=object)}


def get_context(row: pd.Series, retriever: TFIDFRetriever) -> str:
    key = row["language"] if row["language"] in retriever.models else "__all__"
    m = retriever.models[key]
    _, idx = m["nn"].kneighbors(m["vec"].transform([row["input"]]))
    return str(m["ans"][idx[0][0]])


def get_context_k3(row: pd.Series, retriever: TFIDFRetrieverK3) -> str:
    key = row["language"] if row["language"] in retriever.models else "__all__"
    m = retriever.models[key]
    _, idx = m["nn"].kneighbors(m["vec"].transform([row["input"]]))
    return " | ".join(str(m["ans"][i]) for i in idx[0])


def add_rag_context(df: pd.DataFrame, retriever: TFIDFRetriever) -> pd.DataFrame:
    """Vectorised batch context retrieval; adds a 'ctx' column to df."""
    df = df.reset_index(drop=True).copy()
    contexts = [""] * len(df)
    for lang in df["language"].unique():
        idxs = df.index[df["language"] == lang].tolist()
        key = lang if lang in retriever.models else "__all__"
        m = retriever.models[key]
        X = m["vec"].transform(df.loc[idxs, "input"].tolist())
        _, nn_idxs = m["nn"].kneighbors(X)
        for i, row_idx in enumerate(idxs):
            contexts[row_idx] = str(m["ans"][nn_idxs[i][0]])
    df["ctx"] = contexts
    return df


class DenseRetriever:
    """Bi-encoder dense retrieval using sentence-transformers (E5, BGE-M3, etc.).

    Supports the ``query:``/``passage:`` prefix convention used by E5 models.
    For BGE-M3 and other models, prefixes are omitted automatically.
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-large", device: str = "cpu"):
        from sentence_transformers import SentenceTransformer  # lazy import — optional dep
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.corpus_embeddings: np.ndarray | None = None
        self.corpus_answers: list[str] | None = None

    def fit(self, df: pd.DataFrame, batch_size: int = 128) -> "DenseRetriever":
        use_e5 = "e5" in self.model_name.lower()
        texts = [("passage: " if use_e5 else "") + t for t in df["input"].tolist()]
        self.corpus_embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        self.corpus_answers = df["output"].tolist()
        return self

    def retrieve(self, questions: list[str], batch_size: int = 128) -> list[str]:
        use_e5 = "e5" in self.model_name.lower()
        queries = [("query: " if use_e5 else "") + q for q in questions]
        q_embs = self.model.encode(
            queries,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        scores = np.dot(q_embs, self.corpus_embeddings.T)
        best_idx = np.argmax(scores, axis=1)
        return [self.corpus_answers[int(i)] for i in best_idx]


def retrieve_topk(
    retriever: DenseRetriever,
    questions: list[str],
    k: int = 20,
    batch_size: int = 128,
) -> list[list[str]]:
    """Return the top-k candidate answers for each question (used before reranking)."""
    use_e5 = "e5" in retriever.model_name.lower()
    queries = [("query: " if use_e5 else "") + q for q in questions]
    q_embs = retriever.model.encode(
        queries,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    scores = np.dot(q_embs, retriever.corpus_embeddings.T)
    top_k_idx = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
    return [[retriever.corpus_answers[int(i)] for i in row] for row in top_k_idx]


class Reranker:
    """Cross-encoder reranker for re-scoring bi-encoder candidates (e.g. BGE-reranker-v2-m3)."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cpu"):
        from sentence_transformers import CrossEncoder  # lazy import — optional dep
        self.model = CrossEncoder(model_name, device=device, max_length=512)

    def rerank(
        self,
        questions: list[str],
        candidate_lists: list[list[str]],
        batch_size: int = 256,
    ) -> list[str]:
        all_pairs: list[list[str]] = []
        boundaries: list[int] = []
        for q, cands in zip(questions, candidate_lists):
            for c in cands:
                all_pairs.append([q, c])
            boundaries.append(len(all_pairs))

        all_scores = self.model.predict(
            all_pairs, batch_size=batch_size, show_progress_bar=True
        )

        results, prev = [], 0
        for end, cands in zip(boundaries, candidate_lists):
            results.append(cands[int(np.argmax(all_scores[prev:end]))])
            prev = end
        return results


class LangSpecificRetriever:
    """Hybrid retriever: dense (E5/BGE) for high-resource languages, TF-IDF for low-resource.

    Drop-in replacement for TFIDFRetriever inside add_rag_context.
    Dense retrieval is used for languages in dense_langs; all others use TF-IDF.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        tfidf_retriever: TFIDFRetriever,
        dense_langs: set[str],
    ):
        use_e5 = "e5" in dense_retriever.model_name.lower()

        class _DenseVec:
            def transform(self_, texts):
                prefix = "query: " if use_e5 else ""
                return dense_retriever.model.encode(
                    [prefix + t for t in texts],
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )

        class _DenseNN:
            def kneighbors(self_, X, n_neighbors=1):
                scores = np.dot(X, dense_retriever.corpus_embeddings.T)
                best = np.argmax(scores, axis=1)
                return None, np.array([[i] for i in best])

        dense_entry = {
            "vec": _DenseVec(),
            "nn": _DenseNN(),
            "ans": np.array(dense_retriever.corpus_answers, dtype=object),
        }

        self.models = dict(tfidf_retriever.models)
        for lang in dense_langs:
            self.models[lang] = dense_entry
        self.models["__all__"] = dense_entry
