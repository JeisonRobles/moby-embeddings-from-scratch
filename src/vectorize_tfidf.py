from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class TfidfConfig:
    min_df: int = 2
    max_df: float = 0.95
    ngram_range: tuple = (1, 1)
    stop_words: str | None = "english"
    sublinear_tf: bool = True
    norm: str = "l2"


def build_tfidf_matrix(
    documents: List[str],
    cfg: TfidfConfig
) -> Tuple[sparse.csr_matrix, TfidfVectorizer]:
    """
    Convert a list of documents into a TF-IDF matrix.

    Rows    -> documents (paragraphs)
    Columns -> vocabulary terms
    Values  -> TF-IDF weights (L2-normalized by default)
    """
    vectorizer = TfidfVectorizer(
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        ngram_range=cfg.ngram_range,
        stop_words=cfg.stop_words,
        sublinear_tf=cfg.sublinear_tf,
        norm=cfg.norm,
    )

    X = vectorizer.fit_transform(documents)
    return X, vectorizer
