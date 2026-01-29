from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


def top_k_similar_rows(
    X: sparse.csr_matrix,
    query_index: int,
    k: int = 5
):
    """
    Return top-k most similar rows to the query row using cosine similarity.
    """
    similarities = cosine_similarity(X[query_index], X).flatten()

    # Exclude self-similarity
    similarities[query_index] = -1.0

    top_indices = np.argsort(-similarities)[:k]
    return [(int(i), float(similarities[i])) for i in top_indices]

