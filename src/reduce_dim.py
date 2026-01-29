from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA


def pca_2d(X: sparse.csr_matrix, random_state: int = 42) -> np.ndarray:
    """
    Reduce a TF-IDF matrix to 2D using PCA.

    PCA is deterministic and good for explanation purposes.
    """
    X_dense = X.toarray()
    pca = PCA(n_components=2, random_state=random_state)
    return pca.fit_transform(X_dense)
