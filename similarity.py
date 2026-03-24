"""
similarity.py
-------------
Cosine similarity computation, verdict labels, and pairwise matrix
utilities for the Semantic Code Similarity Detector.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity as _sk_cosine
from typing import List, Tuple, Dict


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two document vectors.

    Returns a float in [0.0, 1.0].  Zero vectors yield 0.0.
    """
    if np.all(vec1 == 0) or np.all(vec2 == 0):
        return 0.0
    raw = _sk_cosine(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
    return float(np.clip(raw, 0.0, 1.0))


def get_similarity_info(score: float) -> Dict:
    """
    Return a display-ready info dict for a given similarity score.

    Keys: label, verdict, color, bg_color, level (1–5)
    """
    pct = score * 100
    if pct >= 95:
        return dict(label='Highly Suspicious',    verdict='🚨 Likely Plagiarized',
                    color='#dc2626', bg_color='#fef2f2', level=5)
    if pct >= 80:
        return dict(label='High Similarity',       verdict='⚠️ Strongly Suspicious',
                    color='#ea580c', bg_color='#fff7ed', level=4)
    if pct >= 65:
        return dict(label='Moderate Similarity',   verdict='🔶 Requires Review',
                    color='#ca8a04', bg_color='#fefce8', level=3)
    if pct >= 40:
        return dict(label='Low Similarity',        verdict='✅ Likely Original',
                    color='#16a34a', bg_color='#f0fdf4', level=2)
    return     dict(label='Very Low Similarity',   verdict='✅ Original',
                    color='#15803d', bg_color='#f0fdf4', level=1)


def pairwise_similarities(vectors: List[np.ndarray],
                           labels: List[str]) -> pd.DataFrame:
    """
    Build an N×N symmetric similarity matrix.

    Parameters
    ----------
    vectors : list of np.ndarray   One document vector per file.
    labels  : list of str          File / snippet names.

    Returns
    -------
    pd.DataFrame with shape (N, N), index and columns = labels.
    """
    n = len(vectors)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i][j] = cosine_similarity(vectors[i], vectors[j])
    return pd.DataFrame(matrix, index=labels, columns=labels)


def get_suspicious_pairs(sim_df: pd.DataFrame,
                          threshold: float = 0.80
                          ) -> List[Tuple[str, str, float]]:
    """
    Return all off-diagonal pairs whose similarity ≥ threshold,
    sorted descending by score.
    """
    labels = list(sim_df.columns)
    n = len(labels)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            score = sim_df.iloc[i, j]
            if score >= threshold:
                pairs.append((labels[i], labels[j], float(score)))
    return sorted(pairs, key=lambda x: x[2], reverse=True)
