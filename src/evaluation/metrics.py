"""Ranking and rating metrics for recommender evaluation.

All ranking metrics operate on a single user's ranked recommendation list and a
set of relevant items, then are averaged across users by the caller.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set

import numpy as np


def precision_at_k(recommended: Sequence[int], relevant: Set[int], k: int) -> float:
    if k == 0:
        return 0.0
    topk = recommended[:k]
    hits = sum(1 for i in topk if i in relevant)
    return hits / k


def recall_at_k(recommended: Sequence[int], relevant: Set[int], k: int) -> float:
    if not relevant:
        return 0.0
    topk = recommended[:k]
    hits = sum(1 for i in topk if i in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: Sequence[int], relevant: Set[int], k: int) -> float:
    topk = recommended[:k]
    dcg = 0.0
    for pos, item in enumerate(topk, start=1):
        if item in relevant:
            dcg += 1.0 / np.log2(pos + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(pos + 1) for pos in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def coverage(all_recommended: Iterable[int], catalog_size: int) -> float:
    """Catalog coverage: fraction of distinct items ever recommended."""
    if catalog_size == 0:
        return 0.0
    return len(set(all_recommended)) / catalog_size


def intra_list_diversity(recommended: Sequence[int],
                         genre_vectors: Dict[int, np.ndarray]) -> float:
    """Mean pairwise (1 - cosine similarity) over the recommended items."""
    vecs = [genre_vectors[i] for i in recommended if i in genre_vectors]
    if len(vecs) < 2:
        return 0.0
    dists, pairs = 0.0, 0
    for a in range(len(vecs)):
        for b in range(a + 1, len(vecs)):
            va, vb = vecs[a], vecs[b]
            na, nb = np.linalg.norm(va), np.linalg.norm(vb)
            cos = (va @ vb) / (na * nb) if na > 0 and nb > 0 else 0.0
            dists += 1.0 - cos
            pairs += 1
    return dists / pairs if pairs else 0.0


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))
