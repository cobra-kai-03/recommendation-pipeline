"""Baseline model — Popularity recommender.

Recommends the globally most popular items (by a damped popularity score that
blends interaction count with mean rating), excluding items the user has already
seen. The same ranked list underlies every user's recommendations, which is
exactly what makes it a strong, simple baseline.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


class PopularityRecommender:
    name = "popularity"

    def __init__(self, min_interactions: int = 1):
        self.min_interactions = min_interactions
        self.ranked_items_: List[int] = []
        self.item_score_: dict = {}

    def fit(self, train: pd.DataFrame) -> "PopularityRecommender":
        g = train.groupby("item_id").agg(
            count=("user_id", "count"), avg_rating=("rating", "mean")
        )
        g = g[g["count"] >= self.min_interactions]
        global_mean = train["rating"].mean()
        m = g["count"].quantile(0.6) if len(g) else 0.0
        score = (g["count"] * g["avg_rating"] + m * global_mean) / (g["count"] + m)
        score = score.sort_values(ascending=False)
        self.ranked_items_ = list(score.index)
        self.item_score_ = score.to_dict()
        return self

    def recommend(self, user_id: int, k: int = 10,
                  exclude: Optional[Iterable[int]] = None) -> List[int]:
        exclude = set(exclude or [])
        recs = [i for i in self.ranked_items_ if i not in exclude]
        return recs[:k]

    @property
    def params(self) -> dict:
        return {"model": self.name, "min_interactions": self.min_interactions}
