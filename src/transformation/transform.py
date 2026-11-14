"""Stage 4 — Data Transformation.

Builds reusable derived datasets from the TRAINING interactions (so nothing
leaks from the test period):

  * ``user_activity_summary`` — per-user interaction counts, rating stats, tenure
  * ``item_popularity``       — per-item interaction counts, mean rating, rank
  * ``user_item_matrix``      — the user x item rating matrix (sparse-friendly)
  * ``category_popularity``   — interactions & mean rating per genre/category

Outputs to ``data/processed``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.utils import ensure_dir, get_logger, load_config, resolve_path

logger = get_logger("transformation")


def _user_activity_summary(train: pd.DataFrame) -> pd.DataFrame:
    g = train.groupby("user_id")
    summary = g.agg(
        interaction_count=("item_id", "count"),
        avg_rating=("rating", "mean"),
        rating_std=("rating", "std"),
        min_rating=("rating", "min"),
        max_rating=("rating", "max"),
        first_interaction=("event_time", "min"),
        last_interaction=("event_time", "max"),
        distinct_items=("item_id", "nunique"),
    ).reset_index()
    summary["rating_std"] = summary["rating_std"].fillna(0.0)
    summary["active_days"] = (
        summary["last_interaction"] - summary["first_interaction"]
    ).dt.days.clip(lower=0)
    return summary


def _item_popularity(train: pd.DataFrame) -> pd.DataFrame:
    g = train.groupby("item_id")
    pop = g.agg(
        interaction_count=("user_id", "count"),
        distinct_users=("user_id", "nunique"),
        avg_rating=("rating", "mean"),
        rating_std=("rating", "std"),
    ).reset_index()
    pop["rating_std"] = pop["rating_std"].fillna(0.0)
    pop = pop.sort_values("interaction_count", ascending=False)
    pop["popularity_rank"] = np.arange(1, len(pop) + 1)
    # Bayesian/damped popularity score: blends count with mean rating.
    global_mean = train["rating"].mean()
    m = pop["interaction_count"].quantile(0.6)  # prior strength
    pop["popularity_score"] = (
        (pop["interaction_count"] * pop["avg_rating"] + m * global_mean)
        / (pop["interaction_count"] + m)
    )
    return pop.reset_index(drop=True)


def _category_popularity(train: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    merged = train.merge(items[["item_id", "primary_category"]], on="item_id", how="left")
    cat = merged.groupby("primary_category").agg(
        interaction_count=("item_id", "count"),
        avg_rating=("rating", "mean"),
        distinct_items=("item_id", "nunique"),
    ).reset_index().sort_values("interaction_count", ascending=False)
    return cat.reset_index(drop=True)


def _user_item_matrix(train: pd.DataFrame) -> pd.DataFrame:
    matrix = train.pivot_table(
        index="user_id", columns="item_id", values="rating", aggfunc="mean"
    )
    return matrix


def run(config_path: str | None = None) -> Dict[str, str]:
    cfg = load_config(config_path)
    proc = resolve_path(cfg["paths"]["processed"])
    ensure_dir(proc)

    train = pd.read_parquet(proc / "train.parquet")
    items = pd.read_parquet(proc / "items.parquet")

    user_summary = _user_activity_summary(train)
    item_pop = _item_popularity(train)
    cat_pop = _category_popularity(train, items)
    matrix = _user_item_matrix(train)

    user_summary.to_parquet(proc / "user_activity_summary.parquet", index=False)
    user_summary.to_csv(proc / "user_activity_summary.csv", index=False)
    item_pop.to_parquet(proc / "item_popularity.parquet", index=False)
    item_pop.to_csv(proc / "item_popularity.csv", index=False)
    cat_pop.to_parquet(proc / "category_popularity.parquet", index=False)
    cat_pop.to_csv(proc / "category_popularity.csv", index=False)
    # Matrix kept as parquet (index preserved); it can be large, so no CSV mirror.
    matrix.to_parquet(proc / "user_item_matrix.parquet")

    density = train.shape[0] / (matrix.shape[0] * matrix.shape[1])
    logger.info(
        "Transformation complete -> user_summary=%d, item_pop=%d, "
        "matrix=%dx%d (density=%.3f%%)",
        len(user_summary), len(item_pop), matrix.shape[0], matrix.shape[1],
        density * 100,
    )
    return {"processed_dir": str(proc)}


if __name__ == "__main__":
    run()
