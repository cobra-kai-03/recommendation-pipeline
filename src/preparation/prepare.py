"""Stage 3 — Data Preparation.

Takes the validated tables and produces analysis-ready, standardized data plus a
reproducible train/test split.

Steps:
  * clean & standardize types / formats (datetime, categorical casing)
  * handle any residual missing values on dimension tables
  * drop over-sparse users/items (min interaction thresholds)
  * **time-based train/test split** — per-user leave-last-out: the most recent
    ``test_fraction`` of each user's interactions form the test set.

Outputs Parquet (with CSV mirrors) to ``data/processed``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.utils import ensure_dir, get_logger, load_config, resolve_path

logger = get_logger("preparation")


def _standardize(users: pd.DataFrame, items: pd.DataFrame,
                 interactions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Interactions: enforce dtypes + a proper datetime column.
    interactions = interactions.copy()
    interactions["user_id"] = interactions["user_id"].astype(int)
    interactions["item_id"] = interactions["item_id"].astype(int)
    interactions["rating"] = interactions["rating"].astype(float)
    interactions["timestamp"] = pd.to_numeric(interactions["timestamp"], errors="coerce")
    interactions["event_time"] = pd.to_datetime(interactions["timestamp"], unit="s")
    interactions = interactions.dropna(subset=["event_time"])

    # Users: standardize categorical casing, fill missing demographics.
    users = users.copy()
    users["user_id"] = users["user_id"].astype(int)
    if "gender" in users:
        users["gender"] = users["gender"].astype(str).str.upper().str[0]
    if "occupation" in users:
        users["occupation"] = users["occupation"].astype(str).str.lower().str.strip()
        users["occupation"] = users["occupation"].replace({"nan": "unknown"})
    if "age" in users:
        users["age"] = pd.to_numeric(users["age"], errors="coerce")
        users["age"] = users["age"].fillna(int(users["age"].median()))

    # Items: ensure genre / category present.
    items = items.copy()
    items["item_id"] = items["item_id"].astype(int)
    items["genres"] = items["genres"].fillna("unknown").astype(str)
    items["primary_category"] = (
        items.get("primary_category", items["genres"].str.split("|").str[0])
        .fillna("unknown")
    )
    return users, items, interactions


def _filter_sparse(interactions: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """Iteratively drop users/items below the minimum interaction counts."""
    pcfg = cfg["preparation"]
    min_u = pcfg["min_interactions_per_user"]
    min_i = pcfg["min_interactions_per_item"]

    before = len(interactions)
    while True:
        u_counts = interactions["user_id"].value_counts()
        i_counts = interactions["item_id"].value_counts()
        keep_u = u_counts[u_counts >= min_u].index
        keep_i = i_counts[i_counts >= min_i].index
        filtered = interactions[
            interactions["user_id"].isin(keep_u)
            & interactions["item_id"].isin(keep_i)
        ]
        if len(filtered) == len(interactions):
            break
        interactions = filtered
    logger.info("Sparsity filter (>=%d/user, >=%d/item): %d -> %d interactions",
                min_u, min_i, before, len(interactions))
    return interactions.reset_index(drop=True)


def _time_split(interactions: pd.DataFrame, test_fraction: float
                ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user temporal hold-out (leave-last-out by time)."""
    interactions = interactions.sort_values(["user_id", "timestamp"])
    train_parts, test_parts = [], []
    for _, grp in interactions.groupby("user_id", sort=False):
        n = len(grp)
        n_test = max(1, int(round(n * test_fraction)))
        n_test = min(n_test, n - 1)  # always keep >=1 training row per user
        train_parts.append(grp.iloc[: n - n_test])
        test_parts.append(grp.iloc[n - n_test:])
    train = pd.concat(train_parts).reset_index(drop=True)
    test = pd.concat(test_parts).reset_index(drop=True)
    logger.info("Time-based split: train=%d, test=%d (cutoff per-user, last %.0f%%)",
                len(train), len(test), test_fraction * 100)
    return train, test


def _save(df: pd.DataFrame, directory: Path, name: str) -> None:
    df.to_parquet(directory / f"{name}.parquet", index=False)
    df.to_csv(directory / f"{name}.csv", index=False)


def run(config_path: str | None = None) -> Dict[str, str]:
    cfg = load_config(config_path)
    raw = resolve_path(cfg["paths"]["raw"])
    proc = ensure_dir(resolve_path(cfg["paths"]["processed"]))

    users = pd.read_csv(raw / "users_valid.csv")
    items = pd.read_csv(raw / "items_valid.csv")
    interactions = pd.read_csv(raw / "interactions_valid.csv")

    users, items, interactions = _standardize(users, items, interactions)
    interactions = _filter_sparse(interactions, cfg)

    # Keep only users/items still referenced after filtering.
    users = users[users["user_id"].isin(interactions["user_id"].unique())]
    items = items[items["item_id"].isin(interactions["item_id"].unique())]

    train, test = _time_split(interactions, cfg["preparation"]["test_fraction"])

    _save(users, proc, "users")
    _save(items, proc, "items")
    _save(interactions, proc, "interactions")
    _save(train, proc, "train")
    _save(test, proc, "test")

    logger.info("Preparation complete -> %d users, %d items, train=%d, test=%d",
                len(users), len(items), len(train), len(test))
    return {"processed_dir": str(proc)}


if __name__ == "__main__":
    run()
