"""Stage 1 — Data Ingestion.

Ingests three canonical tables into ``data/raw`` as CSV:

  * ``users.csv``        — user details
  * ``items.csv``        — item (movie) details, including genre columns
  * ``interactions.csv`` — user-item interactions (ratings + timestamps)

Primary source is MovieLens 100K. If the download fails (e.g. no network) and
``allow_synthetic_fallback`` is set, a clearly documented synthetic dataset is
generated instead so the pipeline always runs end-to-end.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.utils import ensure_dir, get_logger, load_config, resolve_path

logger = get_logger("ingestion")

# MovieLens 100K genre columns, in file order (see the ml-100k README).
GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


# ---------------------------------------------------------------------------
# MovieLens 100K
# ---------------------------------------------------------------------------
def _download_movielens(url: str) -> bytes:
    import requests  # imported lazily so synthetic-only runs need no network lib

    logger.info("Downloading MovieLens 100K from %s", url)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content


def _parse_movielens(zip_bytes: bytes) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        def read(name: str) -> bytes:
            inner = [n for n in zf.namelist() if n.endswith(name)][0]
            return zf.read(inner)

        # Interactions: u.data -> user_id, item_id, rating, timestamp(unix)
        interactions = pd.read_csv(
            io.BytesIO(read("u.data")),
            sep="\t",
            names=["user_id", "item_id", "rating", "timestamp"],
            encoding="latin-1",
        )

        # Items: u.item -> id | title | release | video | url | 19 genre flags
        item_cols = ["item_id", "title", "release_date", "video_release", "imdb_url"] + GENRES
        items = pd.read_csv(
            io.BytesIO(read("u.item")),
            sep="|",
            names=item_cols,
            encoding="latin-1",
        )

        # Users: u.user -> id | age | gender | occupation | zip
        users = pd.read_csv(
            io.BytesIO(read("u.user")),
            sep="|",
            names=["user_id", "age", "gender", "occupation", "zip_code"],
            encoding="latin-1",
        )

    interactions["event_type"] = "rating"
    items = _finalize_items(items)
    return users, items, interactions


def _finalize_items(items: pd.DataFrame) -> pd.DataFrame:
    """Add a pipe-joined ``genres`` string and a ``primary_category`` column."""
    def genre_list(row) -> str:
        active = [g for g in GENRES if row.get(g, 0) == 1]
        return "|".join(active) if active else "unknown"

    items["genres"] = items.apply(genre_list, axis=1)
    items["primary_category"] = items["genres"].str.split("|").str[0]
    keep = ["item_id", "title", "release_date", "genres", "primary_category"] + GENRES
    return items[[c for c in keep if c in items.columns]]


# ---------------------------------------------------------------------------
# Synthetic generator (fallback / offline demonstration)
# ---------------------------------------------------------------------------
def _generate_synthetic(cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    s = cfg["ingestion"]["synthetic"]
    seed = cfg["project"]["seed"]
    rng = np.random.default_rng(seed)

    n_users = s["n_users"]
    n_items = s["n_items"]
    n_inter = s["n_interactions"]
    corruption = s.get("corruption_rate", 0.0)

    logger.info(
        "Generating synthetic data: %d users, %d items, %d interactions",
        n_users, n_items, n_inter,
    )

    # Users
    occupations = ["student", "engineer", "artist", "doctor", "teacher", "other"]
    users = pd.DataFrame({
        "user_id": np.arange(1, n_users + 1),
        "age": rng.integers(15, 70, n_users),
        "gender": rng.choice(["M", "F"], n_users),
        "occupation": rng.choice(occupations, n_users),
        "zip_code": rng.integers(10000, 99999, n_users).astype(str),
    })

    # Items with 1-3 genres each
    non_unknown = [g for g in GENRES if g != "unknown"]
    items_rows = []
    for item_id in range(1, n_items + 1):
        k = rng.integers(1, 4)
        chosen = list(rng.choice(non_unknown, size=k, replace=False))
        flags = {g: (1 if g in chosen else 0) for g in GENRES}
        items_rows.append({
            "item_id": item_id,
            "title": f"Item_{item_id}",
            "release_date": "01-Jan-2000",
            "genres": "|".join(chosen),
            "primary_category": chosen[0],
            **flags,
        })
    items = pd.DataFrame(items_rows)

    # Latent preferences drive ratings so collaborative filtering has signal.
    k_lat = 8
    user_lat = rng.normal(0, 1, (n_users, k_lat))
    item_lat = rng.normal(0, 1, (n_items, k_lat))

    # Popularity skew (Zipf-like) so a popularity baseline is meaningful.
    item_pop = rng.zipf(1.3, n_items).astype(float)
    item_pop = np.clip(item_pop, 1, n_items)
    item_prob = item_pop / item_pop.sum()

    base_ts = pd.Timestamp("2022-01-01").value // 10**9  # unix seconds
    span = 365 * 24 * 3600  # one year

    u_idx = rng.integers(0, n_users, n_inter)
    i_idx = rng.choice(n_items, size=n_inter, p=item_prob)
    raw_score = (user_lat[u_idx] * item_lat[i_idx]).sum(axis=1)
    # Map latent score to a 1..5 rating via its rank (quantile buckets).
    ranks = raw_score.argsort().argsort() / max(len(raw_score) - 1, 1)
    ratings = np.clip(np.round(1 + ranks * 4), 1, 5).astype(int)
    timestamps = base_ts + rng.integers(0, span, n_inter)

    interactions = pd.DataFrame({
        "user_id": u_idx + 1,
        "item_id": i_idx + 1,
        "rating": ratings,
        "timestamp": timestamps,
        "event_type": "rating",
    })
    # De-duplicate (user,item): keep the most recent interaction.
    interactions = (
        interactions.sort_values("timestamp")
        .drop_duplicates(["user_id", "item_id"], keep="last")
        .reset_index(drop=True)
    )

    if corruption > 0:
        interactions = _inject_errors(interactions, corruption, n_users, n_items, rng)

    return users, items, interactions


def _inject_errors(df: pd.DataFrame, rate: float, n_users: int, n_items: int, rng) -> pd.DataFrame:
    """Deliberately corrupt a fraction of rows so validation has work to do.

    Introduces: out-of-range ratings, null ratings, duplicate rows,
    orphan references (unknown user/item), and impossible timestamps.
    """
    df = df.copy()
    n_bad = max(1, int(len(df) * rate))
    idx = rng.choice(df.index, size=n_bad, replace=False)
    chunks = np.array_split(idx, 5)

    df.loc[chunks[0], "rating"] = 99                       # out-of-range rating
    df.loc[chunks[1], "rating"] = np.nan                    # missing rating
    df.loc[chunks[2], "user_id"] = n_users + 10_000         # orphan user
    df.loc[chunks[3], "item_id"] = n_items + 10_000         # orphan item
    df.loc[chunks[4], "timestamp"] = -5                     # impossible timestamp

    # A handful of exact duplicate rows.
    dup = df.loc[chunks[0]].copy()
    df = pd.concat([df, dup], ignore_index=True)
    logger.info("Injected ~%d corrupted rows + %d duplicates for validation demo",
                n_bad, len(dup))
    return df


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run(config_path: str | None = None) -> Dict[str, str]:
    cfg = load_config(config_path)
    raw_dir = ensure_dir(resolve_path(cfg["paths"]["raw"]))
    source = cfg["ingestion"]["source"].lower()

    users = items = interactions = None
    used_source = source

    if source == "movielens":
        try:
            zip_bytes = _download_movielens(cfg["ingestion"]["movielens_url"])
            users, items, interactions = _parse_movielens(zip_bytes)
            logger.info("MovieLens 100K ingested successfully.")
        except Exception as exc:  # noqa: BLE001
            if cfg["ingestion"].get("allow_synthetic_fallback", True):
                logger.warning("MovieLens ingestion failed (%s). Falling back to synthetic.", exc)
                users, items, interactions = _generate_synthetic(cfg)
                used_source = "synthetic (fallback)"
            else:
                raise

    if users is None:  # source == "synthetic" or not yet produced
        users, items, interactions = _generate_synthetic(cfg)
        used_source = "synthetic"

    paths = {
        "users": str(raw_dir / "users.csv"),
        "items": str(raw_dir / "items.csv"),
        "interactions": str(raw_dir / "interactions.csv"),
    }
    users.to_csv(paths["users"], index=False)
    items.to_csv(paths["items"], index=False)
    interactions.to_csv(paths["interactions"], index=False)

    logger.info(
        "Ingestion complete [source=%s] -> users=%d, items=%d, interactions=%d",
        used_source, len(users), len(items), len(interactions),
    )
    return paths


if __name__ == "__main__":
    run()
