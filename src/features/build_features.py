"""Stage 5 — Feature Management.

Creates user, item, and interaction features and persists them to
``data/features`` as Parquet. A small ``FeatureStore`` class provides the SAME
feature-retrieval API for both training and prediction, so features are defined
once and reused everywhere (no training/serving skew).

A machine-readable feature catalog (``feature_catalog.csv`` / ``.md``) is
generated alongside the features.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.utils import ensure_dir, get_logger, load_config, resolve_path

logger = get_logger("features")


# ---------------------------------------------------------------------------
# Feature catalog (documentation kept in sync with the code)
# ---------------------------------------------------------------------------
FEATURE_CATALOG: List[Dict[str, str]] = [
    # user features
    {"entity": "user", "feature": "user_avg_rating", "dtype": "float",
     "description": "Mean rating the user has given (train period)."},
    {"entity": "user", "feature": "user_activity_count", "dtype": "int",
     "description": "Number of interactions by the user."},
    {"entity": "user", "feature": "user_rating_std", "dtype": "float",
     "description": "Std-dev of the user's ratings (rating dispersion)."},
    {"entity": "user", "feature": "user_distinct_items", "dtype": "int",
     "description": "Count of distinct items the user interacted with."},
    {"entity": "user", "feature": "user_active_days", "dtype": "int",
     "description": "Days between the user's first and last interaction."},
    {"entity": "user", "feature": "user_days_since_last", "dtype": "int",
     "description": "Days from the user's last interaction to the reference time."},
    {"entity": "user", "feature": "user_preferred_category", "dtype": "category",
     "description": "Genre/category the user interacts with most often."},
    {"entity": "user", "feature": "age", "dtype": "int",
     "description": "User age (demographic)."},
    {"entity": "user", "feature": "gender", "dtype": "category",
     "description": "User gender (demographic)."},
    {"entity": "user", "feature": "occupation", "dtype": "category",
     "description": "User occupation (demographic)."},
    # item features
    {"entity": "item", "feature": "item_popularity", "dtype": "int",
     "description": "Number of interactions the item received."},
    {"entity": "item", "feature": "item_distinct_users", "dtype": "int",
     "description": "Count of distinct users who interacted with the item."},
    {"entity": "item", "feature": "item_avg_rating", "dtype": "float",
     "description": "Mean rating received by the item."},
    {"entity": "item", "feature": "item_rating_std", "dtype": "float",
     "description": "Std-dev of ratings received by the item."},
    {"entity": "item", "feature": "item_popularity_rank", "dtype": "int",
     "description": "Rank of the item by interaction count (1 = most popular)."},
    {"entity": "item", "feature": "item_popularity_score", "dtype": "float",
     "description": "Damped popularity score blending count and mean rating."},
    {"entity": "item", "feature": "primary_category", "dtype": "category",
     "description": "Primary genre/category of the item."},
    {"entity": "item", "feature": "item_genre_vector", "dtype": "int[]",
     "description": "Multi-hot genre flags (used for content/diversity metrics)."},
    # interaction features
    {"entity": "interaction", "feature": "days_since_last_user_interaction",
     "dtype": "int",
     "description": "Days since the user's previous interaction at event time."},
    {"entity": "interaction", "feature": "user_item_category_match", "dtype": "int",
     "description": "1 if the item's category equals the user's preferred category."},
    {"entity": "interaction", "feature": "rating", "dtype": "float",
     "description": "Target variable: the observed rating."},
]


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------
def _genre_columns(items: pd.DataFrame) -> List[str]:
    known = ["unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
             "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
             "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    return [c for c in known if c in items.columns]


def build_user_features(train: pd.DataFrame, users: pd.DataFrame,
                        items: pd.DataFrame, reference_time: pd.Timestamp
                        ) -> pd.DataFrame:
    g = train.groupby("user_id")
    feat = g.agg(
        user_avg_rating=("rating", "mean"),
        user_activity_count=("item_id", "count"),
        user_rating_std=("rating", "std"),
        user_distinct_items=("item_id", "nunique"),
        first_interaction=("event_time", "min"),
        last_interaction=("event_time", "max"),
    ).reset_index()
    feat["user_rating_std"] = feat["user_rating_std"].fillna(0.0)
    feat["user_active_days"] = (
        feat["last_interaction"] - feat["first_interaction"]
    ).dt.days.clip(lower=0)
    feat["user_days_since_last"] = (
        reference_time - feat["last_interaction"]
    ).dt.days.clip(lower=0)

    # Preferred category = the user's modal item category.
    merged = train.merge(items[["item_id", "primary_category"]], on="item_id", how="left")
    pref = (
        merged.groupby("user_id")["primary_category"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "unknown")
        .rename("user_preferred_category").reset_index()
    )
    feat = feat.merge(pref, on="user_id", how="left")

    demo_cols = [c for c in ["user_id", "age", "gender", "occupation"] if c in users.columns]
    feat = feat.merge(users[demo_cols], on="user_id", how="left")
    feat = feat.drop(columns=["first_interaction", "last_interaction"])
    return feat


def build_item_features(train: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    g = train.groupby("item_id")
    feat = g.agg(
        item_popularity=("user_id", "count"),
        item_distinct_users=("user_id", "nunique"),
        item_avg_rating=("rating", "mean"),
        item_rating_std=("rating", "std"),
    ).reset_index()
    feat["item_rating_std"] = feat["item_rating_std"].fillna(0.0)
    feat = feat.sort_values("item_popularity", ascending=False)
    feat["item_popularity_rank"] = np.arange(1, len(feat) + 1)

    global_mean = train["rating"].mean()
    m = feat["item_popularity"].quantile(0.6)
    feat["item_popularity_score"] = (
        (feat["item_popularity"] * feat["item_avg_rating"] + m * global_mean)
        / (feat["item_popularity"] + m)
    )

    genre_cols = _genre_columns(items)
    meta_cols = ["item_id", "primary_category"] + genre_cols
    feat = feat.merge(items[meta_cols], on="item_id", how="left")
    feat["primary_category"] = feat["primary_category"].fillna("unknown")
    return feat.reset_index(drop=True)


def build_interaction_features(pairs: pd.DataFrame, user_features: pd.DataFrame,
                               items: pd.DataFrame) -> pd.DataFrame:
    """Reusable at train AND predict time: enrich (user,item) pairs.

    ``pairs`` must contain at least ``user_id`` and ``item_id`` (``rating`` and
    ``event_time`` optional). Returns the pairs joined with user/item features
    plus interaction-level features.
    """
    df = pairs.copy()
    item_cat = items[["item_id", "primary_category"]]
    df = df.merge(item_cat, on="item_id", how="left")
    df = df.merge(
        user_features[["user_id", "user_preferred_category", "user_avg_rating"]],
        on="user_id", how="left",
    )
    df["user_item_category_match"] = (
        df["primary_category"] == df["user_preferred_category"]
    ).astype(int)

    if "event_time" in df.columns:
        df = df.sort_values(["user_id", "event_time"])
        prev = df.groupby("user_id")["event_time"].shift(1)
        df["days_since_last_user_interaction"] = (
            (df["event_time"] - prev).dt.days.fillna(-1).astype(int)
        )
    return df


# ---------------------------------------------------------------------------
# Feature store — one API for training and prediction
# ---------------------------------------------------------------------------
class FeatureStore:
    """Thin retrieval layer over the persisted feature tables."""

    def __init__(self, features_dir: Path):
        self.dir = Path(features_dir)
        self.user_features = pd.read_parquet(self.dir / "user_features.parquet")
        self.item_features = pd.read_parquet(self.dir / "item_features.parquet")
        self._uf = self.user_features.set_index("user_id")
        self._if = self.item_features.set_index("item_id")

    def get_user_features(self, user_id: int) -> Dict:
        return self._uf.loc[user_id].to_dict() if user_id in self._uf.index else {}

    def get_item_features(self, item_id: int) -> Dict:
        return self._if.loc[item_id].to_dict() if item_id in self._if.index else {}

    def genre_matrix(self) -> pd.DataFrame:
        cols = _genre_columns(self.item_features)
        return self.item_features.set_index("item_id")[cols]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run(config_path: str | None = None) -> Dict[str, str]:
    cfg = load_config(config_path)
    proc = resolve_path(cfg["paths"]["processed"])
    feat_dir = ensure_dir(resolve_path(cfg["paths"]["features"]))
    reports_dir = ensure_dir(resolve_path(cfg["paths"]["reports"]))

    train = pd.read_parquet(proc / "train.parquet")
    users = pd.read_parquet(proc / "users.parquet")
    items = pd.read_parquet(proc / "items.parquet")

    reference_time = train["event_time"].max()
    user_features = build_user_features(train, users, items, reference_time)
    item_features = build_item_features(train, items)
    interaction_features = build_interaction_features(train, user_features, items)

    user_features.to_parquet(feat_dir / "user_features.parquet", index=False)
    user_features.to_csv(feat_dir / "user_features.csv", index=False)
    item_features.to_parquet(feat_dir / "item_features.parquet", index=False)
    item_features.to_csv(feat_dir / "item_features.csv", index=False)
    interaction_features.to_parquet(feat_dir / "interaction_features.parquet", index=False)

    # Feature catalog (CSV + Markdown).
    catalog = pd.DataFrame(FEATURE_CATALOG)
    catalog.to_csv(feat_dir / "feature_catalog.csv", index=False)
    _write_catalog_md(catalog, reports_dir / "feature_catalog.md")

    logger.info("Feature management complete -> %d user feats, %d item feats, "
                "%d interaction rows, %d cataloged features",
                len(user_features), len(item_features),
                len(interaction_features), len(catalog))
    return {"features_dir": str(feat_dir)}


def _write_catalog_md(catalog: pd.DataFrame, path: Path) -> None:
    lines = ["# Feature Catalog", "",
             "All features are defined once in `src/features/build_features.py` "
             "and served through the `FeatureStore` class, so the same definitions "
             "are used for both training and prediction.", ""]
    for entity in ["user", "item", "interaction"]:
        sub = catalog[catalog["entity"] == entity]
        lines += [f"## {entity.capitalize()} features", "",
                  "| Feature | Type | Description |",
                  "|---------|------|-------------|"]
        for _, r in sub.iterrows():
            lines.append(f"| `{r['feature']}` | {r['dtype']} | {r['description']} |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
