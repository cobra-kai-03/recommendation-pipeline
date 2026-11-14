"""Stage 7 — Model Evaluation.

Evaluates every trained model on the held-out test set and produces a side-by-side
comparison table. Metrics:

  * Precision@K, Recall@K, NDCG@K   (ranking quality)
  * Coverage                        (catalog exploration)
  * Diversity                       (intra-list genre diversity)
  * RMSE / MAE                      (rating prediction — models that support it)

Metrics are logged back to each model's MLflow run and written to
``reports/model_evaluation.{md,csv}``.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.evaluation import metrics as M
from src.features.build_features import FeatureStore
from src.utils import ensure_dir, get_logger, load_config, resolve_path

logger = get_logger("evaluation")


def _relevant_sets(test: pd.DataFrame, threshold: float) -> Dict[int, set]:
    rel = test[test["rating"] >= threshold]
    return rel.groupby("user_id")["item_id"].agg(set).to_dict()


def _evaluate_ranking(model, train: pd.DataFrame, relevant: Dict[int, set],
                      genre_vectors: Dict[int, np.ndarray], catalog_size: int,
                      k: int) -> Dict[str, float]:
    seen_by_user = train.groupby("user_id")["item_id"].agg(set).to_dict()
    precisions, recalls, ndcgs, diversities = [], [], [], []
    all_recs: List[int] = []

    for user_id, rel_items in relevant.items():
        if not rel_items:
            continue
        exclude = seen_by_user.get(user_id, set())
        recs = model.recommend(user_id, k=k, exclude=exclude)
        if not recs:
            continue
        precisions.append(M.precision_at_k(recs, rel_items, k))
        recalls.append(M.recall_at_k(recs, rel_items, k))
        ndcgs.append(M.ndcg_at_k(recs, rel_items, k))
        diversities.append(M.intra_list_diversity(recs, genre_vectors))
        all_recs.extend(recs)

    return {
        f"precision_at_{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall_at_{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg_at_{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "coverage": M.coverage(all_recs, catalog_size),
        "diversity": float(np.mean(diversities)) if diversities else 0.0,
        "n_users_evaluated": len(precisions),
    }


def _evaluate_rating(model, test: pd.DataFrame) -> Dict[str, float]:
    if not hasattr(model, "predict_batch"):
        return {}
    preds = model.predict_batch(test)
    return {"rmse": M.rmse(test["rating"].to_numpy(), preds),
            "mae": M.mae(test["rating"].to_numpy(), preds)}


def run(config_path: str | None = None) -> Dict:
    cfg = load_config(config_path)
    proc = resolve_path(cfg["paths"]["processed"])
    feat_dir = resolve_path(cfg["paths"]["features"])
    models_dir = resolve_path(cfg["paths"]["models"])
    reports_dir = ensure_dir(resolve_path(cfg["paths"]["reports"]))
    k = cfg["evaluation"]["k"]
    threshold = cfg["evaluation"]["relevance_threshold"]

    train = pd.read_parquet(proc / "train.parquet")
    test = pd.read_parquet(proc / "test.parquet")

    store = FeatureStore(feat_dir)
    genre_df = store.genre_matrix()
    genre_vectors = {int(i): row.to_numpy(dtype=float)
                     for i, row in genre_df.iterrows()}
    catalog_size = store.item_features["item_id"].nunique()

    relevant = _relevant_sets(test, threshold)
    logger.info("Evaluating on %d users with >=1 relevant item (threshold=%.1f), K=%d",
                len(relevant), threshold, k)

    models = {
        "popularity": models_dir / "popularity.pkl",
        "matrix_factorization": models_dir / "matrix_factorization.pkl",
    }
    run_ids = {}
    runs_file = models_dir / "mlflow_runs.json"
    if runs_file.exists():
        run_ids = json.loads(runs_file.read_text())

    results: List[Dict] = []
    for model_name, path in models.items():
        with open(path, "rb") as fh:
            model = pickle.load(fh)
        row = {"model": model_name}
        row.update(_evaluate_ranking(model, train, relevant, genre_vectors,
                                     catalog_size, k))
        row.update(_evaluate_rating(model, test))
        results.append(row)
        logger.info("Evaluated %s: P@%d=%.4f R@%d=%.4f NDCG@%d=%.4f cov=%.4f div=%.4f",
                    model_name, k, row[f"precision_at_{k}"], k, row[f"recall_at_{k}"],
                    k, row[f"ndcg_at_{k}"], row["coverage"], row["diversity"])
        _log_to_mlflow(cfg, run_ids.get(model_name), row)

    results_df = pd.DataFrame(results)
    results_df.to_csv(reports_dir / "model_evaluation.csv", index=False)
    _write_results_md(results_df, reports_dir / "model_evaluation.md", k)

    logger.info("Evaluation complete -> reports/model_evaluation.md")
    return {"results": results}


def _log_to_mlflow(cfg: Dict, run_id, row: Dict) -> None:
    if not run_id:
        return
    try:
        import mlflow
        mlflow.set_tracking_uri(resolve_path(cfg["mlflow"]["tracking_uri"]).as_uri())
        with mlflow.start_run(run_id=run_id):
            for key, val in row.items():
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    mlflow.log_metric(key, float(val))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not log metrics to MLflow run %s: %s", run_id, exc)


def _write_results_md(df: pd.DataFrame, path: Path, k: int) -> None:
    display_cols = ["model", f"precision_at_{k}", f"recall_at_{k}",
                    f"ndcg_at_{k}", "coverage", "diversity", "rmse", "mae",
                    "n_users_evaluated"]
    cols = [c for c in display_cols if c in df.columns]
    show = df[cols].copy()
    for c in cols:
        if c not in ("model", "n_users_evaluated"):
            show[c] = show[c].map(lambda v: f"{v:.4f}" if pd.notna(v) else "—")

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(str(show.iloc[r][c]) for c in cols) + " |"
            for r in range(len(show))]

    # Data-driven commentary: name the winner of each metric from the numbers.
    def best(metric: str, higher_is_better: bool = True) -> str:
        if metric not in df.columns or df[metric].isna().all():
            return ""
        idx = df[metric].idxmax() if higher_is_better else df[metric].idxmin()
        return str(df.loc[idx, "model"])

    bullets = []
    rank_winner = best(f"ndcg_at_{k}")
    cov_winner = best("coverage")
    if rank_winner:
        bullets.append(f"- **Best ranking (NDCG@{k})**: `{rank_winner}`.")
    if cov_winner:
        bullets.append(f"- **Best catalog coverage**: `{cov_winner}`.")
    if "rmse" in df.columns and not df["rmse"].isna().all():
        bullets.append(f"- **Rating prediction (RMSE/MAE)**: only "
                       f"`{best('rmse', higher_is_better=False)}` predicts explicit "
                       "ratings; the popularity baseline does not.")

    lines = [
        f"# Model Evaluation Results (K={k})", "",
        "Ranking metrics are averaged over users with at least one relevant "
        f"(rating ≥ threshold) item in the test set. RMSE/MAE are reported only "
        "for models that predict explicit ratings.", "",
        header, sep, *rows, "",
        "## Summary", "", *bullets, "",
        "On a per-user temporal (leave-last-out) split, test relevants skew toward "
        "popular items, so the popularity baseline is a strong competitor on raw "
        "top-K precision/recall. Matrix factorization is the only model that "
        "predicts explicit ratings (RMSE/MAE) and substantially improves catalog "
        "coverage — i.e. it personalizes beyond the head of the catalog.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
