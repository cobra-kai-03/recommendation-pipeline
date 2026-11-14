"""Stage 6 — Model Training.

Trains two models and tracks parameters + artifacts in MLflow:

  1. PopularityRecommender      (baseline)
  2. MatrixFactorization        (personalized, SVD + biases)

Each model is pickled to ``models/`` and registered in an MLflow run. Run IDs
are saved to ``models/mlflow_runs.json`` so the evaluation stage can attach
metrics to the same runs.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict

import pandas as pd

from src.training.matrix_factorization import MatrixFactorization
from src.training.popularity import PopularityRecommender
from src.utils import ensure_dir, get_logger, load_config, resolve_path

logger = get_logger("training")


def _init_mlflow(cfg: Dict):
    import mlflow
    tracking_uri = resolve_path(cfg["mlflow"]["tracking_uri"]).as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    return mlflow


def run(config_path: str | None = None) -> Dict[str, str]:
    cfg = load_config(config_path)
    proc = resolve_path(cfg["paths"]["processed"])
    models_dir = ensure_dir(resolve_path(cfg["paths"]["models"]))
    seed = cfg["project"]["seed"]

    train = pd.read_parquet(proc / "train.parquet")
    mlflow = _init_mlflow(cfg)

    run_ids: Dict[str, str] = {}

    # --- Model 1: popularity baseline --------------------------------------
    with mlflow.start_run(run_name="popularity") as run_pop:
        pop = PopularityRecommender(min_interactions=1).fit(train)
        mlflow.log_params(pop.params)
        mlflow.log_param("n_train_interactions", len(train))
        with open(models_dir / "popularity.pkl", "wb") as fh:
            pickle.dump(pop, fh)
        mlflow.log_artifact(str(models_dir / "popularity.pkl"))
        run_ids["popularity"] = run_pop.info.run_id
        logger.info("Trained PopularityRecommender over %d catalog items",
                    len(pop.ranked_items_))

    # --- Model 2: matrix factorization -------------------------------------
    mf_cfg = cfg["training"]["matrix_factorization"]
    with mlflow.start_run(run_name="matrix_factorization") as run_mf:
        mf = MatrixFactorization(seed=seed, **mf_cfg).fit(train)
        mlflow.log_params(mf.params)
        mlflow.log_metric("train_rmse", mf.train_rmse_)
        with open(models_dir / "matrix_factorization.pkl", "wb") as fh:
            pickle.dump(mf, fh)
        mlflow.log_artifact(str(models_dir / "matrix_factorization.pkl"))
        run_ids["matrix_factorization"] = run_mf.info.run_id
        logger.info("Trained MatrixFactorization (train RMSE=%.4f)", mf.train_rmse_)

    with open(models_dir / "mlflow_runs.json", "w", encoding="utf-8") as fh:
        json.dump(run_ids, fh, indent=2)

    logger.info("Training complete -> models saved to %s", models_dir)
    return {"models_dir": str(models_dir), "run_ids": run_ids}


if __name__ == "__main__":
    run()
