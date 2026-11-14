# End-to-End Data Management Pipeline for a Recommendation System

> DM4ML – Assignment 2. A clean, modular, reproducible pipeline that ingests raw
> data, validates it, prepares and transforms it, builds features, trains and
> evaluates two recommender models, and orchestrates the whole workflow.

**Dataset:** [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)
(auto-downloaded) with an automatic, documented **synthetic fallback** so the
pipeline runs end-to-end even with no network access.

---

## Pipeline stages

```
Data Ingestion → Data Validation → Data Preparation → Data Transformation
→ Feature Management → Model Training → Model Evaluation → Orchestration
```

See [`reports/architecture.md`](reports/architecture.md) (Mermaid diagram) and
[`reports/architecture_diagram.png`](reports/architecture_diagram.png).

| Stage | Module | Output |
|-------|--------|--------|
| 1. Ingestion | [`src/ingestion/ingest.py`](src/ingestion/ingest.py) | `data/raw/{users,items,interactions}.csv` |
| 2. Validation | [`src/validation/validate.py`](src/validation/validate.py) | `data/errors/*`, `reports/validation_report.md` |
| 3. Preparation | [`src/preparation/prepare.py`](src/preparation/prepare.py) | `data/processed/{train,test,...}.parquet` |
| 4. Transformation | [`src/transformation/transform.py`](src/transformation/transform.py) | activity summary, item popularity, user-item matrix |
| 5. Feature Management | [`src/features/build_features.py`](src/features/build_features.py) | `data/features/*`, `reports/feature_catalog.md` |
| 6. Model Training | [`src/training/train.py`](src/training/train.py) | `models/*.pkl`, MLflow runs |
| 7. Model Evaluation | [`src/evaluation/evaluate.py`](src/evaluation/evaluate.py) | `reports/model_evaluation.md` |
| 8. Orchestration | [`pipeline.py`](pipeline.py), [`dags/recommendation_dag.py`](dags/recommendation_dag.py) | `reports/pipeline_run.md` |

---

## Setup

Requires **Python 3.9+**.

```bash
# 1. clone, then from the project root:
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Run the full pipeline

```bash
python pipeline.py
```

This runs all eight stages in dependency order, printing a per-stage log and
writing a run summary to `reports/pipeline_run.md`. All artifacts land under
`data/`, `models/`, and `reports/`.

### Run selected stages

```bash
python pipeline.py --list                 # show stages + dependencies
python pipeline.py --stage validation     # run a single stage
python pipeline.py --from transformation  # run from a stage to the end
```

### Run a stage module directly

```bash
python -m src.ingestion.ingest
python -m src.evaluation.evaluate
```

---

## Configuration

Everything is driven by [`configs/config.yaml`](configs/config.yaml):

- `ingestion.source`: `movielens` (default) or `synthetic`
- `preparation.test_fraction`: per-user temporal hold-out size (default 0.2)
- `evaluation.k`: top-K for ranking metrics (set to **5** or **10**)
- `training.matrix_factorization`: factors / epochs / learning rate / regularization
- `project.seed`: global seed for full reproducibility

To force the offline synthetic dataset (which also demonstrates the validation
error-quarantine, since it injects corrupted rows):

```yaml
ingestion:
  source: synthetic
```

---

## Experiment tracking (MLflow)

Training logs parameters + artifacts and evaluation logs metrics to a local
MLflow store (`./mlruns`). View them with:

```bash
mlflow ui --backend-store-uri ./mlruns
# open http://127.0.0.1:5000
```

---

## Orchestration with Airflow (optional)

A production-style DAG mirrors the pipeline. See the header of
[`dags/recommendation_dag.py`](dags/recommendation_dag.py) for setup:

```bash
pip install "apache-airflow>=2.6"
export AIRFLOW_HOME=~/airflow && airflow db init
cp dags/recommendation_dag.py $AIRFLOW_HOME/dags/
export PYTHONPATH=$(pwd)
airflow standalone      # UI at http://localhost:8080
```

Trigger the `recommendation_pipeline` DAG and screenshot the green run for the
deliverable. (The pure-Python `pipeline.py` requires no extra setup and is the
default, always-runnable orchestrator.)

**Evidence of a successful Airflow run** (all 7 tasks `success`, captured from
the live Airflow UI):

- [reports/airflow_ui_graph.png](reports/airflow_ui_graph.png) — Graph view (the task DAG, all green)
- [reports/airflow_ui_grid.png](reports/airflow_ui_grid.png) — Grid + run details (Total success: 1, Total Tasks: 7)
- [reports/airflow_ui_home.png](reports/airflow_ui_home.png) — DAGs list
- [reports/airflow_run_log.txt](reports/airflow_run_log.txt) — full Airflow execution log

> Platform note: Airflow's default webserver uses **gunicorn**, whose forked
> workers segfault under this machine's Apple CommandLineTools Python 3.9. The
> UI above was served single-process via [scripts/serve_airflow_ui.py](scripts/serve_airflow_ui.py)
> (same Flask app, no fork) — run `airflow scheduler &` then
> `python scripts/serve_airflow_ui.py` and open `http://localhost:8080`. On most
> hosts the standard `airflow standalone` works directly.

---

## Models

| Model | Type | Predicts ratings? |
|-------|------|-------------------|
| **PopularityRecommender** | Baseline — damped popularity score | No (ranking only) |
| **MatrixFactorization** | SVD + biases, trained by SGD | Yes → RMSE/MAE |

---

## Project structure

```
recommendation-pipeline/
├── data/{raw,processed,features,errors}/
├── src/{ingestion,validation,preparation,transformation,features,training,evaluation}/
├── dags/recommendation_dag.py
├── configs/config.yaml
├── models/
├── reports/            # validation report, feature catalog, eval results, diagram, final report
├── scripts/generate_architecture_diagram.py
├── pipeline.py         # pure-Python orchestrator
├── requirements.txt
└── README.md
```

---

## Reproducibility checklist

- ✅ Single config file + fixed random seed
- ✅ Deterministic time-based split (no leakage)
- ✅ Features built from training data only, reused at prediction via `FeatureStore`
- ✅ One command (`python pipeline.py`) runs everything
- ✅ Works offline via synthetic fallback
- ✅ Artifacts + logs written for every stage

See [`reports/final_report.md`](reports/final_report.md) for design decisions,
results, and limitations.
