"""Apache Airflow DAG for the recommendation pipeline.

Mirrors the pure-Python orchestrator (`pipeline.py`) as a real Airflow DAG so the
workflow can be scheduled and visualized in the Airflow UI (for the
"screenshot of a successful run" deliverable).

Note on imports: the heavy pipeline modules (pandas/numpy/mlflow/matplotlib) are
imported *lazily inside each task callable*, never at module top level. This is
an Airflow best practice — DAG files are parsed frequently by the scheduler and
webserver, so keeping top-level imports light keeps parsing fast and avoids
importing native libraries inside forked webserver workers.

Setup
-----
    pip install "apache-airflow>=2.6"
    export AIRFLOW_HOME=~/airflow
    airflow db migrate
    export AIRFLOW__CORE__DAGS_FOLDER=/absolute/path/to/recommendation-pipeline/dags
    export PYTHONPATH=/absolute/path/to/recommendation-pipeline
    airflow standalone        # scheduler + webserver on :8080

Then trigger the ``recommendation_pipeline`` DAG from the UI. A convenience
launcher is provided at ``scripts/run_airflow.sh``.

Task dependencies:
    ingestion -> validation -> preparation -> transformation
              -> features -> training -> evaluation
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

# Make the project package importable regardless of where Airflow runs.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from airflow import DAG
from airflow.operators.python import PythonOperator


# --- Task callables: import the stage lazily, then run it --------------------
def ingestion_task(**_):
    from src.ingestion.ingest import run
    return run()


def validation_task(**_):
    from src.validation.validate import run
    return run()


def preparation_task(**_):
    from src.preparation.prepare import run
    return run()


def transformation_task(**_):
    from src.transformation.transform import run
    return run()


def features_task(**_):
    from src.features.build_features import run
    return run()


def training_task(**_):
    from src.training.train import run
    return run()


def evaluation_task(**_):
    from src.evaluation.evaluate import run
    return run()


default_args = {
    "owner": "dm4ml",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="recommendation_pipeline",
    description="End-to-end data management pipeline for a recommendation system",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["dm4ml", "recommendation", "mlops"],
) as dag:

    ingestion = PythonOperator(task_id="ingestion", python_callable=ingestion_task)
    validation = PythonOperator(task_id="validation", python_callable=validation_task)
    preparation = PythonOperator(task_id="preparation", python_callable=preparation_task)
    transformation = PythonOperator(task_id="transformation", python_callable=transformation_task)
    features = PythonOperator(task_id="features", python_callable=features_task)
    training = PythonOperator(task_id="training", python_callable=training_task)
    evaluation = PythonOperator(task_id="evaluation", python_callable=evaluation_task)

    (
        ingestion
        >> validation
        >> preparation
        >> transformation
        >> features
        >> training
        >> evaluation
    )
