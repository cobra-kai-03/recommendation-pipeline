#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Launch Airflow locally and run the recommendation_pipeline DAG.
#
# Usage:
#   bash scripts/run_airflow.sh
#
# Then open http://localhost:8080 (the admin password is printed by
# `airflow standalone`, or set to anonymous access via airflow_home/webserver_config.py)
# and trigger / inspect the `recommendation_pipeline` DAG.
# -----------------------------------------------------------------------------
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export AIRFLOW_HOME="$PROJECT_ROOT/airflow_home"
export AIRFLOW__CORE__DAGS_FOLDER="$PROJECT_ROOT/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export PYTHONPATH="$PROJECT_ROOT"

echo "AIRFLOW_HOME=$AIRFLOW_HOME"
echo "DAGS_FOLDER=$AIRFLOW__CORE__DAGS_FOLDER"

# Initialise the metadata DB (idempotent).
airflow db migrate

# Option A — one-shot, headless execution of the whole DAG (no webserver):
#   airflow dags test recommendation_pipeline "$(date +%Y-%m-%d)"
#
# Option B — full UI for the screenshot deliverable:
#   airflow standalone      # scheduler + gunicorn webserver on :8080
#
# Option C — UI when gunicorn workers segfault on fork (e.g. macOS system
# Python). Start the scheduler, then serve the UI single-process:
#   airflow scheduler &
#   python "$PROJECT_ROOT/scripts/serve_airflow_ui.py"   # UI on :8080

airflow standalone
