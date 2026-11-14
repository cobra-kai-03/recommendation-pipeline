"""Serve the Airflow web UI on a single-process werkzeug server (no gunicorn).

Airflow's normal webserver uses gunicorn, which forks worker processes. On some
Python builds (notably macOS Apple CommandLineTools Python) those forked workers
segfault before booting, so `airflow webserver` never comes up. This launcher
runs the same Airflow Flask app in a single threaded process instead, which
avoids the fork entirely and serves the identical UI.

Usage (from the project root):
    export AIRFLOW_HOME=$PWD/airflow_home
    export AIRFLOW__CORE__DAGS_FOLDER=$PWD/dags
    export AIRFLOW__CORE__LOAD_EXAMPLES=False
    export PYTHONPATH=$PWD
    python scripts/serve_airflow_ui.py
    # then open http://localhost:8080

Note: this is a development server; use the real gunicorn webserver in
production (where the fork issue does not occur).
"""
import os

os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

from airflow.www.app import create_app  # noqa: E402

app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, threaded=True,
            use_reloader=False, debug=False)
