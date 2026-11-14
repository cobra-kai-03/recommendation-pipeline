"""Stage 8 — Orchestration (pure-Python).

A dependency-ordered, reproducible runner for the full pipeline. Each stage is a
task with explicit upstream dependencies; the orchestrator executes them in
topological order, times each one, stops on the first failure, and writes a
run summary to ``reports/pipeline_run.json`` + ``reports/pipeline_run.md``.

Usage
-----
    python pipeline.py                 # run the whole pipeline
    python pipeline.py --stage features  # run one stage (and nothing else)
    python pipeline.py --from transformation  # run from a stage to the end
    python pipeline.py --list          # list stages and dependencies
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from datetime import datetime
from typing import Callable, Dict, List

from src.utils import ensure_dir, get_logger, load_config, resolve_path

logger = get_logger("orchestrator")


# Each task: (name, callable, [upstream dependencies])
def _build_tasks() -> List[Dict]:
    from src.evaluation.evaluate import run as evaluate_run
    from src.features.build_features import run as features_run
    from src.ingestion.ingest import run as ingest_run
    from src.preparation.prepare import run as prepare_run
    from src.training.train import run as train_run
    from src.transformation.transform import run as transform_run
    from src.validation.validate import run as validate_run

    return [
        {"name": "ingestion", "fn": ingest_run, "deps": []},
        {"name": "validation", "fn": validate_run, "deps": ["ingestion"]},
        {"name": "preparation", "fn": prepare_run, "deps": ["validation"]},
        {"name": "transformation", "fn": transform_run, "deps": ["preparation"]},
        {"name": "features", "fn": features_run, "deps": ["transformation"]},
        {"name": "training", "fn": train_run, "deps": ["features"]},
        {"name": "evaluation", "fn": evaluate_run, "deps": ["training"]},
    ]


def _topological_order(tasks: List[Dict]) -> List[str]:
    order, visited = [], set()
    by_name = {t["name"]: t for t in tasks}

    def visit(name: str, stack: set) -> None:
        if name in visited:
            return
        if name in stack:
            raise ValueError(f"Cyclic dependency detected at {name}")
        stack.add(name)
        for dep in by_name[name]["deps"]:
            visit(dep, stack)
        stack.discard(name)
        visited.add(name)
        order.append(name)

    for t in tasks:
        visit(t["name"], set())
    return order


def run_pipeline(config_path: str | None = None, only: str | None = None,
                 start_from: str | None = None) -> Dict:
    cfg = load_config(config_path)
    reports_dir = ensure_dir(resolve_path(cfg["paths"]["reports"]))
    tasks = _build_tasks()
    by_name = {t["name"]: t for t in tasks}
    order = _topological_order(tasks)

    if only:
        to_run = [only]
    elif start_from:
        idx = order.index(start_from)
        to_run = order[idx:]
    else:
        to_run = order

    logger.info("Pipeline plan: %s", " -> ".join(to_run))
    results, started = [], datetime.now()

    for name in to_run:
        t0 = time.time()
        logger.info("")
        logger.info("▶ STAGE START: %s", name)
        try:
            output = by_name[name]["fn"](config_path)
            elapsed = time.time() - t0
            results.append({"stage": name, "status": "SUCCESS",
                            "seconds": round(elapsed, 2)})
            logger.info("✔ STAGE OK: %s (%.2fs)", name, elapsed)
        except Exception as exc:  # noqa: BLE001
            elapsed = time.time() - t0
            results.append({"stage": name, "status": "FAILED",
                            "seconds": round(elapsed, 2), "error": str(exc)})
            logger.error("✖ STAGE FAILED: %s -> %s", name, exc)
            logger.error(traceback.format_exc())
            _write_summary(cfg, reports_dir, results, started, success=False)
            raise

    _write_summary(cfg, reports_dir, results, started, success=True)
    logger.info("")
    logger.info("PIPELINE COMPLETE ✅  (%d stages)", len(results))
    return {"results": results}


def _write_summary(cfg, reports_dir, results, started, success: bool) -> None:
    total = sum(r["seconds"] for r in results)
    summary = {
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "overall_status": "SUCCESS" if success else "FAILED",
        "total_seconds": round(total, 2),
        "stages": results,
    }
    (reports_dir / "pipeline_run.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    lines = ["# Pipeline Run Summary", "",
             f"- Status: **{summary['overall_status']}**",
             f"- Started: {summary['started_at']}",
             f"- Finished: {summary['finished_at']}",
             f"- Total time: **{summary['total_seconds']}s**", "",
             "| Stage | Status | Seconds |", "|-------|--------|--------:|"]
    for r in results:
        lines.append(f"| {r['stage']} | {r['status']} | {r['seconds']} |")
    (reports_dir / "pipeline_run.md").write_text("\n".join(lines) + "\n",
                                                 encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommendation pipeline orchestrator")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--stage", default=None, help="Run only this stage")
    parser.add_argument("--from", dest="start_from", default=None,
                        help="Run from this stage to the end")
    parser.add_argument("--list", action="store_true", help="List stages & exit")
    args = parser.parse_args()

    if args.list:
        for t in _build_tasks():
            deps = ", ".join(t["deps"]) or "—"
            print(f"{t['name']:16s} depends on: {deps}")
        return

    run_pipeline(args.config, only=args.stage, start_from=args.start_from)


if __name__ == "__main__":
    main()
