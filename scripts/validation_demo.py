"""Validation demonstration on deliberately corrupted synthetic data.

MovieLens 100K is already clean, so the production validation report shows zero
quarantined rows. This script generates a small synthetic dataset with injected
errors (bad ratings, nulls, orphan references, impossible timestamps,
duplicates) and runs the SAME validation engine on it, so you can see the
error-quarantine machinery actually catching and separating invalid records.

Run: python scripts/validation_demo.py
Outputs:
    reports/validation_demo_report.md
    reports/validation_demo_errors_sample.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.ingestion.ingest import _generate_synthetic
from src.utils import load_config
from src.validation.validate import (ValidationResult, _validate_dimension,
                                      _validate_interactions)


def main() -> None:
    cfg = load_config()
    # Force a visible corruption rate for the demo.
    cfg["ingestion"]["synthetic"] = {
        "n_users": 300, "n_items": 200, "n_interactions": 8000,
        "corruption_rate": 0.05,
    }
    users, items, interactions = _generate_synthetic(cfg)

    result = ValidationResult()
    users_clean = _validate_dimension(users, "user_id", "users", result)
    items_clean = _validate_dimension(items, "item_id", "items", result)
    interactions_clean = _validate_interactions(
        interactions, users_clean, items_clean, cfg, result
    )

    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)

    # Build the markdown report.
    lines = ["# Validation Demo — Error Quarantine on Corrupted Synthetic Data", "",
             f"- Raw interactions: **{len(interactions)}**",
             f"- Clean interactions retained: **{len(interactions_clean)}**",
             f"- Rows quarantined: **{sum(len(f) for f in result.error_frames.values())}**",
             "", "| Table | Check | Status | Records Failed | Detail |",
             "|-------|-------|--------|---------------:|--------|"]
    for c in result.checks:
        lines.append(f"| {c['table']} | `{c['check']}` | {c['status']} | "
                     f"{c['records_failed']} | {c['detail']} |")
    lines += ["", "## Sample of quarantined rows (by reason)", ""]

    samples = []
    for name, frame in result.error_frames.items():
        samples.append(frame.head(3))
    if samples:
        sample_df = pd.concat(samples, ignore_index=True)
        sample_df.to_csv(reports / "validation_demo_errors_sample.csv", index=False)
        lines.append("See `validation_demo_errors_sample.csv` for examples. Reasons caught:")
        lines.append("")
        for name, frame in result.error_frames.items():
            reason = frame["_error_reason"].iloc[0] if len(frame) else ""
            lines.append(f"- `{name}` — {len(frame)} rows — {reason}")

    (reports / "validation_demo_report.md").write_text("\n".join(lines) + "\n",
                                                       encoding="utf-8")
    print("Wrote reports/validation_demo_report.md")
    print(f"Quarantined {sum(len(f) for f in result.error_frames.values())} rows "
          f"out of {len(interactions)}")


if __name__ == "__main__":
    main()
