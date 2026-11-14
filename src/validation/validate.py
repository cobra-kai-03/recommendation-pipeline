"""Stage 2 — Data Validation.

Runs automated quality checks on the three raw tables:

  * missing values
  * duplicate rows / duplicate keys
  * incorrect data types (coercion failures)
  * invalid ratings (outside the allowed range)
  * invalid timestamps (unparseable or outside a sane range)
  * referential integrity (interactions referencing unknown users/items)

Invalid interaction records are **quarantined** into ``data/errors`` (one CSV
per rule, plus a combined file) instead of silently dropped, and a structured
validation report is written to both ``data/errors`` and ``reports``.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.utils import ensure_dir, get_logger, load_config, resolve_path

logger = get_logger("validation")


class ValidationResult:
    """Accumulates per-check outcomes and the quarantined error rows."""

    def __init__(self) -> None:
        self.checks: List[Dict] = []
        self.error_frames: Dict[str, pd.DataFrame] = {}

    def add(self, table: str, check: str, n_failed: int, detail: str = "") -> None:
        status = "PASS" if n_failed == 0 else "FAIL"
        self.checks.append({
            "table": table,
            "check": check,
            "status": status,
            "records_failed": int(n_failed),
            "detail": detail,
        })
        logger.info("[%s] %-28s %-4s (%d failed) %s",
                    table, check, status, n_failed, detail)

    def quarantine(self, name: str, df: pd.DataFrame, reason: str) -> None:
        if df.empty:
            return
        df = df.copy()
        df["_error_reason"] = reason
        self.error_frames[name] = df


def _validate_interactions(df: pd.DataFrame, users: pd.DataFrame,
                           items: pd.DataFrame, cfg: Dict,
                           result: ValidationResult) -> pd.DataFrame:
    """Validate the interactions table; return the clean subset."""
    vcfg = cfg["validation"]
    n0 = len(df)
    bad_mask = pd.Series(False, index=df.index)

    # --- Missing values in required columns --------------------------------
    required = ["user_id", "item_id", "rating", "timestamp"]
    missing_mask = df[required].isna().any(axis=1)
    result.add("interactions", "missing_values", missing_mask.sum())
    result.quarantine("missing_values", df[missing_mask], "missing required value")
    bad_mask |= missing_mask

    # --- Type coercion (ids & rating numeric, timestamp parseable) ---------
    rating_num = pd.to_numeric(df["rating"], errors="coerce")
    uid_num = pd.to_numeric(df["user_id"], errors="coerce")
    iid_num = pd.to_numeric(df["item_id"], errors="coerce")
    type_mask = (rating_num.isna() & df["rating"].notna()) | uid_num.isna() | iid_num.isna()
    result.add("interactions", "incorrect_data_types", type_mask.sum())
    result.quarantine("type_errors", df[type_mask], "non-numeric id/rating")
    bad_mask |= type_mask

    # --- Invalid ratings (out of [min, max]) -------------------------------
    rmin, rmax = vcfg["rating_min"], vcfg["rating_max"]
    rating_mask = rating_num.notna() & ((rating_num < rmin) | (rating_num > rmax))
    result.add("interactions", "invalid_rating",
               rating_mask.sum(), f"allowed [{rmin}, {rmax}]")
    result.quarantine("invalid_rating", df[rating_mask], "rating out of range")
    bad_mask |= rating_mask

    # --- Invalid timestamps -------------------------------------------------
    ts = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"),
                        unit="s", errors="coerce")
    tmin = pd.Timestamp(vcfg["timestamp_min"])
    tmax = pd.Timestamp(vcfg["timestamp_max"])
    ts_mask = ts.isna() | (ts < tmin) | (ts > tmax)
    result.add("interactions", "invalid_timestamp",
               ts_mask.sum(), f"allowed [{tmin.date()}, {tmax.date()}]")
    result.quarantine("invalid_timestamp", df[ts_mask], "timestamp unparseable/out of range")
    bad_mask |= ts_mask

    # --- Duplicate (user, item) keys ---------------------------------------
    dup_mask = df.duplicated(subset=["user_id", "item_id"], keep="first")
    result.add("interactions", "duplicate_keys", dup_mask.sum())
    result.quarantine("duplicates", df[dup_mask], "duplicate (user_id,item_id)")
    bad_mask |= dup_mask

    # --- Referential integrity ---------------------------------------------
    valid_users = set(pd.to_numeric(users["user_id"], errors="coerce").dropna())
    valid_items = set(pd.to_numeric(items["item_id"], errors="coerce").dropna())
    ref_user_mask = ~uid_num.isin(valid_users)
    ref_item_mask = ~iid_num.isin(valid_items)
    ref_mask = ref_user_mask | ref_item_mask
    result.add("interactions", "orphan_user_ref", ref_user_mask.sum())
    result.add("interactions", "orphan_item_ref", ref_item_mask.sum())
    result.quarantine("referential_integrity", df[ref_mask],
                      "unknown user_id or item_id")
    bad_mask |= ref_mask

    clean = df[~bad_mask].copy()
    result.add("interactions", "TOTAL_CLEAN_ROWS", n0 - len(clean),
               f"{len(clean)}/{n0} rows retained")
    return clean


def _validate_dimension(df: pd.DataFrame, key: str, table: str,
                        result: ValidationResult) -> pd.DataFrame:
    """Validate a dimension table (users/items): missing key + duplicate key."""
    missing = df[key].isna()
    result.add(table, "missing_primary_key", missing.sum())
    dup = df.duplicated(subset=[key], keep="first")
    result.add(table, "duplicate_primary_key", dup.sum())
    clean = df[~(missing | dup)].copy()
    return clean


def run(config_path: str | None = None) -> Dict:
    cfg = load_config(config_path)
    raw = resolve_path(cfg["paths"]["raw"])
    err_dir = ensure_dir(resolve_path(cfg["paths"]["errors"]))
    reports_dir = ensure_dir(resolve_path(cfg["paths"]["reports"]))

    users = pd.read_csv(raw / "users.csv")
    items = pd.read_csv(raw / "items.csv")
    interactions = pd.read_csv(raw / "interactions.csv")

    result = ValidationResult()
    users_clean = _validate_dimension(users, "user_id", "users", result)
    items_clean = _validate_dimension(items, "item_id", "items", result)
    interactions_clean = _validate_interactions(
        interactions, users_clean, items_clean, cfg, result
    )

    # Persist quarantined error rows (one file per rule + combined).
    combined = []
    for name, frame in result.error_frames.items():
        path = err_dir / f"errors_{name}.csv"
        frame.to_csv(path, index=False)
        combined.append(frame)
        logger.info("Quarantined %d rows -> %s", len(frame), path.name)
    if combined:
        pd.concat(combined, ignore_index=True).to_csv(
            err_dir / "errors_all.csv", index=False
        )

    # Persist the clean interactions for downstream stages.
    interactions_clean.to_csv(raw / "interactions_valid.csv", index=False)
    users_clean.to_csv(raw / "users_valid.csv", index=False)
    items_clean.to_csv(raw / "items_valid.csv", index=False)

    total_failed = sum(c["records_failed"] for c in result.checks
                       if c["check"] not in ("TOTAL_CLEAN_ROWS",))
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "raw_interactions": int(len(interactions)),
            "clean_interactions": int(len(interactions_clean)),
            "quarantined_rows": int(sum(len(f) for f in result.error_frames.values())),
            "total_check_failures": int(total_failed),
            "checks_run": len(result.checks),
            "checks_failed": sum(1 for c in result.checks if c["status"] == "FAIL"),
        },
        "checks": result.checks,
    }

    with open(err_dir / "validation_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    _write_markdown_report(report, reports_dir / "validation_report.md")

    logger.info("Validation complete: %d clean interactions, %d quarantined.",
                len(interactions_clean), report["summary"]["quarantined_rows"])
    return report


def _write_markdown_report(report: Dict, path: Path) -> None:
    lines = ["# Data Validation Report", "",
             f"_Generated: {report['generated_at']}_", ""]
    s = report["summary"]
    lines += [
        "## Summary", "",
        f"- Raw interactions: **{s['raw_interactions']}**",
        f"- Clean interactions retained: **{s['clean_interactions']}**",
        f"- Rows quarantined to `data/errors/`: **{s['quarantined_rows']}**",
        f"- Checks run: **{s['checks_run']}**, failed: **{s['checks_failed']}**",
        "", "## Checks", "",
        "| Table | Check | Status | Records Failed | Detail |",
        "|-------|-------|--------|---------------:|--------|",
    ]
    for c in report["checks"]:
        lines.append(
            f"| {c['table']} | `{c['check']}` | {c['status']} | "
            f"{c['records_failed']} | {c['detail']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
