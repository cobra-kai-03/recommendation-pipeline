# Data Validation Report

_Generated: 2026-11-14T12:15:43_

## Summary

- Raw interactions: **100000**
- Clean interactions retained: **100000**
- Rows quarantined to `data/errors/`: **0**
- Checks run: **12**, failed: **0**

## Checks

| Table | Check | Status | Records Failed | Detail |
|-------|-------|--------|---------------:|--------|
| users | `missing_primary_key` | PASS | 0 |  |
| users | `duplicate_primary_key` | PASS | 0 |  |
| items | `missing_primary_key` | PASS | 0 |  |
| items | `duplicate_primary_key` | PASS | 0 |  |
| interactions | `missing_values` | PASS | 0 |  |
| interactions | `incorrect_data_types` | PASS | 0 |  |
| interactions | `invalid_rating` | PASS | 0 | allowed [1, 5] |
| interactions | `invalid_timestamp` | PASS | 0 | allowed [1990-01-01, 2026-12-31] |
| interactions | `duplicate_keys` | PASS | 0 |  |
| interactions | `orphan_user_ref` | PASS | 0 |  |
| interactions | `orphan_item_ref` | PASS | 0 |  |
| interactions | `TOTAL_CLEAN_ROWS` | PASS | 0 | 100000/100000 rows retained |
