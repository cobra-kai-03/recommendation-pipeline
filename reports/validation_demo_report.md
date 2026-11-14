# Validation Demo — Error Quarantine on Corrupted Synthetic Data

- Raw interactions: **6572**
- Clean interactions retained: **6182**
- Rows quarantined: **486**

| Table | Check | Status | Records Failed | Detail |
|-------|-------|--------|---------------:|--------|
| users | `missing_primary_key` | PASS | 0 |  |
| users | `duplicate_primary_key` | PASS | 0 |  |
| items | `missing_primary_key` | PASS | 0 |  |
| items | `duplicate_primary_key` | PASS | 0 |  |
| interactions | `missing_values` | FAIL | 65 |  |
| interactions | `incorrect_data_types` | PASS | 0 |  |
| interactions | `invalid_rating` | FAIL | 130 | allowed [1, 5] |
| interactions | `invalid_timestamp` | FAIL | 65 | allowed [1990-01-01, 2026-12-31] |
| interactions | `duplicate_keys` | FAIL | 96 |  |
| interactions | `orphan_user_ref` | FAIL | 65 |  |
| interactions | `orphan_item_ref` | FAIL | 65 |  |
| interactions | `TOTAL_CLEAN_ROWS` | FAIL | 390 | 6182/6572 rows retained |

## Sample of quarantined rows (by reason)

See `validation_demo_errors_sample.csv` for examples. Reasons caught:

- `missing_values` — 65 rows — missing required value
- `invalid_rating` — 130 rows — rating out of range
- `invalid_timestamp` — 65 rows — timestamp unparseable/out of range
- `duplicates` — 96 rows — duplicate (user_id,item_id)
- `referential_integrity` — 130 rows — unknown user_id or item_id
