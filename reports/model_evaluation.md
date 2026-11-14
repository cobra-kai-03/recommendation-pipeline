# Model Evaluation Results (K=10)

Ranking metrics are averaged over users with at least one relevant (rating ≥ threshold) item in the test set. RMSE/MAE are reported only for models that predict explicit ratings.

| model | precision_at_10 | recall_at_10 | ndcg_at_10 | coverage | diversity | rmse | mae | n_users_evaluated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| popularity | 0.0579 | 0.0540 | 0.0669 | 0.0467 | 0.7448 | — | — | 906 |
| matrix_factorization | 0.0446 | 0.0420 | 0.0518 | 0.3062 | 0.7071 | 0.9761 | 0.7700 | 906 |

## Summary

- **Best ranking (NDCG@10)**: `popularity`.
- **Best catalog coverage**: `matrix_factorization`.
- **Rating prediction (RMSE/MAE)**: only `matrix_factorization` predicts explicit ratings; the popularity baseline does not.

On a per-user temporal (leave-last-out) split, test relevants skew toward popular items, so the popularity baseline is a strong competitor on raw top-K precision/recall. Matrix factorization is the only model that predicts explicit ratings (RMSE/MAE) and substantially improves catalog coverage — i.e. it personalizes beyond the head of the catalog.
