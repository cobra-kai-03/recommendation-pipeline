"""Personalized model — Matrix Factorization (SVD with biases, SGD).

Learns latent user/item factors plus global/user/item biases by stochastic
gradient descent over the observed ratings:

    r_hat(u, i) = mu + b_u + b_i + p_u . q_i

Because it predicts explicit ratings it supports both:
  * top-K ranking  -> Precision/Recall/NDCG/Coverage/Diversity
  * rating prediction -> RMSE / MAE

This is implemented in NumPy (no heavyweight native deps) so it installs and
runs reliably across platforms.
"""
from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


class MatrixFactorization:
    name = "matrix_factorization"

    def __init__(self, n_factors: int = 30, n_epochs: int = 25, lr: float = 0.01,
                 reg: float = 0.05, seed: int = 42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.seed = seed

    # -- fitting -----------------------------------------------------------
    def fit(self, train: pd.DataFrame) -> "MatrixFactorization":
        rng = np.random.default_rng(self.seed)
        users = train["user_id"].unique()
        items = train["item_id"].unique()
        self.u_index = {u: idx for idx, u in enumerate(users)}
        self.i_index = {i: idx for idx, i in enumerate(items)}
        self.items_ = items

        n_users, n_items = len(users), len(items)
        self.mu = float(train["rating"].mean())
        self.b_u = np.zeros(n_users)
        self.b_i = np.zeros(n_items)
        self.P = rng.normal(0, 0.1, (n_users, self.n_factors))
        self.Q = rng.normal(0, 0.1, (n_items, self.n_factors))

        u_arr = train["user_id"].map(self.u_index).to_numpy()
        i_arr = train["item_id"].map(self.i_index).to_numpy()
        r_arr = train["rating"].to_numpy(dtype=float)
        n = len(r_arr)

        for epoch in range(self.n_epochs):
            order = rng.permutation(n)
            sq_err = 0.0
            for idx in order:
                u, i, r = u_arr[idx], i_arr[idx], r_arr[idx]
                pred = self.mu + self.b_u[u] + self.b_i[i] + self.P[u] @ self.Q[i]
                err = r - pred
                sq_err += err * err
                self.b_u[u] += self.lr * (err - self.reg * self.b_u[u])
                self.b_i[i] += self.lr * (err - self.reg * self.b_i[i])
                p_u = self.P[u].copy()
                self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (err * p_u - self.reg * self.Q[i])
            self.train_rmse_ = float(np.sqrt(sq_err / n))
        return self

    # -- rating prediction -------------------------------------------------
    def predict(self, user_id: int, item_id: int) -> float:
        u = self.u_index.get(user_id)
        i = self.i_index.get(item_id)
        est = self.mu
        if u is not None:
            est += self.b_u[u]
        if i is not None:
            est += self.b_i[i]
        if u is not None and i is not None:
            est += self.P[u] @ self.Q[i]
        return float(np.clip(est, 1.0, 5.0))

    def predict_batch(self, pairs: pd.DataFrame) -> np.ndarray:
        return np.array([self.predict(u, i)
                         for u, i in zip(pairs["user_id"], pairs["item_id"])])

    # -- ranking -----------------------------------------------------------
    def recommend(self, user_id: int, k: int = 10,
                  exclude: Optional[Iterable[int]] = None) -> List[int]:
        u = self.u_index.get(user_id)
        if u is None:
            return []  # cold-start user: no personalized signal
        # Note: wrap the matmul in errstate — NumPy's matmul can raise spurious
        # divide/overflow/invalid warnings from stale CPU FP flags even though
        # the operands (verified finite, O(1) magnitude) produce a valid result.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            scores = self.mu + self.b_u[u] + self.b_i + self.Q @ self.P[u]
        exclude = set(exclude or [])
        order = np.argsort(-scores)
        recs: List[int] = []
        for idx in order:
            item = self.items_[idx]
            if item in exclude:
                continue
            recs.append(int(item))
            if len(recs) >= k:
                break
        return recs

    @property
    def params(self) -> dict:
        return {
            "model": self.name,
            "n_factors": self.n_factors,
            "n_epochs": self.n_epochs,
            "lr": self.lr,
            "reg": self.reg,
        }
