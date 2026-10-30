"""Exact Shapley values by coalition enumeration (feasible for <=15 features).

All coalitions x background rows are stacked into ONE prediction call, which is
what makes exact enumeration practical here.
"""
import numpy as np, pandas as pd
from itertools import combinations
from math import comb

def _coalitions(n):
    for r in range(n + 1):
        for S in combinations(range(n), r):
            yield S

def exact_shapley(predict, X_bg, x, features):
    """predict: f(DataFrame)->1d array; X_bg: background DataFrame; x: 1-row DataFrame."""
    n, B = len(features), len(X_bg)
    cols = list(X_bg.columns)
    Ss = list(_coalitions(n))
    blocks = []
    for S in Ss:
        Xs = X_bg.copy()
        for j in S:
            Xs[features[j]] = x[features[j]].iloc[0]
        blocks.append(Xs)
    big = pd.concat(blocks, ignore_index=True)[cols]
    p = predict(big).reshape(len(Ss), B).mean(axis=1)
    v = {S: p[i] for i, S in enumerate(Ss)}

    phi = np.zeros(n)
    for j in range(n):
        rest = [k for k in range(n) if k != j]
        for r in range(n):
            w = 1.0 / (n * comb(n - 1, r))
            for S in combinations(rest, r):
                phi[j] += w * (v[tuple(sorted(S + (j,)))] - v[tuple(sorted(S))])
    return phi, v[()], v[tuple(range(n))]
