import numpy as np
import pandas as pd

class ZScoreBaseline:
    def __init__(self, agg="max", eps=1e-8):
        self.mu = None
        self.sd = None
        self.agg = agg
        self.eps = eps

    def fit(self, X: pd.DataFrame):
        self.mu = X.mean()
        self.sd = X.std().replace(0, 1.0)
        return self

    def score(self, X: pd.DataFrame):
        z = (X - self.mu) / (self.sd + self.eps)
        z = z.abs()
        if self.agg == "max":
            return z.max(axis=1).values
        elif self.agg == "mean":
            return z.mean(axis=1).values
        else:
            return z.quantile(0.9, axis=1).values
