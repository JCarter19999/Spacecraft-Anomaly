import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

class PCABaseline:
    def __init__(self, n_components=20):
        self.pca = PCA(n_components=n_components, whiten=False)

    def fit(self, X: pd.DataFrame):
        self.pca.fit(X.values)
        return self

    def score(self, X: pd.DataFrame):
        Z = self.pca.transform(X.values)
        Xh = self.pca.inverse_transform(Z)
        err = ((X.values - Xh)**2).sum(axis=1) ** 0.5
        return err
