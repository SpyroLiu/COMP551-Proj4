import csv
import time
import numpy as np
import pandas as pd

from coreset import coreset
from Booster import linregcoreset

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import GridSearchCV

def get_new_clf(solver, *args, **kwargs):
    if solver == "linear":
        kwargs.pop('alpha')
        return LinearRegression(*args, **kwargs)
    elif solver == "lasso":
        return Lasso(*args, **kwargs)
    elif solver == "ridge":
        return Ridge(*args, **kwargs)
    if solver == "elastic":
        return ElasticNet(*args, **kwargs)

class CoresetLMS(BaseEstimator, RegressorMixin):
    """
    """

    def __init__(self, solver="linear", alpha=1, k=None, size=None, tol=1e-8, dtype=np.float64):
        self.solver = solver
        self.alpha = alpha
        self.k = k
        self.size = size
        self.tol = tol
        self.dtype = dtype

    def fit(self, X, y):
        self.clf_ = get_new_clf(self.solver, alpha=self.alpha)
        C, b, w = coreset(X, y, None, self.k, self.size, self.tol, self.dtype)
        #S = w * C
        #assert np.allclose(X.T @ X, S.T @ S)
        self.clf_.fit(w * C, b)
        return self

    def predict(self, X):
        return self.clf_.predict(X)

def load_datasets():
    ds1 = pd.read_csv(
        "3D_spatial_network.csv",
        header=None,
    ).iloc[:,1:].to_numpy()

    ds2 = pd.read_csv(
        "household_power_consumption.csv",
        sep=';',
        na_values='?'
    ).iloc[:,2:5].dropna().to_numpy()

    ds3 = pd.read_csv(
        "kc_house_data.csv",
        index_col=0
    ).iloc[:, [2, 4, 5, 6, 7, 11, 12, 13, 1]].to_numpy().astype(np.float32)

    return (ds1,ds2,ds3)


def main():
    ds1, ds2, ds3 = load_datasets();
    from sklearn.model_selection import cross_validate

    clf = CoresetLMS()
    lnr = LinearRegression()


    n = 100000000
    d = 3
    data_range = 100
    num_of_alphas = 300
    folds = 3
    data = np.floor(np.random.rand(n, d) * data_range)
    labels = np.floor(np.random.rand(n, 1) * data_range)
    weights = np.ones(n)

    #clf.fit(ds1[:,:-1], ds1[:,-1])
    print(cross_validate(clf, data, labels))
    print(cross_validate(lnr, data, labels))



if __name__ == '__main__':
    main()