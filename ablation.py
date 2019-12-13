import csv
import time
import numpy as np
import pandas as pd
import Booster as bst

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import GridSearchCV

def get_new_clf(solver):
    if solver == "linear":
        return LinearRegression(fit_intercept=False)
    elif solver == "lasso":
        return Lasso(fit_intercept=False)
    elif solver == "ridge":
        return Ridge(fit_intercept=False)
    if solver == "elastic":
        return ElasticNet(fit_intercept=False)

class CoresetLMS(BaseEstimator, RegressorMixin):
    """
    """

    def __init__(self, solver="linear", k=None, c_size=None, dtype='float64'):
        self.clf = get_new_clf(solver)
        self.k = k
        self.c_size

    def fit(self, X, y=None, weights=None):
        if not weights:
            weights = np.ones((X.shape[0],))
        C, u, b = bst.linregcoreset(X,weights,y,c_size,dtype)
        self.clf.fit()
        return self


def test(data,labels,weights,solver,folds,alphas):
    clf = bst.get_new_clf(solver, folds=folds, alphas=alphas)
    time_coreset, clf_coreset = bst.coreset_train_model(data, labels, weights, clf, folds=folds, solver=solver)
    score_coreset = bst.test_model(data, labels, weights, clf)

    clf = bst.get_new_clf(solver, folds=folds, alphas=alphas)
    time_real, clf_real = bst.train_model(data, labels, weights,clf)
    score_real = bst.test_model(data, labels, weights, clf)

    print(
        f'solver: {solver}\n'
        f'alphas: {alphas},\n'
        f'original_time = {time_real}\n'
        f'coreset_time = {time_coreset}\n'
        f'score_diff = {np.abs(score_coreset - score_real)}\n'
        f'coef_diff = {np.sum(np.abs(clf_real.coef_ - clf_coreset.coef_))}\n'
        )

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
    ).iloc[:, [2, 4, 5, 6, 7, 11, 12, 13, 1]].to_numpy().astype(np.int)

    return (ds1,ds2,ds3)

def test_real(datasets,folds=3,alphas=100):

    selected_solver = ["lasso", "ridge", "elastic"]
    count = 0

    for i, ds in enumerate(datasets):
        print(f'Test result of dataset{i+1}\n')
        for solver in selected_solver:
            X = ds[:, :-1]
            y = ds[:, -1].reshape(ds.shape[0], 1)
            w = np.ones(ds.shape[0])

            test(X,y,w,solver,folds,alphas)

def test_synthetic(data_size = 10000, feature_size =[3,5,7], folds=3,alphas=100,range=100,solver='lasso'):
    n = data_size
    d = feature_size
    weights = np.ones(n)
    labels = np.floor(np.random.rand(n, 1) * range)

    for i in feature_size:
        print(f'\nTest result of d={i}')
        data = np.floor(np.random.rand(n, i) * range)
        test(data,labels,weights,solver,folds,alphas)


        # #########RIDGE REGRESSION#############
        # clf = Booster.get_new_clf(solver, folds=folds, alphas=num_of_alphas)
        # time_coreset, clf_coreset = Booster.coreset_train_model(data, labels, weights, clf, folds=folds, solver=solver)
        # score_coreset = Booster.test_model(data, labels, weights, clf)
        #
        # clf = Booster.get_new_clf(solver, folds=folds, alphas=num_of_alphas)
        # time_real, clf_real = Booster.train_model(data, labels, weights, clf)
        # score_real = Booster.test_model(data, labels, weights, clf)
        #
        # print(
        #     " solver: {}\n number_of_alphas: {}, \nscore_diff = {}\n---->coef diff = {}\n---->coreset_time = {}\n---->data time = {}".format(
        #         solver,
        #         num_of_alphas,
        #         np.abs(score_coreset - score_real),
        #         np.sum(np.abs(clf_real.coef_ - clf_coreset.coef_)),
        #         time_coreset,
        #         time_real))

# def test_synthetic_size(num_of_data=10000, num_of_feature=3,sol='lasso',folds=3,alphas=100,range=100):

if __name__ == '__main__':
    datasets = load_datasets();
    test_real(datasets, alphas=10)