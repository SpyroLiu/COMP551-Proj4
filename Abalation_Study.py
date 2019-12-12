import csv
import time
import numpy as np
import Booster

def Test_result(data,labels,weights,solver,folds,alphas):
    clf = Booster.get_new_clf(solver, folds=folds, alphas=alphas)
    time_coreset, clf_coreset = Booster.coreset_train_model(data, labels, weights, clf, folds=folds, solver=solver)
    score_coreset = Booster.test_model(data, labels, weights, clf)

    clf = Booster.get_new_clf(solver, folds=folds, alphas=alphas)
    time_real, clf_real = Booster.train_model(data, labels, weights,clf)
    score_real = Booster.test_model(data, labels, weights, clf)

    print(" solver: {}\n number_of_alphas: {}, \nscore_diff = {}\n---->coef diff = {}\n---->coreset_time = {}\n---->data time = {}".format
        (solver,alphas,np.abs(score_coreset - score_real),np.sum(np.abs(clf_real.coef_ - clf_coreset.coef_)),time_coreset,time_real))

def ThreeExperiement_DataLoading():
    with open("3D_spatial_network.csv", 'r') as f:
        dataset1 = list(csv.reader(f, delimiter=","))
        ds1 = np.array(dataset1)
        ds1 = ds1[0:, 1:]
        ds1 = ds1.astype(np.float).astype(np.int)

    with open("household_power_consumption.csv", 'r') as f:
        dataset2 = list(csv.reader(f, delimiter=";"))
        ds2 = np.array(dataset2)
        ds2 = ds2[np.logical_not(ds2[:, 2] == '?')]
        ds2 = ds2[1:, 2:5]
        ds2 = ds2.astype(np.float).astype(np.int)

    with open("kc_house_data.csv", 'r') as f:
        dataset3 = list(csv.reader(f, delimiter=","))
        ds3 = np.array(dataset3)
        ds3 = ds3[1:, [3, 5, 6, 7, 8, 12, 13, 14, 2]]
        ds3 = ds3.astype(np.float).astype(np.int)

    return [ds1,ds2,ds3]

def test_experiment(test_data,folds=3,alphas=100):

    selected_solver = ["lasso", "ridge", "elastic"]
    reproduce_dataset = test_data
    count = 0

    for ds in reproduce_dataset:
        count +=1
        print('\n Test result of dataset{}'.format(count))
        for solver in selected_solver:
            data = ds[:, :-1]
            labels = ds[:, -1].reshape(ds.shape[0], 1)
            weights = np.ones(ds.shape[0])

            Test_result(data,labels,weights,solver,folds,alphas)

def test_synthetic_features(data_size = 10000, feature_size =[3,5,7], folds=3,alphas=100,range=100,solver='lasso'):
    n = data_size
    d = feature_size
    weights = np.ones(n)
    labels = np.floor(np.random.rand(n, 1) * range)

    for i in feature_size:
        print('\n Test result of d={}'.format(i))
        data = np.floor(np.random.rand(n, i) * range)
        Test_result(data,labels,weights,solver,folds,alphas)


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

