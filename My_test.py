import Abalation_Study

data = Abalation_Study.ThreeExperiement_DataLoading()
"""
We choose alpha to test and validate the result from paper
"""
Abalation_Study.test_experiment(data, alphas=20)

"""
We choose different feature size to test and validate the result from paper
We test especially when feature sizes goes large here. Something interesting happen.
"""
Abalation_Study.test_synthetic_features(data_size = 10000, feature_size =[9,10,11],folds=3, alphas=5, range=100,solver='lasso')
