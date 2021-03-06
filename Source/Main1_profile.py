'''
INSTRUCTION:

$ python3 -m cProfile Main1_profile.py > profile_full.txt &&
    cat profile_full.txt | grep -f <(printf "Main1_\|ncalls") > profile_filtered.txt

Produces;
- profile_full.txt: Containing all output and full profiling
- profile_filtered.txt: Containing only the output of _our_ functions

'''

import DataLoader
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas
import pickle
import sys
import warnings
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

if not sys.warnoptions:
        warnings.simplefilter("ignore")

def main():

    # --- < Set parameters> ---
    seed = 5
    test_size = 0.30
    path = '../Json2/'
    ann_maxiter = 2000
    ann_epsilon = 1e-6
    reverse_datasets = False
    # --- </Set parameters> ---

    # --- < Load all data> ---
    DATA = DataLoader.DataLoader()
    all_Data = [DATA.Data_mias,
                DATA.Cleaned_data,
                DATA.FNA_gb,
                DATA.data_FNA
                # DATA.GSE58606_data
    ]
    all_Data_names = DATA.list_names[:-1]
    if reverse_datasets:
        all_Data, all_Data_names = all_Data[::-1], all_Data_names[::-1]
    # --- </Load all data > ---

    # --- < Prepare classifiers > ---
    models = []
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('SVM', SVC()))
    models.append(('NB', GaussianNB()))
    models.append(('ANN', MLPClassifier(
        max_iter = ann_maxiter, epsilon=ann_epsilon))
    )
    # --- </Prepare classifiers > ---

    # --- < Collect feature selection methods > ---
    fs_methods = []
    fs_methods.append('sfs')
    fs_methods.append('sbs')
    fs_methods.append('chi2')
    fs_methods.append('entropy')
    # --- </Collect feature selection methods > ---

    # --- < Run >
    for dataset, data_name in zip(all_Data, all_Data_names):
        print('* Preparing the ' + data_name + ' data set...\n')
        X, y = dataset
        Xtr, Xtest, Ytr, Ytest = splitData(X, y, test_size, seed)
        # Check distribbution in test data
        # print('X_test M/B ratio: ', np.size(np.where(Xtest==0))/np.size(Xtest)*100, '%')
        for method_name in fs_methods:
            results, fname = evaluateMethod(method_name, models,
                Xtr, Xtest, Ytr, Ytest, data_name
            )
        # dumpJson(results, fname, path)
    # --- </Run >
    print('---*-*-*--- EOF ---*-*-*---')

def splitData(X, y, test_size, seed):
    return train_test_split(X, y, test_size=test_size, random_state=seed)

def chi2Function(x, X, Y, model):
    Xtr_subset = SelectKBest(chi2, k=x).fit_transform(X, Y)
    return (
        Xtr_subset,
        SelectKBest(chi2, k=x).fit(X, Y).get_support(),
        model.fit(Xtr_subset, Y)
    )

def entropyFunction(x, X, Y, model):
    Xtr_subset = SelectKBest(mutual_info_classif, k=x).fit_transform(X, Y)
    return (
        Xtr_subset,
        SelectKBest(mutual_info_classif, k=x).fit(X, Y).get_support(),
        model.fit(Xtr_subset, Y)
    )

def sfsFunction(x, X, Y, model, Xtr, Ytr):
    sXs = SFS(model, k_features=x, forward=True,
            floating=False, n_jobs = -1)
    return (
        sXs.fit_transform(Xtr, Ytr),
        sXs
    )

def sbsFunction(x, X, Y, model, Xtr, Ytr):
    sXs = SFS(model, k_features=x, forward=False,
            floating=False, n_jobs = -1)
    return (
        sXs.fit_transform(Xtr, Ytr),
        sXs
    )

def subsetFrom(mask, Xtest):
    mask = np.nonzero(mask)
    return Xtest[:,mask[0]]

def evaluateMethod(f_name, models, Xtr, Xtest, Ytr, Ytest, data_name):
    print('* * Evaluating ' + f_name + '...\n')
    results = {'CART':[], 'SVM':[], 'NB':[], 'ANN':[]}
    features = Xtr.shape[1]
    step_sz = 1 if features < 100 else 100

    for model_name, model in models:
        print('\n* * * Evaluating ' + model_name + '...\n')
        for num_features in range(1, features, step_sz):
            # --- Filter methods ---
            if f_name == 'chi2' or f_name == 'entropy':
                if f_name == 'chi2':
                    Xtr_subset, X_mask, model = chi2Function(num_features, Xtr, Ytr, model)
                else:
                    Xtr_subset, X_mask, model = entropyFunction(num_features, Xtr, Ytr, model)
                Xtest_subset = subsetFrom(X_mask, Xtest)
            # --- Wrapper methods ---
            if f_name == 'sbs' or f_name == 'sfs':
                if f_name == 'sbs':
                    Xtr_subset, sXs = sbsFunction(num_features, Xtr, Ytr, model, Xtr, Ytr)
                else:
                    Xtr_subset, sXs = sfsFunction(num_features, Xtr, Ytr, model, Xtr, Ytr)
                Xtest_subset = subsetFrom(sXs.k_feature_idx_, Xtest)
                parameters = sXs.get_params(False)
                model.set_params(**parameters['estimator'].get_params())
            # --- Evaluate method ---
            kfold = model_selection.StratifiedKFold(n_splits=10)
            cv_results = model_selection.cross_val_score(
                model, Xtest_subset, Ytest, cv=kfold, scoring='accuracy'
            )
            model_score_mean = cv_results.mean()
            model_score_std = cv_results.std()
            results[model_name].append((
                num_features, cv_results.tolist(),
                model_score_mean, model_score_std
            ))
            msg = "Model: %s | #F: %i | Mean: %f | Std: %f " % (
                model_name, num_features, model_score_mean, model_score_std
            )
            print(msg)
    filename = data_name + '_' + f_name + '.json'
    return results, filename

def dumpJson(results, fname, path):
    with open(path+fname, 'w') as fp:
        json.dump(results, fp)

if __name__ == '__main__':
    main()
