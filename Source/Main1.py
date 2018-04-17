import pandas
import numpy as np
import DataLoader
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import pprint as pp
import sys
import json

# TODO: Integrate this function into new structure
def evaluate_sbs(fs_method):
    print("Evaluating", fs_method)
    # evaluate each model in turn
    rez = {'CART':[], 'SVM':[], 'NB':[], 'ANN':[]}
    results = []
    names = []
    scoring = 'accuracy'
    features = list(range(1, X.shape[1]))
    # features = list(range(1, 5))
    for num_features in features:
        for name, model in models:
            sfs1 = SFS(model,
                   k_features=num_features,
                   forward=True,
                   floating=False,
                   scoring=scoring,
                   cv=10,
                   n_jobs = 1)
            X_new = sfs1.fit_transform(X, Y)
            kfold = model_selection.KFold(n_splits=10, random_state=seed)
            cv_results = model_selection.cross_val_score(model, X_new, Y, cv=kfold, scoring=scoring)
            results.append(cv_results)
            rez[name].append((num_features, cv_results.mean()))
            names.append(name)
            msg = "#features: %f, Model: %s:  %f (%f)" % (num_features, name, cv_results.mean(), cv_results.std())
            print(msg)
    # features = list(range(5, X.shape[1]))
    # for num_features in features:
    #     for name, model in models:
    #         kfold = model_selection.KFold(n_splits=10, random_state=seed)
    #         sbs = SFS(  model, k_features=num_features, forward=True, floating=False,
    #                     scoring='accuracy', cv=10, n_jobs=-1)
    #         sbs = sbs.fit(X, Y)
    #         rez[name].append((num_features, sbs.k_score_))
    #         msg = "#features: %f, Model: %s:  %f, feature_idx: %s " % (num_features, name, sbs.k_score_, sbs.k_feature_idx_)
    #         print(msg)
    plotFilter(rez, fs_method)


def main():
    # --- < Set parameters> ---
    seed = 5
    test_size = 0.25
    path = '../Json/'
    # --- </Set parameters> ---

    # --- < Load all data> ---
    DATA = DataLoader.DataLoader()
    all_Data = [DATA.FNA_gb,
                DATA.Cleaned_data,
                DATA.Data_mias,
                DATA.data_FNA,
                DATA.GSE58606_data
    ]
    all_Data_names = [  DATA.FNA_gb_name,
                        DATA.Cleaned_data_name,
                        DATA.Data_mias_name,
                        DATA.data_FNA_name,
                        DATA.GSE58606_data_name
    ]
    # all_Data_names = DATA.list_names
    # --- </Load all data > ---

    # --- < Prepare classifiers > ---
    models = []
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('SVM', SVC()))
    models.append(('NB', GaussianNB()))
    models.append(('ANN', MLPClassifier()))
    # --- </Prepare classifiers > ---

    # --- < Collect feature selection methods > ---
    fs_methods = []
    fs_methods.append((chi2Function(), 'chi2'))
    fs_methods.append((entropyFunction(), 'entropy'))
    # TODO: Add SBS and SFS here!
    # --- </Collect feature selection methods > ---

    # --- < Run >
    for dataset, data_name in zip(all_Data, all_Data_names):
        print('Preparing the ' + data_name + ' data set...')
        X, y = dataset
        Xtr, Xtest, Ytr, Ytest = splitData(X, y, test_size, seed)
        # Check distribbution in test data
        # print('X_test M/B ratio: ', np.size(np.where(Xtest==0))/np.size(Xtest)*100, '%')
        for method, method_name in fs_methods:
            results, fname = evaluateMethod(method, method_name, models,
                Xtr, Xtest, Ytr, Ytest, data_name
            )
        dumpJson(results, fname, path)
    # --- </Run >

def splitData(X, y, test_size, seed):
    return train_test_split(X, y, test_size=test_size, random_state=seed)

def chi2Function():
    f = lambda x, X, Y: (
        SelectKBest(chi2, k=x).fit_transform(X, Y),
        SelectKBest(chi2, k=x).fit(X, Y).get_support()
    )
    return f

def entropyFunction():
    f = lambda x, X, Y: (
        SelectKBest(mutual_info_classif, k=x).fit_transform(X, Y),
        SelectKBest(mutual_info_classif, k=x).fit(X, Y).get_support()
    )
    return f

def evaluateMethod(f, f_name, models, Xtr, Xtest, Ytr, Ytest, data_name):
    print('Evaluating ' + f_name + '...')
    results = {'CART':[], 'SVM':[], 'NB':[], 'ANN':[]}
    scoring = 'accuracy'
    for num_features in range(1, Xtr.shape[1]):
        Xtr_subset, X_mask = f(num_features, Xtr, Ytr)
        X_mask = np.nonzero(X_mask)
        Xtest_subset = Xtest[:,X_mask[0]]
        for name, model in models:
            model.fit(Xtr_subset, Ytr)
            kfold = model_selection.KFold(n_splits=10)
            cv_results = model_selection.cross_val_score(
                model, Xtest_subset, Ytest, cv=kfold, scoring=scoring
            )
            model_score = cv_results.mean()
            model_error = cv_results.std()
            results[name].append((num_features, model_score))
            msg = "#features: %i, Model: %s:  %f (%f)" % (
                num_features, name, model_score, model_error
            )
            print(msg)
    filename = data_name + '_' + f_name + '.json'
    return results, filename

def dumpJson(results, fname, path):
    with open(path+fname, 'w') as fp:
        json.dump(results, fp)

if __name__ == '__main__':
    main()
