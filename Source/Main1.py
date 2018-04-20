import DataLoader
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas
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

def main():

    # --- < Set parameters> ---
    seed = 5
    test_size = 0.25
    path = '../Json/'
    ann_maxiter = 2000
    ann_epsilon = 1e-6
    reverse_datasets = False
    # --- </Set parameters> ---

    # --- < Load all data> ---
    DATA = DataLoader.DataLoader()
    all_Data = [DATA.Data_mias,
                DATA.Cleaned_data,
                DATA.FNA_gb,
                DATA.data_FNA,
                DATA.GSE58606_data
    ]
    all_Data_names = DATA.list_names
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
    fs_methods.append((sfsFunction(), 'sfs'))
    fs_methods.append((sfsFunction(), 'sbs'))
    fs_methods.append((chi2Function(), 'chi2'))
    fs_methods.append((entropyFunction(), 'entropy'))
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
    print('---*-*-*--- EOF ---*-*-*---')

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

def sfsFunction():
    f = lambda x, X, Y, model: SFS(model, k_features=x, forward=True,
            floating=False, n_jobs = 1)
    return f

def sbsFunction():
    f = lambda x, X, Y, model: SFS(model, k_features=x, forward=False,
            floating=False, n_jobs = 1)
    return f

def subsetFrom(mask, Xtest):
    mask = np.nonzero(mask)
    return Xtest[:,mask[0]]

def evaluateMethod(f, f_name, models, Xtr, Xtest, Ytr, Ytest, data_name):
    print('Evaluating ' + f_name + '...')
    results = {'CART':[], 'SVM':[], 'NB':[], 'ANN':[]}
    scoring = 'accuracy'
    features = Xtr.shape[1]
    step_sz = 1 if features < 100 else 100
    for num_features in range(1, features, step_sz):
        if f_name != 'sbs' and f_name != 'sfs':
            # Filter methods extract features _before_ model is known
            Xtr_subset, X_mask = f(num_features, Xtr, Ytr)
            Xtest_subset = subsetFrom(X_mask, Xtest)
        for name, model in models:
            if f_name == 'sbs' or f_name == 'sfs':
                # Wrapper methods extract features _after_ model is known
                sXs = f(num_features, Xtr, Ytr, model)
                Xtr_subset = sXs.fit_transform(Xtr, Ytr)
                Xtest_subset = subsetFrom(sXs.k_feature_idx_, Xtest)
                parameters = sXs.get_params(False)
                model.set_params(**parameters['estimator'].get_params())
            else:
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
