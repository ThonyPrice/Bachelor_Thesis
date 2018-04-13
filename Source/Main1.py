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

def evaluate_filter(f, fs_method, X_tr, X_test, Y_tr, Y_test):
    print("Evaluating", fs_method)
    # evaluate each model in turn
    rez = {'CART':[], 'SVM':[], 'NB':[], 'ANN':[]}
    scoring = 'accuracy'
    features = list(range(1, X.shape[1]))
    for num_features in features:
        X_tr, X_mask = f(num_features)
        X_mask = np.nonzero(X_mask)
        X_te = X_test[:,X_mask[0]]
        for name, model in models:
            kfold = model_selection.KFold(n_splits=10)
            # model_selection.fit(X_tr, Y_tr)
            model_selection.cross_val_score(model, X_tr, Y_tr, cv=kfold, scoring=scoring)
            cv_results = model_selection.cross_val_score(model, X_te, Y_test, cv=kfold, scoring=scoring)

            # y_pred = model.predict(X_test)
            # model_score = model.score(X_te, Y_test)
            # kfold = model_selection.KFold(n_splits=100, random_state=seed)
            # cv_results = model_selection.cross_val_score(model, X_te, Y_test, cv=kfold, scoring=scoring)
            model_score = cv_results.mean()

            # model_mse = model.f1_score(X_test, Y_test)
            rez[name].append((num_features, model_score))
            msg = "#features: %i, Model: %s:  %f (%f)" % (num_features, name, model_score, 0.0)
            print(msg)

    filename = fs_method + '.json'
    print(filename)

    with open(filename, 'w') as fp:
        json.dump(rez, fp)

    plotFilter(rez, fs_method)

def plotFilter(rez, fs_method):
    # Plot algorithmic comparsion
    plt.subplot(2, 2, 1)
    plt.plot(*zip(*rez['CART']))
    plt.title('CART')
    plt.ylabel('Mean accuracy')
    plt.subplot(2, 2, 2)
    plt.plot(*zip(*rez['SVM']))
    plt.title('SVM')
    plt.subplot(2, 2, 3)
    plt.plot(*zip(*rez['NB']))
    plt.title('NB')
    plt.xlabel('# features')
    plt.ylabel('Mean accuracy')
    plt.subplot(2, 2, 4)
    plt.plot(*zip(*rez['ANN']))
    plt.title('ANN')
    plt.xlabel('# features')
    plt.tight_layout()
    plt.suptitle(fs_method)
    plt.savefig('../plots/updated_%s.png' % (fs_method.replace(" ", "_")), bbox_inches='tight')
    plt.show()

def evaluate_wrapper():
    print("Evaluating FS by RFS")
    # evaluate each model in turn
    rez = {'CART':[], 'SVM':[], 'NB':[], 'ANN':[]}
    scoring = 'accuracy'
    for name, model in models:
        try:
            kfold = model_selection.KFold(n_splits=10, random_state=seed)
            rfecv = RFECV(estimator=model, step=1, cv=kfold,
                          scoring=scoring)
            rfecv.fit(X, Y_tr)
            data = list(zip(list(range(1, len(rfecv.grid_scores_) + 1)), rfecv.grid_scores_))
            rez[name] = data
            msg = "Model: %s:  %f (%f)" % (name, rfecv.grid_scores_.mean(), rfecv.grid_scores_.std())
            print(msg)
        except:
            print("Model %s can't use RFS" % (name))
    plt.subplot(1, 2, 1)
    plt.plot(*zip(*rez['CART']))
    plt.title('CART')
    plt.ylabel('Mean accuracy')
    plt.xlabel('# features')
    plt.subplot(1, 2, 2)
    plt.plot(*zip(*rez['SVM']))
    plt.title('SVM')
    plt.xlabel('# features')
    plt.tight_layout()
    plt.suptitle("Feature selection by RFS")
    plt.savefig('plots/updated_RFS.png', bbox_inches='tight')
    plt.show()

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

seed = 5
# load dataset
DATA = DataLoader.DataLoader()
X, Y = DATA.FNA_gb
X_tr, X_test, Y_tr, Y_test = train_test_split(X, Y, test_size=0.25, random_state=seed)
print('X_test M/B ratio: ', np.size(np.where(X_test==0))/np.size(X_test)*100, '%')



# prepare models
models = []
models.append(('CART', DecisionTreeClassifier()))
models.append(('SVM', SVC()))
models.append(('NB', GaussianNB()))
models.append(('ANN', MLPClassifier()))

# # Evaluate SBS
# evaluate_sbs('FS by SBS')
# sys.exit('Early exit')
# Evaluate Chi2
f = lambda x: (SelectKBest(chi2, k=x).fit_transform(X_tr, Y_tr), SelectKBest(chi2, k=x).fit(X_tr, Y_tr).get_support())
evaluate_filter(f, 'FS by Chi2', X_tr, X_test, Y_tr, Y_test)

# Evaluate Entropy
f = lambda x: (SelectKBest(mutual_info_classif, k=x).fit_transform(X_tr, Y_tr), SelectKBest(mutual_info_classif, k=x).fit(X_tr, Y_tr).get_support())
evaluate_filter(f, 'FS by Entropy', X_tr, X_test, Y_tr, Y_test)

# # Set SVM kernel to linear to funtion with RFS
# models = [models[0]] + [('SVM', SVC(kernel='linear'))] + models[2:]
# # Evaluate RFE
# evaluate_wrapper()
