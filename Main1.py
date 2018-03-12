import pandas
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
from sklearn.feature_selection import RFECV
import pprint as pp

def evaluate_filter(f, fs_method):
    print("Evaluating", fs_method)
    # evaluate each model in turn
    rez = {'CART':[], 'SVM':[], 'NB':[], 'ANN':[]}
    results = []
    names = []
    scoring = 'accuracy'
    features = list(range(1, X.shape[1]))
    for num_features in features:
        X_new = f(num_features)
        for name, model in models:
            kfold = model_selection.KFold(n_splits=10, random_state=seed)
            cv_results = model_selection.cross_val_score(model, X_new, Y, cv=kfold, scoring=scoring)
            results.append(cv_results)
            rez[name].append((num_features, cv_results.mean()))
            names.append(name)
            msg = "#features: %f, Model: %s:  %f (%f)" % (num_features, name, cv_results.mean(), cv_results.std())
            print(msg)
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
    plt.show()
    plt.savefig('plots/%s.png' % (fs_method), bbox_inches='tight')

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
            rfecv.fit(X, Y)
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
    plt.show()
    plt.savefig('plots/RFS.png', bbox_inches='tight')

# load dataset
dataframe = pandas.read_csv("data.csv")
dataframe = dataframe.drop(['id'], axis=1)
array = dataframe.values
X = array[:,1:]
Y = array[:,0]

# prepare configuration for cross validation test harness
seed = 7

# prepare models
models = []
models.append(('CART', DecisionTreeClassifier()))
models.append(('SVM', SVC()))
models.append(('NB', GaussianNB()))
models.append(('ANN', MLPClassifier()))

# Evaluate Chi2
f = lambda x: SelectKBest(chi2, k=x).fit_transform(X, Y)
evaluate_filter(f, 'FS by Chi2')

# Evaluate Entropy
f = lambda x: SelectKBest(mutual_info_classif, k=x).fit_transform(X, Y)
evaluate_filter(f, 'FS by Entropy')

# Set SVM kernel to linear to funtion with RFS
models = [models[0]] + [('SVM', SVC(kernel='linear'))] + models[2:]
# Evaluate RFE
evaluate_wrapper()
