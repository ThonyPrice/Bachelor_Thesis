import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc, roc_auc_score
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


def plotFilter(rez, fs_method,acc):
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
    plt.plot(acc)
    plt.title('roc_curve')
    plt.xlabel('# features')
    plt.tight_layout()
    plt.suptitle(fs_method)
    plt.savefig('../plots/updated_%s.png' % (fs_method.replace(" ", "_")), bbox_inches='tight')
    plt.show()

def evaluate_filter(fs_method,X, X_test, Y, Y_test):
    print("Evaluating", fs_method)
    # evaluate each model in turn
    rez = {'CART':[], 'SVM':[], 'NB':[], 'ANN':[]}
    results = []
    names = []
    scoring = 'accuracy'
    features = list(range(1, X.shape[1]))
    i = 1
    acc = []
    for num_features in features:
        SKB = SelectKBest(chi2, k=i)
        X_new = SKB.fit_transform(X, Y)
        X_mask = SKB.get_support()
        X_mask = np.nonzero(X_mask)
        X_new_test = X_test[:,X_mask[0]]

        for name, model in models:
            model.fit(X_new, Y)

            y_score = model.fit(X_new, Y).decision_function(X_new_test)
            #print(model.predict(X_new_test))
            #print(Y_test)
            model_score = model.score(X_new_test, Y_test)
            rez[name].append((num_features, model_score))
            msg = "#features: %f, Model: %s:  %f (%f)" % (num_features, name, model_score, 0)
            print(msg)
            #----------roc curve-------------


            roc = roc_auc_score(Y_test, y_score)
            acc.append(roc)

        i += 1

    plotFilter(rez, fs_method, acc)



    #plt.plot(list(range(1,len(features)+1)), acc)
    #plt.show()


seed = 5
# load dataset
dataframe = pandas.read_csv("../Data/data_FNA.csv")
dataframe = dataframe.drop(['id'], axis=1)
array = dataframe.values
X = array[:,1:]
Y = array[:,0]
X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.25, random_state=seed)

X[X == 'B'] = 0
X[X == 'M'] = 1
Y_test[Y_test == 'B'] = 0
Y_test[Y_test == 'M'] = 1
X_test[X_test == 'B'] = 0
X_test[X_test == 'B'] = 1

Y[Y == 'B'] = 0
Y[Y == 'M'] = 1

Y = Y.astype('int')
X = X.astype('int')
Y_test = Y_test.astype('int')
X_test = X_test.astype('int')




# prepare models
models = []
#models.append(('CART', DecisionTreeClassifier()))
models.append(('SVM', SVC(C=1.0, kernel='poly', degree=3, gamma='auto', coef0=0.0, shrinking=True,
probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False,
max_iter=100000, decision_function_shape='ovr', random_state=None)))
#models.append(('NB', GaussianNB()))
#models.append(('ANN', MLPClassifier()))



evaluate_filter('FS by Chi2', X, X_test, Y, Y_test)
