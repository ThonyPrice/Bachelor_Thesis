from readData import readData
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np

def select(k):

    data = readData().data
    target = readData().targets
    labels = readData().labels

    X = data

    y = target
    X_new = SelectKBest(chi2, k).fit(X, y)
    return X_new, X, labels, target

def getBestFeatures(k):
    X_new, X, labels, target = select(k)
    feature = []
    for i in X_new.get_support(True):
        feature.append(labels[i])
    return feature


def getX_new(k):
    X_new, X, labels, target = select(k)
    return X_new.transform(X), getBestFeatures(k), target
