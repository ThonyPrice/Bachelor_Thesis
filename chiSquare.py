import readData
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np

def select():
    data, target, labels = readData.convertData()
    X = data

    y = target
    X_new = SelectKBest(chi2, k=3).fit(X, y)
    return X_new, X, labels

def getBestFeatures():
    X_new, X, labels = select()
    feature = []
    for i in X_new.get_support(True):
        feature.append(labels[i])
    return feature


def getX_new():
    X_new, X, labels = select()
    return X_new.transform(X)

print(select())


print(getBestFeatures())

print(getX_new())
