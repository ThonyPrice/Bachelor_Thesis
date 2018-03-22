from readData import convertData as getData
from sklearn.feature_selection import mutual_info_classif
import csv
import numpy as np
import pprint as pp

def getLabels():
    dataReader = csv.reader(open("data.csv", newline=''), delimiter=',', quotechar='|')
    for row in dataReader:
        labels = row
        break
    labels = np.asarray(labels[:-1]) # Remove last element (empty)
    labels = np.delete(labels, [1]) # Remove diagnosis
    return np.asarray(labels)

def entropyFeatures():
    data, target, label = getData()
    print("DATA: ", data)
    X, y = data.astype(float), target.astype(float)
    return mutual_info_classif(X, y)

labels = getLabels()
entropies = entropyFeatures()
pairs = [(entropies[i], labels[i]) for i in range(len(entropies))]
pairs = sorted(pairs)
pairs = pairs[::-1]
pp.pprint(pairs)
