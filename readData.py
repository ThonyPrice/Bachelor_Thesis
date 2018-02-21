import csv
import numpy as np

def getData():
    dataReader = csv.reader(open("data.csv", newline=''), delimiter=',', quotechar='|')
    dataMartix = np.zeros(shape=(569,32), dtype=object)
    i = 0

    for row in dataReader:

        if i == 0: #first row includes the labels

            i += 1
            labels = np.asarray(row[:-1]) # Remove last element (empty)
            labels = np.delete(row, [1]) # Remove diagnosis

        else:
            dataMartix[i-1] = row
            i += 1



    return dataMartix, labels

def convertData():
    data, labels = getData()
    target = data[:, [1]]
    data = np.delete(data, [1], 1)
    data = data.astype(float)
    target[target == 'M'] = 1.
    target[target == 'B'] = 0.
    target = np.squeeze(np.asarray(target))
    target = target.astype(float)
    return data, target, labels

"""
def getLabels():
    dataReader = csv.reader(open("data.csv", newline=''), delimiter=',', quotechar='|')
    for row in dataReader:
        labels = row
        break
    labels = np.asarray(labels[:-1]) # Remove last element (empty)
    labels = np.delete(labels, [1]) # Remove diagnosis
    return np.asarray(labels)
"""
