import csv
import numpy as np

def data():
    dataReader = csv.reader(open("data.csv", newline=''), delimiter=',', quotechar='|')
    dataMartix = np.zeros(shape=(569,32), dtype=object)
    i = 0

    for row in dataReader:
        if i == 0:
            #print(row)
            i += 1
        else:
            dataMartix[i-1] = row
            i += 1

    return dataMartix

def convertData():
    data = getData()
    target = data[:, [1]]
    data = np.delete(data, [1], 1)
    data = data.astype(float)
    target[target == 'M'] = 1.
    target[target == 'B'] = 0.
    target = np.squeeze(np.asarray(target))
    return data, target