import csv
import numpy as np


class readData(object):

    def __init__(self):
        self.raw_data = ""
        self.raw_labels = ""
        self.data = ""
        self.target = ""
        self.labels = ""
        self.getData()
        self.convertData()

    def getData(self):
        dataReader = csv.reader(open("data.csv", newline=''), delimiter=',', quotechar='|')
        dataMartix = np.zeros(shape=(569,32), dtype=object)
        i = 0
        for row in dataReader:
            if i == 0: #first row includes the labels
                i += 1
                labels = np.asarray(row[:-1]) # Remove last element (empty)
                labels = np.delete(row, [0,1],) # Remove ID and Remove diagnosis form labels
            else:
                dataMartix[i-1] = row
                i += 1
        self.raw_data =  dataMartix
        self.raw_labels = labels

    def convertData(self):
        data = self.raw_data
        labels = self.raw_labels
        target = data[:, [1]]
        data = np.delete(data, [0, 1], 1) #removes ID and removes target from data array
        data = data.astype(float)
        target[target == 'M'] = 1.
        target[target == 'B'] = 0.
        target = np.squeeze(np.asarray(target))
        target = target.astype(float)
        self.data = data
        self.targets = target
        self.labels = labels

    def splitTrainTest(self, data, target, ratio: int):
        '''
        Example:
        train_data, train_target, test_data, test_target = splitTrainTest(*args)
        '''
        return data[:ratio], target[:ratio], data[ratio:], target[ratio:]    
