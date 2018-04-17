import pandas
import numpy as np

class DataLoader(object):
    """Load and clean all datasets"""
    def __init__(self):
        self.Cleaned_data = self.loadData3()
        self.Cleaned_data_name = 'Cleaned_data'
        self.data_FNA = self.loadData1()
        self.data_FNA_name = 'data_FNA'
        self.Data_mias = self.loadData4()
        self.Data_mias_name = 'Data_mias'
        self.FNA_gb = self.loadData5()
        self.FNA_gb_name = 'FNA_gb'
        self.GSE58606_data = self.loadData2()
        self.GSE58606_data_name = 'GSE58606_data'
        self.list_names = [self.FNA_gb_name,self.Cleaned_data_name,self.Data_mias_name,self.GSE58606_data_name]

    def loadData1(self):
        dataframe = pandas.read_csv("../Data/data_FNA.csv")
        dataframe = dataframe.drop(['id'], axis=1)
        array = dataframe.values
        X = array[:,1:]
        Y = array[:,0]
        Y[Y == 'B'] = 0
        Y[Y == 'M'] = 1
        Y = Y.astype('int')
        return [X, Y]

    def loadData2(self):
        dataframe = pandas.read_csv("../Data/GSE58606_data.csv")
        array = dataframe.values
        X = array[:,0:-2]
        Y = array[:,-2]
        Y = Y.astype('int')
        return [X, Y]

    def loadData3(self):
        dataframe = pandas.read_csv("../Data/Cleaned_data.csv")
        array = dataframe.values
        array = array
        X = array[:,1:]
        Y = array[:,0]
        Y = Y.astype('int')
        return [X, Y]

    def loadData4(self):
        dataframe = pandas.read_csv("../Data/Data_mias.txt", sep=' ')
        dataframe = dataframe.drop(['REFNUM'], axis=1)
        array = dataframe.values
        array = array[array[:,1]!='NORM',:]
        X = array[:,[0,1,3,4,5]]
        # convert labels to classes
        X[X == 'D'] = 0
        X[X == 'F'] = 1
        X[X == 'G'] = 2
        X[X == 'ARCH'] = 0
        X[X == 'ASYM'] = 1
        X[X == 'CIRC'] = 1
        X[X == 'CALC'] = 1
        X[X == 'MISC'] = 1
        X[X == 'SPIC'] = 1
        Y = array[:,2]
        Y[Y == 'B'] = 0
        Y[Y == 'M'] = 1
        Y = Y.astype('int')
        return X, Y

    def loadData5(self):
        dataframe = pandas.read_csv("../Data/FNA_gb.csv")
        array = dataframe.values
        X = array[:,0:-1]
        X[X == 'Absent'] = 0
        X[X == 'Present'] = 1
        Y = array[:,-1]
        Y[Y == 'Benign'] = 0
        Y[Y == 'Malignant'] = 1
        Y = Y.astype('int')
        return X, Y
