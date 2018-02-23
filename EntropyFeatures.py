from readData import convertData as getData
from sklearn.feature_selection import mutual_info_classif
import numpy as np

class EntropyFeatures(object):
    
    """
    - Needs to be initialized with data, target and labels, by:
        >>> EntropyFeatures = EntropyFeatures(data, target, label)
    - Initialization sets up the class with an index of 'best' features
    - Extract dataset with k (integer) features by:
        >>> EntropyFeatures.selectData(k)
    """
    
    def __init__(self, data, target, labels):
        self.data = data
        self.target = target
        self.labels = labels
        self.top_features_index = self.rankEntropies()

    def rankEntropies(self):
        ''' 
        Runs by initializing class, constructs index of top features
        '''
        feature_entropies = mutual_info_classif(self.data, self.target)
        sorted_entropies = sorted(feature_entropies)[::-1]
        return [
            np.asscalar(np.where(
                feature_entropies == x)[0]
            ) for x in sorted_entropies
        ]
    
    def selectData(self, k):
        '''
        @inparam: k, interger representing number of attributes
        @return: numpy array with only the k best attributes
        '''
        return self.data[:, self.top_features_index[:k]]
    
    def selectLabels(self, k):
        '''
        @inparam: k, interger representing number of attributes
        @return: numpy array with only the k best attribute names
        '''
        return self.labels[self.top_features_index[:k]]

# ---*--- For testing purposes only ---*---
# data, target, label = getData()
# EntropyFeatures = EntropyFeatures(data, target, label)
# print(EntropyFeatures.selectData(5))
# print(EntropyFeatures.selectLabels(5))
