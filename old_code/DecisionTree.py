from sklearn import tree
from sklearn import metrics
from readData import convertData as getData
import numpy as np
import pprint as pp

class DecisionTree(object):
    
    """docstring for DecisionTree..."""
    
    def __init__(self, data, target):
        # Split into test and training data
        ratio = 0.5
        split = int(len(data)*ratio)
        self.train_data = data[:split]
        self.train_target = target[:split]
        self.test_data = data[split:]
        self.test_target = target[split:]
        self.clf = tree.DecisionTreeClassifier()
        self.trainClf()
        prediction = self.predict()
        for idx, val in enumerate(self.test_target):
            print(
                "Class: ", self.test_target[idx],
                "Predicted: ", prediction[idx]
            )
        print("F1 score: ", 
            metrics.f1_score(self.test_target, prediction)
        )

    
    def trainClf(self):
        self.clf.fit(self.train_data, self.train_target)

    def predict(self):
        return self.clf.predict(self.test_data)

# --- * --- This part below is for testing puposes only --- * ---

def main():
    data, target, _ = getData() 
    DecisionTree(data, target)

if __name__ == '__main__': 
    main()