import chiSquare
from sklearn import svm
from sklearn import metrics

class SVM(object):

    """docstring for SVM..."""

    def __init__(self, data, target):
        # Split into test and training data
        #permutate data?
        ratio = 0.6
        split = int(len(data)*ratio)
        self.train_data = data[:split]
        self.train_target = target[:split]
        self.test_data = data[split:]
        self.test_target = target[split:]
        self.clf = svm.SVC(C=5.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=2, gamma='auto', kernel="poly",
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
        self.trainClf()
        prediction = self.predictClf()
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

    def predictClf(self):
        return self.clf.predict(self.test_data)


def main():
    numberOfFeatures = 5
    data, lable, target = chiSquare.getX_new(numberOfFeatures)
    testSVM = SVM(data, target)


main()
