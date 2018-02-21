import readData
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


data, target = readData.convertData()

print(data)

print(target)

X = data
y = target

print(X.shape)
print(y.shape)

X_new = SelectKBest(chi2, k=10).fit(X, y)

print(X_new.get_support(True) )
print(X_new.transform(X))
