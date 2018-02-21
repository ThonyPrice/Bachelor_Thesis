import readData
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2



data = readData.data()
X = []
for row in data:
    print(row[2:])
    X.append(row[2:])

print(X)
y = []



for row in data:
    if row[1] == "M":
        y.append(1)
    else:
        y.append(0)



print(X.shape)

X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
print(X_new.shape)



print(X_new)
