import csv
import numpy as np

dataReader = csv.reader(open("data.csv", newline=''), delimiter=',', quotechar='|')
dataMartix = np.zeros(shape=(569,32), dtype=object)
i = 0



for row in dataReader:
    if i == 0:
        print(row)
        i += 1
    else:
        dataMartix[i-1] = row
        i += 1

return dataMartix
