from readData import data as getData
import numpy as np
import pprint as pp

def convertData():
    data = getData()
    target = data[:, [1]]
    data = np.delete(data, [1], 1)
    data = data.astype(float)
    target[target == 'M'] = 1.
    target[target == 'B'] = 0.
    target = np.squeeze(np.asarray(target))
    return data, target

data, target = convertData()    
pp.pprint(data[0:20])
pp.pprint(np.shape(target))