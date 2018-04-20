import json
import pandas
import numpy as np
import DataLoader as DataLoader
import matplotlib.pyplot as plt

def table(path, filename):

    X = pandas.read_json(path+filename)
    df = pandas.DataFrame(X)
    df = df.applymap(lambda x : x[1])
    df = df.max()
    print(df)

    #use: \input{../tables/Cleaned_data_Chi2.json} in tex

    def f1(x):
        return '%1.5f' % x[1]

    with open("../tables/" + filename + '.tex','w') as tf:
        tf.write(df.to_latex(buf=None, columns=None, col_space=None, header=True, index=True, na_rep='NaN',
        formatters=[f1,f1,f1,f1], float_format=True, sparsify=None, index_names=True, bold_rows=False, column_format='|l|l|l|l|l|',
        longtable=None, escape=None, encoding=None, decimal='.', multicolumn=None, multicolumn_format=None, multirow=False))




DATA = DataLoader.DataLoader()

list_data_names= DATA.list_names

path = "../Json/"

for name in list_data_names:
    for method in ['_Chi2', '_Entropy']:
        table(path, name+method+'.json')
