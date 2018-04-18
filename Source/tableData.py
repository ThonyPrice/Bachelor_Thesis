import json
import pandas
import numpy as np
import DataLoader as DataLoader
import matplotlib.pyplot as plt



def table(path, filename):

    X = pandas.read_json(path+filename)

    df = pandas.DataFrame(X)

    #use: \input{../tables/Cleaned_data_Chi2.json} in tex
    with open("../tables/" + filename + '.tex','w') as tf:
        tf.write(df.to_latex(buf=None, columns=None, col_space=None, header=True, index=True, na_rep='NaN',
        formatters=None, float_format=None, sparsify=None, index_names=True, bold_rows=False, column_format=None,
        longtable=None, escape=None, encoding=None, decimal='.', multicolumn=None, multicolumn_format=None, multirow=None))




DATA = DataLoader.DataLoader()

list_data_names= DATA.list_names

path = "../Json/"

for name in list_data_names:
    for method in ['_Chi2', '_Entropy']:
        table(path, name+method+'.json')
