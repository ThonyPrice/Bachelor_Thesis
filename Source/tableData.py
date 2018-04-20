import json
import pandas as pd
import numpy as np
import DataLoader as DataLoader
import matplotlib.pyplot as plt


def tableTransform(path, filename):

    X = pd.read_json(path+filename)
    df = pd.DataFrame(X)
    df = df.applymap(lambda x : x[1])
    df = df.max()
    return df




def table(path, lista_data_names):

    frames = []
    for name in list_data_names:
        for method in ['_Chi2', '_Entropy','_sbs', '_sfs']:
            print(path+name+method)
            df = tableTransform(path, name+method+'.json')
            print(df)

            # with open(path+name+method+'.json') as json_data:
            #
            #     tmp = json.load(json_data)
            #
            #     for key,value in tmp.items():
            #
            #         new_values = []
            #
            #         for element in value:
            #             new_values.append(element[1])
            #
            #
            #         value = max(new_values)
            #
            #         tmp[key] = value
            #
            # print(tmp)


            # df = pd.DataFrame.from_dict(tmp)

    def f1(x):
        return '%1.5f' % x[1]

    with open("../tables/" + filename + '.tex','w') as tf:
        tf.write(df.to_latex(buf=None, columns=None, col_space=None, header=True, index=True, na_rep='NaN',
        formatters=[f1,f1,f1,f1], float_format=True, sparsify=None, index_names=True, bold_rows=False, column_format='|l|l|l|l|l|',
        longtable=None, escape=None, encoding=None, decimal='.', multicolumn=None, multicolumn_format=None, multirow=False))




DATA = DataLoader.DataLoader()

list_data_names= DATA.list_names

path = "../Json/"

table(path, list_data_names)
