import json
import pandas as pd
import numpy as np
import DataLoader as DataLoader
import matplotlib.pyplot as plt
import sys

def mkTable(path, list_data_names):
    classifiers = ['CART', 'SVM', 'NB', 'ANN']
    for classifier in classifiers:
        data_row_vals = []
        for dataset in list_data_names:
            data_col_vals, data_ful_vals = [], []
            for method in ['_Chi2', '_Entropy','_sbs', '_sfs']:
                f_name = path + dataset + method +'.json'
                data_col_vals.append(getMax(f_name, classifier))
                data_ful_vals.append(getFull(f_name, classifier))

            data_col_vals.append(max(data_ful_vals))
            df = pd.DataFrame({dataset : data_col_vals})
            data_row_vals.append(df)

        df = pd.concat(data_row_vals, axis=1)
        df = renameLabels(df, list_data_names)
        print(df)
        mkLaTeX(df, classifier)

def mkLaTeX(df, classifier):
    f1 = lambda x : '%1.5f' % x
    with open("../tables/" + classifier + '.tex','w') as tf:
        tf.write(df.to_latex(
            buf=None, columns=None, col_space=None,
            header=True, index=True, na_rep='NaN',
            formatters=[f1,f1,f1,f1], float_format=True, sparsify=None,
            index_names=True, bold_rows=False, column_format='|l|l|l|l|l|',
            longtable=None, escape=None, encoding=None, decimal='.',
            multicolumn=None, multicolumn_format=None, multirow=False)
        )

def getMax(f_name, classifier):
    X = pd.read_json(f_name)
    df = pd.DataFrame(X)
    df = df.applymap(lambda x : x[1])
    df = df.drop(df.index[len(df)-1])
    idx = df.columns.get_loc(classifier)
    return df[classifier].max()

def getFull(f_name, classifier):
    X = pd.read_json(f_name)
    df = pd.DataFrame(X)
    df = df.applymap(lambda x : x[1])
    return df[classifier][len(df)-1]

def renameLabels(df, list_data_names):
    df = df.rename({0 : 'Chi2',
                    1 : 'Entropy',
                    2 : 'SBS',
                    3 : 'SFS',
                    4 : 'Full'
    }, axis = 'index')
    df = df.rename({list_data_names[0] : 'MIAS',
                    list_data_names[1] : 'EN',
                    list_data_names[2] : 'RHH',
                    list_data_names[3] : 'WBCD'
    }, axis = 'columns')
    return df

def main():
    DATA = DataLoader.DataLoader()
    list_data_names = DATA.list_names[:-1]
    path = "../Json_1d_run/"
    mkTable(path, list_data_names)
    print('---EOF---')

if __name__ == '__main__':
    main()
