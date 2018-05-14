import json
import pandas as pd
import numpy as np
import DataLoader as DataLoader
import matplotlib.pyplot as plt
import sys

def getFilesByDataset(path):
    list_data_names = DataLoader.DataLoader().list_names[:-1]
    all_datasets = []
    for name in list_data_names:
        for method in ['_chi2', '_entropy', '_sbs', '_sfs']:
            all_datasets.append(path + name + method + '.json')
    return all_datasets

def readJson(filename):
    with open(filename) as json_data:
        data = json.load(json_data)
    return data

def getMaxAndFull(f_name, classifier):
    data = readJson(f_name)
    accs = [i[2] for i in data[classifier]]
    return (max(accs[:-1]), accs[-1])

def concludeRow(file, classifier, dataset_vals, all_values):
    _, max = getMaxAndFull(file, classifier)
    dataset_vals.append(max)
    all_values.append(dataset_vals)
    return ([], all_values)

def mkTable(path, files):
    classifiers = ['ANN', 'CART', 'NB', 'SVM']
    for classifier in classifiers:
        # --- Extract data for each classifier ---
        all_values = []
        dataset_vals = []
        for i, file in enumerate(files):
            if len(dataset_vals) == 4:
                dataset_vals, all_values = concludeRow(
                    file, classifier, dataset_vals, all_values
                )
            acc, _ = getMaxAndFull(file, classifier)
            dataset_vals.append(acc)
        _, all_values = concludeRow(
            file, classifier, dataset_vals, all_values
        )

        # --- Mk DataFrame ---
        df = pd.DataFrame.from_records(all_values)
        df = df.transpose()
        df = renameLabels(df)

        # --- Include Gain row ---
        fs_accs = df[:4].max()
        full_accs = df.loc['Full']
        gain = (fs_accs/full_accs)-1
        df.loc['Gain'] = gain

        mkLaTeX(df, classifier)

def mkLaTeX(df, classifier):
    f1 = lambda x : '%1.2f' % x
    with open("../tables/" + classifier + '.tex','w') as tf:
        tf.write(df.to_latex(
            buf=None, columns=None, col_space=None,
            header=True, index=True, na_rep='NaN',
            formatters=[f1,f1,f1,f1,f1], float_format=True, sparsify=None,
            index_names=True, bold_rows=False, column_format='|l|l|l|l|l|l|l|',
            longtable=None, escape=None, encoding=None, decimal='.',
            multicolumn=None, multicolumn_format=None, multirow=False)
        )

def renameLabels(df):
    df = df.rename({0 : 'Chi2',
                    1 : 'Entropy',
                    2 : 'SBS',
                    3 : 'SFS',
                    4 : 'Full'
    }, axis = 'index')
    df = df.rename({0 : 'MIAS',
                    1 : 'EN',
                    2 : 'RHH',
                    3 : 'WBCD'
    }, axis = 'columns')
    return df

def main():
    path = "../Json2/"
    files = getFilesByDataset(path)
    path = "../Json2/"
    mkTable(path, files)
    print('---EOF---')

if __name__ == '__main__':
    main()
