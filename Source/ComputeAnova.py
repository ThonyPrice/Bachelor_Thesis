import DataLoader
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from operator import add

# Copied imports
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.graphics.factorplots import interaction_plot
import matplotlib.pyplot as plt
from scipy import stats

def getFilesByDataset(path):
    list_data_names = DataLoader.DataLoader().list_names[:-1]
    all_datasets = []
    for name in list_data_names:
        dataset = (name, [])
        for method in ['_chi2', '_entropy', '_sbs', '_sfs']:
            dataset[1].append(path + name + method + '.json')
        all_datasets.append(dataset)
    return all_datasets

def manipulateData(datasets, path):
    classifiers = ['ANN', 'CART', 'NB', 'SVM']
    results = []
    for data_name, dataset in datasets:
        # Iterate over each dataset (i.e. Data_mias)
        for file in dataset:
            # Iterate over each dataset file (i.e. Data_mias_chi2.json)
            #   That is, every FS method applied to each dataset
            method_name = file.replace(path+data_name+'_', '') \
                              .replace('.json', '')
            data = readJson(file)
            df1 = pd.DataFrame.from_records(data['ANN'])
            df2 = pd.DataFrame.from_records(data['CART'])
            df3 = pd.DataFrame.from_records(data['NB'])
            df4 = pd.DataFrame.from_records(data['SVM'])
            attributes, _ = df1.shape
            mean_accuracy_for_each_attribute = []
            for i in range(attributes):
                # Iterate over each attribute
                mean_each_fold_all_classifiers = list(map(sum, zip(
                    df1.iloc[i,1],
                    df2.iloc[i,1],
                    df3.iloc[i,1],
                    df4.iloc[i,1]
                )))
                mean_each_fold_all_classifiers \
                    = [x/4 for x in mean_each_fold_all_classifiers]
                mean_accuracy_all_folds_all_clssifiers \
                    = np.asarray(mean_each_fold_all_classifiers).mean()
                results.append([
                    data_name,
                    method_name,
                    mean_accuracy_all_folds_all_clssifiers
                ])
            # results.append(mean_accuracy_for_each_attribute)
    df = pd.DataFrame.from_records(results)
    df = renameLabels2(df)
    print(df)
    anova_val = computeAnova(df)
            # sys.exit(0)

            # results.append((file, np.asarray(mean_accuracy_for_each_attribute).mean()))
            # results.append((file, np.asarray(mean_values).max()))
            # results.append([data_name, method_name, np.asarray(mean_accuracy_for_each_attribute).max()])
            # results.append(mean_accuracy_for_each_attribute)

    df = pd.DataFrame.from_records(results)
    print(df)

    # print(df.iloc[1])
    sys.exit(0)
    df = renameLabels2(df)
    print(df)
    return df
    # sys.exit(0)
    # for tup in results:
    #     print(tup)
    #
    l1, l2, l3, l4 = [], [], [], []
    for i in range(4):
        l1.append(results[4*i][1])
        l2.append(results[4*i+1][1])
        l3.append(results[4*i+2][1])
        l4.append(results[4*i+3][1])

    x = ['D1', 'D2','D3','D4']
    plt.plot(x, l1, label='Chi2')
    plt.plot(x, l2, label='Entropy')
    plt.plot(x, l3, label='SBS')
    plt.plot(x, l4, label='SFS')
    plt.legend()
    # plt.show()
    df = pd.DataFrame.from_records([l1, l2, l3, l4])
    df = renameLabels(df)
    print(df)
    return df

def mkDataFrame(acc_data, d_name, m_name):
    atts = len(acc_data)
    df = pd.DataFrame.from_records([
        [d_name]*atts,
        [m_name]*atts,
        acc_data
    ]).transpose()
    return df

def calc2way(data):
    print(type(data))
    print('TYPE: ', type(data.iloc[1,0]))
    formula = 'accuracy ~ dataset + method'
    model = ols(formula, data).fit()
    print('model: ', model)
    aov_table = anova_lm(model, typ=2)
    eta_squared(aov_table)
    omega_squared(aov_table)
    print(aov_table)
    sys.exit(0)

def computeAnova(data):
    print(type(data))
    print('TYPE: ', type(data.iloc[1,2]))
    formula = 'accuracy ~ dataset + method'
    model = ols(formula, data).fit()
    aov_table = anova_lm(model, typ=2)
    eta_squared(aov_table)
    omega_squared(aov_table)
    print(aov_table)

def readJson(filename):
    with open(filename) as json_data:
        data = json.load(json_data)
    return data

def renameLabels(df):
    df = df.rename({0 : 'Chi2',
                    1 : 'Entropy',
                    2 : 'SBS',
                    3 : 'SFS',
    }, axis = 'columns')
    df = df.rename({0 : 'MIAS',
                    1 : 'EN',
                    2 : 'RHH',
                    3 : 'WBCD'
    }, axis = 'index')
    return df

def renameLabels2(df):
    df = df.rename({0 : 'dataset',
                    1 : 'method',
                    2 : 'accuracy'
    }, axis = 'columns')
    return df

def renameLabels3(df):
    df = df.rename({0 : 'accuracy',
                    1 : 'dataset',
                    2 : 'method'
    }, axis = 'columns')
    return df

def eta_squared(aov):
    aov['eta_sq'] = 'NaN'
    aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
    return aov

def omega_squared(aov):
    mse = aov['sum_sq'][-1]/aov['df'][-1]
    aov['omega_sq'] = 'NaN'
    aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*mse))/(sum(aov['sum_sq'])+mse)
    return aov

def plotAnova(df):
    pass

def main():
    # --- Where to collect data? ---
    path = '../Json2/'

    files = getFilesByDataset(path)
    df = manipulateData(files, path)
    df = computeAnova(df)
    plotAnova(df)

if __name__ == '__main__':
    main()
