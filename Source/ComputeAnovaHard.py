'''
DO NOT RUN THIS FILE,
RUN ComputeAnova.py INSTEAD
'''

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
        dataset_accuracies = []
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
                # Average each fold over classifiers
                mean_each_fold_all_classifiers \
                    = [x/4 for x in mean_each_fold_all_classifiers]
                # Mean of all folds
                mean_accuracy_all_folds_all_classifiers \
                    = np.asarray(mean_each_fold_all_classifiers).mean()
                mean_accuracy_for_each_attribute.append(
                    mean_accuracy_all_folds_all_classifiers
                )
            # Add max of accuracies over number of attributes
            dataset_accuracies.append(
                np.asarray(mean_accuracy_for_each_attribute).max()
            )
        # Add dataset accuracy for each method
        results.append(dataset_accuracies)

    # Build dataframe
    return pd.DataFrame.from_records(results)

def readJson(filename):
    with open(filename) as json_data:
        data = json.load(json_data)
    return data

def mkAnovaTable(df):
    datasets, methods = df.shape
    anova_table =  [
        [0 for i in range(datasets)]
        for j in range(methods)
    ]
    for d in range(datasets):
        for m in range(methods):
            # Collect values
            data = []
            m_value = df.iloc[d,m]
            data.append(
                [m_value, 'curr_data', 'curr_method']
            )
            row_vals = df.loc[d].drop([m])
            for val in row_vals:
                data.append(
                    [val, 'curr_data', 'other_method']
                )
            col_vals = df.loc[:,m].drop([m])
            for val in col_vals:
                data.append(
                    [val, 'other_data', 'curr_method']
                )
            # Mk dataframe
            new_df = pd.DataFrame.from_records(data)
            new_df = renameCols(new_df)
            anova_val = computeAnova(new_df)
            anova_table[d][m] = anova_val
    df = pd.DataFrame.from_records(anova_table)
    df = renameLabels(df)
    return df

def renameCols(df):
    df = df.rename({0 : 'accuracy',
                    1 : 'dataset',
                    2 : 'method'
    }, axis = 'columns')
    return df

def computeAnova(data):
    formula = 'accuracy ~ dataset + method'
    model = ols(formula, data).fit()
    aov_table = anova_lm(model, typ=2)
    eta_squared(aov_table)
    omega_squared(aov_table)
    # Extract F-values
    data_Fval = aov_table.loc['dataset','F']
    method_Fval = aov_table.loc['method','F']
    return (data_Fval, method_Fval)

def eta_squared(aov):
    aov['eta_sq'] = 'NaN'
    aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
    return aov

def omega_squared(aov):
    mse = aov['sum_sq'][-1]/aov['df'][-1]
    aov['omega_sq'] = 'NaN'
    aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*mse))/(sum(aov['sum_sq'])+mse)
    return aov

def renameLabels(df):
    df = df.rename({0 : 'Chi2',
                    1 : 'Entropy',
                    2 : 'SBS',
                    3 : 'SFS'
    }, axis = 'columns')
    df = df.rename({0 : 'MIAS',
                    1 : 'EN',
                    2 : 'RHH',
                    3 : 'WBCD'
    }, axis = 'index')
    return df

def plotAnova(df):
    for dataset in ['MIAS', 'EN', 'RHH', 'WBCD']:
        data_points = df.loc[dataset]
        f_dataset_vals = [value[0] for value in data_points]
        f_method_vals = [value[1] for value in data_points]
        x_vals = ['Chi2', 'Entropy', 'SBS', 'SFS']
        plt.plot(x_vals, f_dataset_vals, label=dataset)
    plt.legend()
    plt.show()
    sys.exit(0)
    return

def main():
    path = '../Json2/' # Where to collect data? ---
    files = getFilesByDataset(path)
    df = manipulateData(files, path)
    df = mkAnovaTable(df)
    print('***')
    print(df)
    print('***')
    plotAnova(df)


if __name__ == '__main__':
    pass
    # main()
