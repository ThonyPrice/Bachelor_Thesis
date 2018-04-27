import DataLoader
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from operator import add
# Necessay ANOVA imports
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
            for i in range(attributes-1):
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
            results.append([
                np.asarray(mean_accuracy_for_each_attribute).max(),
                data_name,
                method_name
            ])

    df = pd.DataFrame.from_records(results)
    return df

def readJson(filename):
    with open(filename) as json_data:
        data = json.load(json_data)
    return data

def plotData(df):
    d1, d2, d3, d4 = [], [], [], []
    for i in range(4):
        d1.append(df.iloc[i*4+0,0])
        d2.append(df.iloc[i*4+1,0])
        d3.append(df.iloc[i*4+2,0])
        d4.append(df.iloc[i*4+3,0])
    print(d1)
    x_axis = ['EN (4)', 'MIAS (5)', 'RHH (10)', 'WBCD (30)']
    plt.plot(x_axis, d1, marker='^', label='Chi2')
    plt.plot(x_axis, d2, marker='^', label='Entropy')
    plt.plot(x_axis, d3, marker='^', label='SBS')
    plt.plot(x_axis, d4, marker='^', label='SFS')
    plt.suptitle('Mean accuarcy comparing datasets & FS-methods')
    plt.xlabel('Dataset (#features)')
    plt.ylabel('Mean accuracy over classifiers')
    plt.legend()
    plt.show()
    return

def computeAnova(data):
    formula = 'accuracy ~ dataset + method'
    model = ols(formula, data).fit()
    aov_table = anova_lm(model, typ=2)
    eta_squared(aov_table)
    omega_squared(aov_table)
    print(aov_table)
    return

def eta_squared(aov):
    aov['eta_sq'] = 'NaN'
    aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
    return aov

def omega_squared(aov):
    mse = aov['sum_sq'][-1]/aov['df'][-1]
    aov['omega_sq'] = 'NaN'
    aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*mse))/(sum(aov['sum_sq'])+mse)
    return aov

def renameCols(df):
    df = df.rename({0 : 'accuracy',
                    1 : 'dataset',
                    2 : 'method'
    }, axis = 'columns')
    return df

def main():
    path = '../Json2/' # Where to collect data? ---
    files = getFilesByDataset(path)
    df = manipulateData(files, path)
    print('\n---*--- Plotting data ---*---\n')
    plotData(df)
    print('\n---*--- Computing anova ---*---\n')
    df = renameCols(df)
    df = computeAnova(df)
    print('\n---*--- EOF ---*---\n')

if __name__ == '__main__':
    main()
