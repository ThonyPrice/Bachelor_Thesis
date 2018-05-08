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

def IV_dataset_method(datasets, path):
    classifiers = ['ANN', 'CART', 'NB', 'SVM']
    plot_results = []
    anova_results = []
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
            anova_classifs = [[], [], [], []]
            for i in range(attributes-1):
                # Iterate over each attribute
                mean_each_fold_all_classifiers = list(map(sum, zip(
                    df1.iloc[i,1],
                    df2.iloc[i,1],
                    df3.iloc[i,1],
                    df4.iloc[i,1]
                )))
                # Save value of each classif for ANOVA
                anova_classifs[0].append(np.asarray(df1.iloc[i,1]).mean())
                anova_classifs[1].append(np.asarray(df2.iloc[i,1]).mean())
                anova_classifs[2].append(np.asarray(df3.iloc[i,1]).mean())
                anova_classifs[3].append(np.asarray(df4.iloc[i,1]).mean())
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
            plot_results.append([
                np.asarray(mean_accuracy_for_each_attribute).mean(),
                data_name,
                method_name,
                np.asarray(mean_accuracy_for_each_attribute).std()
            ])
            # Add each classifiers max for ANOVA computations
            for i in range(4):
                anova_results.append((max(anova_classifs[i]), data_name, method_name))

    plot_df = pd.DataFrame.from_records(plot_results)
    anova_df = pd.DataFrame.from_records(anova_results)
    x_axis = ['EN (4)', 'MIAS (5)', 'RHH (10)', 'WBCD (30)']
    return  renameCols(plot_df, 'dataset', 'method'), \
            renameCols(anova_df, 'dataset', 'method'), \
            x_axis

def IV_classifier_method(datasets, path):
    classifiers = ['ANN', 'CART', 'NB', 'SVM']
    plot_results = []
    anova_results = []
    for classifier in classifiers:
        classifier_accs = []
        for data_name, dataset in datasets:
            dataset_accs = []
            # Iterate over each dataset (i.e. Data_mias)
            for file in dataset:
                # method_name = file.replace(path+data_name+'_', '') \
                #                   .replace('.json', '')
                data = readJson(file)
                df1 = pd.DataFrame.from_records(data[classifier])
                attributes, _ = df1.shape
                mean_accuracy_for_each_attribute = []
                for i in range(attributes-1):
                    # Compute mean of all folds
                    mean_accuracy_for_each_attribute.append(
                        np.asarray(df1.iloc[i,1]).mean()
                    )
                # Add result of best perfoming #attributes
                dataset_accs.append(
                    np.asarray(mean_accuracy_for_each_attribute).max()
                )
            classifier_accs.append(dataset_accs)
        # Average classifier-method accuracies over datasets
        # - Save every instance for ANOVA computations
        methods = ['chi2', 'entropy', 'sbs', 'sfs']
        for i in range(4):
            for j in range(4):
                anova_results.append((
                    classifier_accs[j][i],
                    classifier,
                    methods[j]
                ))
        # - Save average and means for plotting
        avg, std = [0]*4, [0]*4
        for i in range(4):
            tmp = [
                classifier_accs[0][i],
                classifier_accs[1][i],
                classifier_accs[2][i],
                classifier_accs[3][i]
            ]
            avg[i] = np.asarray(tmp).mean()
            std[i] = np.asarray(tmp).std()
        methods = ['chi2', 'entropy', 'sbs', 'sfs']
        for i in range(4):
            plot_results.append((avg[i], classifier, methods[i], std[i]))

    plot_df = pd.DataFrame.from_records(plot_results)
    anova_df = pd.DataFrame.from_records(anova_results)
    x_axis = ['ANN', 'CART', 'NB', 'SVM']
    return  renameCols(plot_df, 'classifier', 'method'), \
            renameCols(anova_df, 'classifier', 'method'), \
            x_axis

def readJson(filename):
    with open(filename) as json_data:
        data = json.load(json_data)
    return data

def plotData(df, x_axis, plt_show):
    d1, d2, d3, d4 = [], [], [], []
    e1, e2, e3, e4 = [], [], [], []
    for i in range(4):
        # Append acc
        d1.append(df.iloc[i*4+0,0])
        d2.append(df.iloc[i*4+1,0])
        d3.append(df.iloc[i*4+2,0])
        d4.append(df.iloc[i*4+3,0])
        # Append std
        e1.append(df.iloc[i*4+0,3])
        e2.append(df.iloc[i*4+1,3])
        e3.append(df.iloc[i*4+2,3])
        e4.append(df.iloc[i*4+3,3])
    # --- Plot with std fill ---
    plot_fill(x_axis, d1, e1, 'Chi2')
    plot_fill(x_axis, d2, e2, 'Entropy')
    plot_fill(x_axis, d3, e3, 'SBS')
    plot_fill(x_axis, d4, e4, 'SFS')
    # --- Plot w std bars ---
    # plt.errorbar(x_axis, d1, e1, marker='^', label='Chi2')
    # plt.errorbar(x_axis, d2, e2, marker='^', label='Entropy')
    # plt.errorbar(x_axis, d3, e3, marker='^', label='SBS')
    # plt.errorbar(x_axis, d4, e4, marker='^', label='SFS')
    # --- Plot w/o std ---
    # plt.plot(x_axis, d1, e1, marker='^', label='Chi2')
    # plt.plot(x_axis, d2, e2, marker='^', label='Entropy')
    # plt.plot(x_axis, d3, e3, marker='^', label='SBS')
    # plt.plot(x_axis, d4, e4, marker='^', label='SFS')
    if 'ANN' in x_axis:
        plt.suptitle('Max accuarcy of datasets - comparing Classifiers & FS-methods')
        plt.xlabel('ML methods')
        plt.ylabel('Max accuracy over FS methods')
        plt.legend()
        plt.savefig('../plots_with_std_fill/%s.png' %('comp_classif_datasets'))
    else:
        plt.suptitle('Mean accuarcy of classifiers - comparing Datasets & FS-methods')
        plt.xlabel('Dataset (#features)')
        plt.ylabel('Mean accuracy over classifiers')
        plt.legend()
        plt.savefig('../plots_with_std_fill/%s.png' %('comp_acc_datasets'))
    if plt_show:
        plt.show()
    plt.close()
    return

def plot_fill(x, y, err, label):
    ax = plt.gca()
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(x, y, marker='^', label=label, color=color)
    ax.fill_between(x,
        [y[i]+err[i] for i in range(len(y))],
        [y[i]-err[i] for i in range(len(y))],
        facecolor=color, alpha=0.1
    )

def computeAnova(data):
    try:
        formula = 'accuracy ~ dataset * method'
        model = ols(formula, data).fit()
    except:
        formula = 'accuracy ~ classifier * method'
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

def renameCols(df, a, b):
    df = df.rename({0 : 'accuracy',
                    1 : a,
                    2 : b,
                    3 : 'std'
    }, axis = 'columns')
    return df

def main():
    path = '../Json2/'  # Where to collect data?
    plt_show = True    # Show plot?
    files = getFilesByDataset(path)

    plot_df, anova_df, x_axis = IV_dataset_method(files, path)
    print('\n---*--- Plotting data - Dataset/method ---*---\n')
    plotData(plot_df, x_axis, plt_show)
    print('\n---*--- Computing anova - Dataset/method ---*---\n')
    computeAnova(anova_df)

    plot_df, anova_df, x_axis = IV_classifier_method(files, path)
    print('\n---*--- Plotting data - Classifier/method ---*---\n')
    plotData(plot_df, x_axis, plt_show)
    print('\n---*--- Computing anova - Classifier/method ---*---\n')
    computeAnova(anova_df)

    print('\n---*--- EOF ---*---\n')

if __name__ == '__main__':
    main()
