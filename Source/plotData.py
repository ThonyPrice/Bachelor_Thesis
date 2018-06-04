import json
import sys
import numpy as np
import DataLoader as DataLoader
import matplotlib.pyplot as plt
import math

def getFilesByDataset(path):
    list_data_names = DataLoader.DataLoader().list_names[:-1]
    all_datasets = []
    for name in list_data_names:
        for method in ['_chi2', '_entropy', '_sbs', '_sfs']:
            all_datasets.append(path + name + method + '.json')
    return all_datasets

def plotFilter1(rez, filename):
    # Plot each dataset/classifier/fs-method by itself
    plt.subplot(2, 2, 1)
    plt.plot(*zip(*rez['CART']))
    plt.title('CART')
    plt.ylabel('Mean accuracy')
    plt.subplot(2, 2, 2)
    plt.plot(*zip(*rez['SVM']))
    plt.title('SVM')
    plt.subplot(2, 2, 3)
    plt.plot(*zip(*rez['NB']))
    plt.title('NB')
    plt.xlabel('# features')
    plt.ylabel('Mean accuracy')
    plt.subplot(2, 2, 4)
    plt.plot(*zip(*rez['ANN']))
    plt.title('ANN')
    plt.xlabel('# features')
    plt.tight_layout()
    plt.suptitle(filename)
    plt.savefig('../plots/%s.png' %(filename+'_subplots'), bbox_inches='tight')
    plt.close()

def plotFilter2(rez,filename):
    # Plot combined comparisions for each classifier
    plt.ylabel('Mean accuracy')
    plt.xlabel('# features')
    plt.plot(*zip(*rez['CART']), label="CART")
    plt.plot(*zip(*rez['SVM']), label="SVM")
    plt.plot(*zip(*rez['NB']), label="NB")
    plt.plot(*zip(*rez['ANN']), label="ANN")
    plt.legend()
    plt.suptitle(filename)
    plt.savefig('../plots/%s.png' %(filename.strip('.json')+'_combined'))
    plt.close()

def plotWithStd(rez, filename):
    # Plot combined comparisions for each classifier with errorbars
    path = '../plots_with_std/'
    plt.ylabel('Mean accuracy')
    plt.xlabel('# features')
    for classifier in ['ANN', 'CART', 'NB', 'SVM']:
        features, mean, std = extractData(rez, classifier)
        plt.errorbar(
            features, mean, std, marker='^',
            capsize=3, label=classifier
        )
    plt.legend()
    plt.suptitle(filename)
    plt.savefig(path + '%s.png' %(filename.strip('.json')+'_combined'))
    plt.close()

def plotWithStdFill(rez, filename):
    # Plot combined comparisions for each classifier with errorbars
    path = '../plots_with_std_fill/'
    plt.ylabel('Mean accuracy')
    plt.xlabel('# features')
    for classifier in ['ANN', 'CART', 'NB', 'SVM']:
        features, mean, std = extractData(rez, classifier)
        plot_fill(features, mean, std, classifier)
    plt.legend()
    plt.suptitle(filename)
    plt.savefig(path + '%s.png' %(filename.strip('.json')+'_combined'))
    plt.close()

def mk4plot(data, filename):
    # Plot combined comparisions for each classifier with errorbars
    path = '../plots_with_std_fill/'
    plt.ylabel('Mean accuracy')
    plt.xlabel('# features')
    methods = ['chi2', 'entropy', 'sbs', 'sfs']
    dsets_name = ['d1', 'd2', 'd3', 'd4']
    for j, dataset in enumerate(data):
        for i, d in enumerate(dataset):
            features, mean, std = d
            std_error = np.divide(std, 2) # sqrt(N_datasets) N_datasets = 4 => 2
            plot_fill(features, mean, std_error, methods[i])
        plt.legend(fontsize=14)
        plt.savefig(path + filename + dsets_name[j] + '.png')
        plt.close()

def plot_fill(x, y, err, label):
    ax = plt.gca()
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(x, y, marker='^', label=label, color=color)
    ax.fill_between(x,
        [y[i]+err[i] for i in range(len(y))],
        [y[i]-err[i] for i in range(len(y))],
        facecolor=color, alpha=0.1
    )

def readJson(filename):
    with open(filename) as json_data:
        data = json.load(json_data)
    return data

def extractData(rez, classifier):
    data = rez[classifier]
    features, mean, std = [], [], []
    for datapoint in data:
        features.append(datapoint[0])
        mean.append(datapoint[2])
        std.append(datapoint[3])
    return features, mean, std

def plot(path,filename):
    rez = readJson(path+filename)
    # plotFilter1(rez, filename)
    # plotFilter2(rez, filename)
    # plotWithStd(rez, filename)
    plotWithStdFill(rez, filename)

def main():
    # --- Group by classifier ---
    path = "../Json2/"
    files = getFilesByDataset(path)
    DATA = DataLoader.DataLoader()
    list_data_names= DATA.list_names[:-1]
    for classifier in ['ANN', 'CART', 'NB', 'SVM']:
        plots = []
        for dataset in list_data_names:
            plot_one_dataset = []
            for method in ['_chi2', '_entropy', '_sbs', '_sfs']:
                f_name = path + dataset + method + '.json'
                data = readJson(f_name)
                data = extractData(data, classifier)
                plot_one_dataset.append(data)
            plots.append(plot_one_dataset)
        mk4plot(plots, classifier)

    sys.exit(0)
    # --- Group by FS method ---
    DATA = DataLoader.DataLoader()
    list_data_names= DATA.list_names
    for name in list_data_names:
        for method in ['_chi2', '_entropy', '_sbs', '_sfs']:
            plot(path, name+method+'.json')

if __name__ == '__main__':
    main()
