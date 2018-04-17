


import json
import numpy as np
import DataLoader as DataLoader
import matplotlib.pyplot as plt



def readJson(filename):

    with open(filename) as json_data:
        data = json.load(json_data)

    return data

def plotFilter1(rez, filename):

    # Plot algorithmic comparsion
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


def plotFilter2(rez, filename):
    # Plot algorithmic comparsion

    rez = readJson(filename) #read json data


    plt.ylabel('Mean accuracy')
    plt.xlabel('# features')

    plt.plot(*zip(*rez['CART']), label="CART")
    plt.plot(*zip(*rez['SVM']), label="SVM")
    plt.plot(*zip(*rez['NB']), label="NB")
    plt.plot(*zip(*rez['ANN']), label="ANN")

    plt.legend()
    plt.suptitle(filename)
    plt.savefig('../plots/%s.png' %(filename+'_combined'))




def plot(filename):

    rez = readJson(filename)
    plotFilter1(rez, filename)
    plotFilter2(rez, filename)


DATA = DataLoader.DataLoader()

list_data_names= DATA.list_names



for name in list_data_names:
    for method in ['_FSbyChi2', '_FSbyEntropy']
    plot(name+method+'.json')
