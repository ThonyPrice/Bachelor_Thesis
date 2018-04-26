import DataLoader
import json
import pandas as pd
import sys

def getFilesByDataset(path):
    list_data_names = DataLoader.DataLoader().list_names[:-1]
    all_datasets = []
    for name in list_data_names:
        dataset = (name, [])
        for method in ['_chi2', '_entropy', '_sbs', '_sfs']:
            dataset[1].append(path + name + method + '.json')
        all_datasets.append(dataset)
    return all_datasets

def manipulateData(datasets):
    classifiers = ['ANN', 'CART', 'NB', 'SVM']
    for name, dataset in datasets:
        # print(dataset)
        for file in dataset:
            data = readJson(file)
            # print(data['ANN'])
            # print('***')
            df1 = pd.DataFrame.from_records(data['ANN']))
            df2 = pd.DataFrame.from_records(data['CART']))
            df3 = pd.DataFrame.from_records(data['NB']))
            df4 = pd.DataFrame.from_records(data['SVM']))
            sys.exit(0)
            df = pd.read_json(file)
            df_1 = df['ANN'].to_frame()
            df_2 = df['CART'].to_frame()
            df_3 = df['NB'].to_frame()
            df_4 = df['SVM'].to_frame()
            print(df)
            print('****')
            # dd = df_1.add(df_2)
            print(df_1.add(df_2))
            sys.exit(0)
            data = averageOverClassifiers(data)

def readJson(filename):
    with open(filename) as json_data:
        data = json.load(json_data)
    return data

def computeAnova(df):
    pass

def plotAnova(df):
    pass

def main():
    # --- Where to collect data? ---
    path = '../Json2/'


    files = getFilesByDataset(path)
    df = manipulateData(files)
    df = computeAnova(df)
    plotAnova(df)

if __name__ == '__main__':
    main()
