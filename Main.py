from readData import readData

"""
Proposed functionallity of Main file

1. Collect the data
2. Shuffle and split with ratio x, for each x (could be .5, .6, .7, .8, .9)
    2.1 Perform feature selection by filtering
    2.2 Evaluate performance on classification (run 10 times and take average)
    2.3 Run Wrapper methods on data and evaluate performance
3. Dump collected data in relevant files

"""

def main():
    # ---*--- Step 1 ---*---
    Data = readData()                   # Initialize data class
    splits = [.5, .6, .7, .8, .9]       # Let's do these splits of data
    features = list(range(5,10))        # How many selected features we'll test
    results = buildResultsDictionary()  # Here we'll store the results
    # ---*--- Step 2 ---*---
    for k in features:
        for ratio in splits:
            # This call must be synced qith the not yet implemented
            # shuffleSplit in readData class
            train_data, train_target,
            test_data, test_target = readData.shuffleSplit(ratio)
            # ---*--- Entropy features ---*---
            EntropyFeatures = EntropyFeatures(train_data, train_target)
            ef_data, ef_target = EntropyFeatures.selectData(k)
            prediction = EntropyFeatures.predict(test_data)
            f1_score = EntropyFeatures.score(prediction, test_target)
            results['Entropy']['DT'].append(f1_score)
            # ---*--- Chi square ---*---
            ChiSquare = ChiSquare(train_data, train_target)
            cs_data, cs_target = ChiSquare.selectData(k)
            prediction = ChiSquares.predict(test_data)
            f1_score = ChiSquares.score(prediction, test_target)
            results['Chi2']['DT'].append(f1_score)
            # ---*--- Sequential feature selection ---*---
            pass
            # ---*--- Recursive feeature selection ---*---
            pass

def buildResultsDictionary():
    '''
    Store the results of each run in a dictionary.
    The dictionary is double indexed on first method then classifier.
    Example, store the value of a run with Entropy selection and Decisiontrees:
        >>> results['Entropy']['DT'] = value
    '''
    results = {}
    methods = ['Entropy', 'Chi2', 'SFS', 'RFS']
    classifiers = ['DT', 'SVM', 'ANN', 'PM']
    for method in methods:
        results[method] = {}
    for method in methods:
        for classifier in classifiers:
            results[method][classifier] = []
    return results

if __name__ == '__main__':
    main()
