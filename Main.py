"""
Proposed functionallity of Main file

1. Collect the data 
2. Shuffle and split with ratio x, for each x (could be .5, .6, .7, .8, .9)
    2.1 Perform feature selection by filtering
    2.2 Evaluate performance on classification (run 10 times and take average)
    2.3 Run Wrapper methods on data and evaluate performance
3. Dump collected data in relevant files

"""

# ---*--- Step 1 ---*---

Data = readData()               # Initialize data class
splits = [.5, .6, .7, .8, .9]   # Let's do these splits of data
results = {}                    # Here we'll store the results
methods = ['Entropy', 'Chi2', 'SFS', 'RFS']
classifiers = ['TP', 'SVM', 'ANN', 'PM']
results[method] = {} for method in methods
results[method][classifier] = [] for 
    method in methods for classifier in classifiers
print(results)
for ratio in splits:
    pass
