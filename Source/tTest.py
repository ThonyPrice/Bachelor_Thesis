from scipy import stats

# Data - Copied from tabls in report
acc_full = [
    0.6,
    0.52714,
    0.57320,
    0.64143,
    0.76667,
    0.68786,
    0.89608,
    0.93714,
    0.76667,
    0.65976,
    0.93639,
    0.95905,
    0.56667,
    0.72690,
    0.89641,
    0.61381
]

acc_fs = [
    0.63333,
    0.83214,
    0.90196,
    0.72810,
    0.7,
    0.83214,
    0.91895,
    0.96524,
    0.76667,
    0.74071,
    0.93693,
    0.97286,
    0.56667,
    0.83214,
    0.91373,
    0.92952
]

print('--- Student t-test ---')
t_stat, prob = (stats.ttest_ind(acc_full, acc_fs))
print(  't-stat: ', t_stat, '\n', \
        'prob: ', prob)
