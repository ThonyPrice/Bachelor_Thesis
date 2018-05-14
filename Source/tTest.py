import sys
from scipy import stats

# Data - Copied from tabls in report
acc_fs_ann = [0.59, 0.84, 0.90, 0.74]
acc_fs_car = [0.70, 0.84, 0.94, 0.98]
acc_fs_nbs = [0.75, 0.75, 0.96, 0.98]
acc_fs_svm = [0.59, 0.84, 0.94, 0.94]

acc_full_ann = [0.57, 0.68, 0.51, 0.41]
acc_full_car = [0.75, 0.95, 0.95, 0.95]
acc_full_nbs = [0.57, 0.93, 0.96, 0.96]
acc_full_svm = [0.74, 0.90, 0.64, 0.64]

acc_fs = acc_fs_ann+acc_fs_car+acc_fs_nbs+acc_fs_svm
acc_full = acc_full_ann+acc_full_car+acc_full_nbs+acc_full_svm


print('--- Student t-test [FULL] ---')
t_stat, prob = (stats.ttest_ind(acc_fs, acc_full))
print(  't-stat: ', t_stat, '\n', \
        'prob: ', prob)

print('--- Student t-test [ANN] ---')
t_stat, prob = (stats.ttest_ind(acc_fs_ann, acc_full_ann))
print(  't-stat: ', t_stat, '\n', \
        'prob: ', prob)

print('--- Student t-test [CART] ---')
t_stat, prob = (stats.ttest_ind(acc_fs_car, acc_full_car))
print(  't-stat: ', t_stat, '\n', \
        'prob: ', prob)

print('--- Student t-test [NB] ---')
t_stat, prob = (stats.ttest_ind(acc_fs_nbs, acc_full_nbs))
print(  't-stat: ', t_stat, '\n', \
        'prob: ', prob)

print('--- Student t-test [SVM] ---')
t_stat, prob = (stats.ttest_ind(acc_fs_svm, acc_full_svm))
print(  't-stat: ', t_stat, '\n', \
        'prob: ', prob)
