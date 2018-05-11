import pandas as pd
import pprint as pp
import sys

def filterData(fname):
    data = []
    with open(fname, 'r') as f:
        # f.readline() # Skip labels
        for line in f:
            d = f.readline().split()
            add = True
            p = float(d[4])
            if p > 0.1:
                add = False
            elif p > 0.05:
                d.append('.')
            elif p > 0.01:
                d.append('*')
            elif p > 0.001:
                d.append('**')
            elif p < 0.001:
                d.append('***')
            if add:
                data.append(d)
    return data

def renameCols(df):
    df = df.rename({0 : 'Classif:method',
                    1 : 'diff',
                    2 : 'lwr',
                    3 : 'upr',
                    4 : 'p',
                    5 : 'sign'
    }, axis = 'columns')
    return df

def floatFormatter(x):
    try:
        return '%1.2f' % float(x)
    except:
        return x

def mkTex(df, output_file):
    f1 = lambda x : floatFormatter(x)
    with open(output_file,'w') as tf:
        tf.write(df.to_latex(
            buf=None, columns=None, col_space=None,
            header=True, index=False, na_rep='NaN',
            formatters=[f1,f1,f1,f1,f1,f1], float_format=True, sparsify=None,
            index_names=False, bold_rows=False, column_format='|l|l|l|l|l|l|',
            longtable=None, escape=None, encoding=None, decimal='.',
            multicolumn=None, multicolumn_format=None, multirow=False)
        )

def main():
    fname = '../Data/tukey_res.txt'
    output_file = '../tables/Tukeys_test.tex'
    data = filterData(fname)
    df = pd.DataFrame.from_records(data)
    df = renameCols(df)
    mkTex(df, output_file)

if __name__ == '__main__':
    main()
