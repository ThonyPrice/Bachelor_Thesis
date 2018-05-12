import pandas as pd
import pprint as pp
import sys

def filterData(fname):
    data = []
    with open(fname, 'r') as f:
        for line in f:
            d = line.split()
            p = float(d[4])
            if p >= 0.1:
                d.append(' ')
            elif 0.05 <= p and p < 0.1:
                d.append('.')
            elif 0.01 <= p and p < 0.1:
                d.append('*')
            elif 0.001 <= p and p < 0.01:
                d.append('**')
            elif p < 0.001:
                d.append('***')
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

def floatFormatter(x, decimals):
    f_string = '%1.' + str(decimals) + 'f'
    try:
        return f_string % float(x)
    except:
        return x

def mkTex(df, output_file):
    f2 = lambda x : floatFormatter(x, 2)
    f3 = lambda x : floatFormatter(x, 3)
    with open(output_file,'w') as tf:
        tf.write(df.to_latex(
            buf=None, columns=None, col_space=None,
            header=True, index=False, na_rep='NaN',
            formatters=[f2,f2,f2,f2,f3,f3], float_format=True, sparsify=None,
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
    df = df.sort_values(by=['p'])
    mkTex(df, output_file)

if __name__ == '__main__':
    main()
