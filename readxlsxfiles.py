import pandas as pd
import openpyxl as xl
import os

def get_exp2_tables_df(FD):
    
    df: pd.DataFrame = pd.read_excel(os.path.join('datafiles','results.xlsx'), sheet_name=FD, header=0)
    df = df.set_index('Number of Periods')
    df.columns = pd.MultiIndex.from_product([['Number of Bins'], df.columns])
    df = df.round(0)
    df = df.astype(int)
    
    return df

def get_exp2_df():

    df = pd.read_excel(os.path.join('datafiles','results.xlsx'), sheet_name="Sheet1", header=0, usecols=[1,2,3,4])

    return df

def get_boxplot_df():

    df = pd.read_excel(os.path.join('datafiles','exp1_boxplot.xlsx'), sheet_name="vVelTab", header=0)
    
    return df

def get_feature_domain_20p(feature_domain):

    df: pd.DataFrame = pd.read_excel(os.path.join('datafiles','featuresLineCharts.xlsx'), sheet_name=feature_domain, header=0)
    #df = df.set_index('category', drop=True)
    df = df.transpose()
    map = {}
    for i in df.columns:
        map[i] = df.iat[0,i]
    df = df.rename(columns=map)
    df = df.drop(['category'], axis=0)
    df = df.reset_index(drop=True)
    df['period'] = [i + 1 for i in range(df.shape[0])]
    colOrder = ['period', 'healthy', 'inner race fault', 'outer race fault', 'ball fault', 'combination fault']
    df = df[colOrder]

    return df

def get_all_feature_domains():

    # creates dictionary: dom = {sheet name: (feature, domain)}
    wb = xl.load_workbook(os.path.join('datafiles','featuresLineCharts.xlsx'))
    sheets = wb.worksheets
    fds = {}
    for n in sheets[1:]:
        title = n.title
        if 'Velocity' in title:
            dom = 'Velocity'
        else:
            dom = 'Acceleration'
        fds[title] = (title[:title.index(dom)], dom)

    return fds

def get_exp1():

    df = pd.read_excel(os.path.join('datafiles','exp1.xlsx'), header=0)
    
    return df

def test():
    from pandas import DataFrame
    assert type(get_exp2_tables_df('CA')) == type(DataFrame())
    assert type(get_exp2_df()) == type(DataFrame())
    assert type(get_boxplot_df()) == type(DataFrame())
    assert type(get_feature_domain_20p('KurtosisAcceleration')) ==type(DataFrame())
    assert type(get_all_feature_domains()) == dict
    assert type(get_exp1()) == type(DataFrame())

    print('...test successfull!')

if __name__ == "__main__":
    test()




