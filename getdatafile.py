import os
import pickle as pk
from scipy.io import loadmat
from classBearingFeatures2 import BearingFeatures as BF
import numpy as np
import pandas as pd

def get_dataframe_subset_for_sample(master_dataframe: pd.DataFrame, periods, feat, domain):

    arr = master_dataframe.to_numpy()
    bf = BF(xy_array= arr, category='healthy', nPeriods= periods, feature= feat)
    farr = bf.featuresArray(ar= bf.xy_array, x_feat= domain)
    farr = farr.transpose()
    df = pd.DataFrame(farr, columns= [feat.lower() + ' vibration velocity'])
    df['period'] = [i + 1 for i in range(df.shape[0])]

    return df

def get_dataframe_from_label():

    filepath = os.path.join("datafiles", "H-A-1.mat")
    matfile: dict = loadmat(filepath)
    dfn = pd.DataFrame()
    dfn['vibration velocity'] = tuple(np.reshape(matfile.get('Channel_1'), (2000000,)))
    dfn['shaft speed'] = tuple(np.reshape(matfile.get('Channel_2'), (2000000,)))
    dfn['label'] = tuple('healthy' for i in range(len(dfn['vibration velocity'])))
    dfn['instance'] = ['H-A-1' for _ in range(len(dfn['vibration velocity']))]
    columnsOrder=['instance', 'label', 'shaft speed', 'vibration velocity']
    dfn = dfn.reindex(columns= columnsOrder)
    
    return dfn

def test():
    return

if __name__ == "__main__":
    test()
