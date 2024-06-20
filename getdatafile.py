import boto3
import os
import pickle as pk
from scipy.io import loadmat
from original_files.classBearingFeatures2 import BearingFeatures as BF
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

def get_dataframe_from_label(label, qty=12):

    with open(os.path.join('datafiles', 'vibs.pk'), 'rb') as filehandle:
        keyDict = pk.load(filehandle)
    keys = keyDict.get(label)

    labelDict= {"H": 'healthy', "I": 'inner race', "O": 'outer race', "B": 'ball', "C": "combination"}

    assert len(keys) >= qty # qty value cannot exceed the number of total objects

    client = boto3.client('s3')
    dfs = []
    temp_file = 'temp_file.mat'
    for i in range(qty):
        objkey = keys[i]
        with open(temp_file, 'wb') as data:
            client.download_fileobj('bearingvibrations', objkey, data)
        data.close()
        matfile = loadmat(temp_file)
        dfn = pd.DataFrame()
        dfn['vibration velocity'] = tuple(np.reshape(matfile.get('Channel_1'), (2000000,)))
        dfn['shaft speed'] = tuple(np.reshape(matfile.get('Channel_2'), (2000000,)))
        dfn['label'] = tuple([labelDict.get(objkey[-9]) for i in range(len(dfn['vibration velocity']))])
        dfn['instance'] = [objkey[-9:-4] for _ in range(len(dfn['vibration velocity']))]
        columnsOrder=['instance', 'label', 'shaft speed', 'vibration velocity']
        dfn = dfn.reindex(columns= columnsOrder)
        dfs.append(dfn)
    dfm = pd.concat(dfs, ignore_index=True)
    os.remove(temp_file)
    
    return dfm

def get_keys_file(bucket_name= 'bearingvibrations'):

    keyDict = {'healthy': [], 'inner race': [], 'outer race': [], 'ball': [], 'combination': []}
    s3 = boto3.resource('s3')
    for obj in s3.Bucket(bucket_name).objects.all():
        for k in keyDict.keys():
            if k in obj.key:
                keyDict[k].append(obj.key)

    file = open(os.path.join('datafiles','vibs.pk'), 'wb')
    pk.dump(keyDict, file)
    file.close()

    return

def test():
    return

if __name__ == "__main__":
    test()
