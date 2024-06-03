import boto3
import os
import pickle as pk
from scipy.io import loadmat
import original_files.classBearingFeatures2 as BF

os.environ['AWS_SHARED_CREDENTIALS_FILE'] = r'../.aws/bearingDefectLocator/credentials'
os.environ['AWS_CONFIG_FILE'] = r'../.aws/bearingDefectLocator/config'

def get_data_from_label(label):

    with open('vibs.pk', 'rb') as filehandle:
        keyDict = pk.load(filehandle)
    keys = keyDict.get(label)

    client = boto3.client('s3')
    arrs = []
    for objkey in keys:
        object = client.get_object(Bucket= 'bearingvibrations', key= objkey)
        bf = BF(file_location= object)
        ar = bf.get_xy()
        arrs.append(ar)
    
    return arrs

def get_keys_file(bucket_name= 'bearingvibrations'):

    keyDict = {'healthy': [], 'inner race': [], 'outer race': [], 'ball': [], 'combination': []}
    s3 = boto3.resource('s3')
    for obj in s3.Bucket(bucket_name).objects.all():
        for k in keyDict.keys():
            if k in obj.key:
                keyDict[k].append(obj.key)

    file = open('vibs.pk', 'wb')
    pk.dump(keyDict, file)
    file.close()

    return

if __name__ == "__main__":
    arrs = get_data_from_label(label='healthy')
    print(len(arrs))