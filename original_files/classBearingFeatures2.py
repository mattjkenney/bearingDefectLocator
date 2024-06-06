from scipy import io
import numpy as np
import pandas as pd
import scipy.stats as st
from original_files.featureFuncs import Features as F

class BearingFeatures():

    def __init__(
        self,
        file_location= None,
        xy_array= None,
        category= 'unknown',
        nPeriods= 20,
        feature = 'Kurtosis'
        ):

        self.file_location = file_location
        self.xy_array = xy_array
        self.category = category
        self.nPeriods = nPeriods
        self.feature = feature

    def get_xy(self, file_location= None, timespan_sec= 10, shaft_speed=True):

        try:
            matfile = io.loadmat(file_location)
        except:
            file_location = self.file_location
            matfile = io.loadmat(file_location)
        else:
            self.file_location = file_location
            
        y = matfile.get('Channel_1')
        x = matfile.get('Channel_2')
        ar = np.column_stack((x, y))
        if shaft_speed == True:
            ar[:,0] = ar[:,0] * (ar.shape[0] / timespan_sec) * 2 / 1024

        return ar

    def featuresArray(self, ar= [], x_feat= 'Velocity', timespan_sec= 10): 

        if len(ar) == 0:
            ar = self.get_xy()
        
        if x_feat == 'Acceleration':
            ar2 = ar[:,0]
            ar3 = np.hstack((np.array([0]), ar2))
            ar4 = np.hstack((ar2, np.array([0])))
            accels = ar4 - ar3
            accels[0] = 0
            accels = np.delete(accels, -1)
            time_int = timespan_sec / accels.shape[0]
            accels /= time_int
            ar[:,0] = accels
            
        df = pd.DataFrame(ar, columns=['x','y'])
        df = df.sort_values('x')
        df.reset_index(inplace=True, drop=True)
        ar = df.to_numpy()

        feat_dict = {
            'Skewness': lambda tF: tF.skewness(),
            'Kurtosis': lambda tF: tF.kurtosis(),
            'Crest': lambda tF: tF.crest(),
            'Shape': lambda tF: tF.shape(),
            'Impulse': lambda tF: tF.impulse(),
            'Margin': lambda tF: tF.margin(),
            'Mean': lambda tF: tF.mean()}
        
        nTotal = df.shape[0]
        chunk = int(nTotal / self.nPeriods)
        ys = []
        for p in range(self.nPeriods):
            s = p * chunk
            nar = ar[s:s + chunk,1]
            tF = F(nar)
            yf = feat_dict.get(self.feature)(tF)
            ys.append(yf)
        ys.insert(0, self.category)
        ar = np.array(ys)
       

        return ar
            
        
        
