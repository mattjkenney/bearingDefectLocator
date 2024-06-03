# summation(function, (interative symbol, starting integer, end integer))
import numpy as np
from pandas import DataFrame

class Features():

    def __init__(
        self,
        dataset):
        
        self.dataset = dataset

    def tools(self):

        dar = self.dataset                              # dar= 0
        n = len(dar)                                   #    n= 1
        x_bar = np.mean(dar)                        #   x_bar= 2
        N = 1 / n                                     #     N= 3
        maxabs = np.max(np.abs(dar))                   #  maxabs= 4

        return dar, n, x_bar, N, maxabs
    
    def skewness(self):

        dar, n, x_bar, N, maxabs = self.tools()
        
        num = N * sum(np.power((dar - x_bar), 3))
        den = np.power(np.sqrt(N * sum(np.power(dar - x_bar, 2))), 3)
        
        return num / den

    def kurtosis(self):

        dar, n, x_bar, N, maxabs = self.tools()
        
        num = N * sum(np.power(dar - x_bar, 4))
        den = np.power(np.sqrt(N * sum(np.power(dar - x_bar, 2))), 4)
        
        return num / den

    def crest(self):

        dar, n, x_bar, N, maxabs = self.tools()
        
        num = maxabs
        den = np.sqrt(N * sum(np.power(dar, 2)))
        
        return num / den

    def shape(self):
        
        dar, n, x_bar, N, maxabs = self.tools()
        
        num = np.sqrt(N * sum(np.power(dar, 2)))
        den = N * sum(abs(dar))
        
        return num / den

    def impulse(self):

        dar, n, x_bar, N, maxabs = self.tools()
        
        num = maxabs
        den = N * sum(abs(dar))
        
        return num / den

    def margin(self):
        
        dar, n, x_bar, N, maxabs = self.tools()
        
        num = maxabs
        den = np.power(N * sum(np.sqrt(abs(dar))), 2)

        return num / den

    def dataframe(self):

        df = DataFrame({
            'name': [self.name],
            'class': [self.class_],
            'skewness': [self.skewness()],
            'kurtosis': [self.kurtosis()],
            'crest': [self.crest()],
            'shape': [self.shape()],
            'impulse': [self.impulse()],
            'margin': [self.margin()]
            })

        return df
