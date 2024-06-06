import numpy as np
import pandas as pd
import original_files.classBearingFeatures2 as bF


class BearingConditionPredictor():

    def __init__(
        self,
        feature= 'Crest',
        x_feature= 'Velocity',
        timespan= 10,
        nPeriods= 10,
        nBins= 60):

        self.feature = feature
        self.x_feature = x_feature
        self.timespan = timespan
        self.nPeriods = nPeriods
        self.nBins = nBins
        # Other parameters set by methods:
        #   bArray
        #   Pcategory
        #   Pperiod_bin
        #   Pperiod_bin_category
        #   charBinsDict
        #   catBinsDict

    def to_bearingFeat(self, files=[], categories= []):
        # files:
        #   list or tuple for the filepaths of the vibration
        #   data files
        # categories:
        #   list or tuple of the category of each bearing in
        #   the same index of the filepath in files

        if len(files) == 0:
            return print('...at least one filepath is needed')
        else:
            feats= []
            for i in range(len(files)):
                feat = bF.BearingFeatures(
                    file_location= files[i],
                    category= categories[i],
                    nPeriods= self.nPeriods,
                    feature= self.feature)
                feats.append(feat)

            return feats
        
    def featsArray(self, feats= []):
        # feats:
        #   list or tuple of BearingFeatures objects
        
        ars = []
        for f in feats:
            nar = f.featuresArray(x_feat= self.x_feature,
                                  timespan_sec= self.timespan)
            ars.append(nar)
            print(f.category)
        far = np.vstack(ars)
        
        return far

    def to_df(self, feats_Array=[]):
        # feats_Array:
        #   np.ndArray returned from the featsArray method
        # this method returns a pandas DataFrame

        if len(feats_Array) == 0:
            return print('...a features Array is required.')
        else:
            cols = ['p' + str(i + 1) for i in range(self.nPeriods)]
            dfCol = pd.DataFrame(feats_Array[:,0], columns= ['category'])
            dfFeats = pd.DataFrame(feats_Array[:,1:], columns= cols,
                                   dtype= float)
            df = dfCol.join(dfFeats, how= 'right')

        return df

    def sample(self, dataframe= None, sampPerc= 0.5):
        # dataframe:
        #   A pandas DataFrame
        # returns two pandas DataFrame: a training set and a test set

        if type(dataframe) != type(pd.DataFrame({})):
            return print('...a pandas DataFrame is required as an argument.')
        else:
            cdfs = []
            for c in dataframe.value_counts('category').index:
                t1df = dataframe[dataframe['category'] == c]
                t1 = t1df.sample(frac= sampPerc)
                cdfs.append(t1)
                
            training = pd.concat(cdfs)
            test = dataframe.drop(training.index, axis= 0)
            
            return training, test

    def add_bins(self, df= None):

        nDict = {}
        colsb = []
        for p in range(self.nPeriods):
            for b in range(len(self.bArray[p])):
                s = 'period' + str(p + 1) + '_bin' + str(b + 1)
                colsb.append(s)

                ns = []
                for c in range(df.shape[0]):
                    val = float(df.iat[c, p + 1])
                    if ((val >= self.bArray[p][b][0]) and \
                        (val < self.bArray[p][b][1])) or \
                       ((self.bArray[p][b][1] == self.bArray[p][-1][1]) and \
                        (val >= self.bArray[p][-1][1])) or \
                       ((self.bArray[p][b][0] == self.bArray[p][0][0]) and \
                        (val <= self.bArray[p][0][0])):
                        ns.append(1)
                    else:
                        ns.append(0)
                nDict[s] = ns
        
        dfcs = pd.DataFrame(nDict, index= df.index)
        dfn = df.join(dfcs)
        
        return dfn

    def train(self, df= None):
        # df:
        #   A pandas DataFrame containing the training set

        df.reset_index(inplace= True, drop= True)
        
        nS = df.shape[0]
        nP = self.nPeriods
        nC = len(df.value_counts('category').index)
        
        # Group the y-values in bins for each period.
        ys = df.drop('category', axis= 1)
        ysar = ys.to_numpy(dtype= float)
        binsL = []
        for p in range(nP):
            props, bins = np.histogram(a= ysar[:,p], bins= self.nBins, density= False)
            binsL.append(bins)
            
        bs = []
        for bp in range(len(binsL)):
            bperiod = binsL[bp]
            ts = []
            for b in range(len(bperiod) - 1):
                t = (bperiod[b], bperiod[b + 1])
                ts.append(t)
            bs.append(ts)
            
        self.bArray = bs
        
        # [period1_bin1, period1_bin2,..., period1_binB, period2_bin1, ..., periodN_binB]   
        dfn = self.add_bins(df)

        # P(category)
        dfCat = dfn.value_counts('category')
        dfCatP = dfCat / nS
        dfCatP.name = 'prob_cat'
        self.Pcategory = dfCatP

        # P(period_n_bin_b) - takes into account that every sample will go into one
        # bin per period.
        dfPB = dfn.sum()
        dfPB = dfPB.filter(dfn.columns[1 + nP: ])
        s = 0
        for i in range(nP):
            nB = len(self.bArray[i])
            e = s + nB
            dfPB[s:e] = (dfPB[s:e] / np.sum(dfPB[s:e]))
            s += nB
        self.Pperiod_bin = dfPB

        # P(period_n_bin_b | category)
        dfFC = dfn.groupby('category').sum()
        dfFC = dfFC.filter(dfn.columns[1 + nP: ])
        s = 0
        for i in range(nP):
            nB = len(self.bArray[i])
            e = s + nB
            for c in range(len(dfFC.index)):
                factor = dfCat[dfCat.index[c]]
                row = dfFC[dfFC.columns[s:e]]
                row = row.filter([dfCat.index[c]], axis=0)
                row = (row / factor)
                dfFC.update(other= row, overwrite= True)
            s += nB

        self.Pperiod_bin_category = dfFC

        # Builds the dictionary charBinsDict:
        #   (key, value) = (category, list of bins where instances are exclusive
        #                   to that category)
        #
        # Build the dictionary catBinsDict:
        #   (key, value) = (category, list of bins where instances are > 0
        #                   to that category)
        charColsDict = {}
        catBinsDict = {}
        for catI in range(len(dfFC.index)):
            charCols = []
            ars = []
            catCols = []
            for tcatI in range(len(dfFC.index)):
                if catI != tcatI:
                    elm = False
                else:
                    elm = True
                tAr = np.array([elm])
                ars.append(tAr)
            arAr = np.column_stack(ars)[0]
            arAr2 = np
            
            for colS in dfFC.columns:
                tCol = dfFC[colS].to_numpy(dtype= float) > 0
                if (tCol == arAr).all():
                    charCols.append(colS)
                if tCol[catI] == True:
                    catCols.append(colS)
                    
            charColsDict[dfFC.index[catI]] = charCols
            catBinsDict[dfFC.index[catI]] = catCols

        self.charBinsDict = charColsDict
        self.catBinsDict = catBinsDict

    def test(self, df= None, categories_known= False):        

        if categories_known:
            # Set P(category) for this test set
            dfVC = df.value_counts('category')
            dfVCP = dfVC / df.shape[0]
            dfVCP.name = 'prob_cat'
            self.Pcategory = dfVCP

        # Add bin data
        df = self.add_bins(df)
        
        nC = len(self.Pcategory.index)

        nCorr = 0
        df = df.drop(df.columns[1:self.nPeriods], axis= 1)
        for r in range(df.shape[0]):
            this_r_bins = []
            cat_tests = []
            for cat_item in list(self.charBinsDict.items()):
                for col in range(df.shape[1] - 1):
                    if df.iat[r, col + 1] == 1:
                        this_r_bins.append(df.columns[col + 1])
                        
                # Removes columns from the test sample's bin list not
                # uniquely characteristic for this category or where
                # probability is 0% for this category, whichever retains
                # the most columns. If the number of columns to remove are
                # equal, the characteristic columns are favored.
                
                to_removeChar = []
                to_removeCat = []
                for b in this_r_bins:
                    if b not in cat_item[1]:
                        to_removeChar.append(b)
                    if b not in self.catBinsDict.get(cat_item[0]):
                        to_removeCat.append(b)
                if len(to_removeChar) <= len(to_removeCat):
                    for toR in to_removeChar:
                        this_r_bins.remove(toR)
                else:
                    for toR in to_removeCat:
                        this_r_bins.remove(toR)
                
                Ppbc = self.Pperiod_bin_category
                Ppbc = Ppbc.filter(this_r_bins, axis= 1)
                numDf = Ppbc.join(self.Pcategory)
                numDf = numDf.filter([cat_item[0]], axis= 0)
                num = numDf.product(1)[0]

                denDf = self.Pperiod_bin
                denDf = denDf[this_r_bins]
                den = denDf.product(0)

                probC = num / den
                    
                cat_tests.append((cat_item[0], probC))
                
            pMax = cat_tests[0]
            for p in range(len(cat_tests)):
                if cat_tests[p][1] > pMax[1]:
                    pMax = cat_tests[p]
            this_cat = pMax[0]
            
            if df.iat[r, 0] == 'unknown':
                df.iat[r, 0] = this_cat
            else:
                if this_cat == df.iat[r, 0]:
                    nCorr += 1
        pCorr = 100 * nCorr / df.shape[0]

        return pCorr, df

    def train_test(self, files= [], cats= [], df= None):
        # Trains and tests the model
        # Required arguments:
        #   files and cats
        #   or
        #   df - a pandas DataFrame
        
        feats = self.to_bearingFeat(files, cats)
        featAr = self.featsArray(feats)
        df = self.to_df(featAr)
        training, testing = self.sample(df)
        self.train(training)
        pCorr, df = self.test(testing, True)

        return pCorr, df
            
                    

            
        
        
        
