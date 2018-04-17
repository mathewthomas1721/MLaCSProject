"""
Python code to create a few baseline models for analyzing performance



@Authors:
    James Bannon

"""

import numpy as np
import pandas as pd
import scipy as sp
import sys
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression as LogisticModel

DATA_PATH = "../Data/"
WRITE_PATH= "../Data/OutData/"
FIG_PATH = "../Figs/"
POST_SEASON_ALL = DATA_PATH + "AtBats_PostSeason_2012-2017_sorted.csv"
REG_SEASON_ALL = DATA_PATH + "AtBats_RegularSeason_2012-2017_sorted.csv"
SPRING_TRN_ALL = DATA_PATH + "AtBats_SpringTraining_2012-2017_sorted.csv"



"""
returns a data frame with season 
"""
def season_subset(df_all, year):
    df_year = df_all[df_all.year == year]
    return df_year



  

"""
Runs majority classifier

"""


#return the strikeout rate of a pitcher over the last [window_size] at bats
def Sliding_Window_Pitcher(seas_df, row, window_size):
    season_to_date = seas_df[seas_df.index < row['index']]
    pitcher_data = season_to_date[season_to_date.pitcher == row['pitcher']]
    pitcher_window = pitcher_data[-window_size:]
    num_bf = max(pitcher_window.shape[0], 1)
    return pitcher_window['y'].sum() / num_bf


#return the strikeout rate of a batter over the last [window_size] at bats
def Sliding_Window_Batter(seas_df, row, window_size):
    season_to_date = seas_df[seas_df.index < row['index']]
    batter_data = season_to_date[season_to_date.batter == row['batter']]
    batter_window = batter_data[-window_size:]

    num_bf = max(batter_window.shape[0], 1)
    return batter_window['y'].sum() / num_bf


#return the strikeout rate of a pitcher over a set of atbats
def Whole_Set_Pitcher(train_df, row):
    pitcher_data = train_df[train_df.pitcher == row['pitcher']]

    num_bf = max(pitcher_data.shape[0], 1)
    return pitcher_data['y'].sum() / num_bf

#return the strikeout rate of a batter over a set of atbats
def Whole_Set_Batter(train_df, row):
    batter_data = train_df[train_df.batter == row['batter']]

    num_bf = max(batter_data.shape[0], 1)
    return batter_data['y'].sum() / num_bf    


def main():

    ### ---- Prepreocess Season Data --- ###
        # gives a column indexing season by year
        # binarizes the outcome of strikeouts to be 1 (yes strikeout) and 0 (not strikeout)
        #TODO: establish unique within-season game ID numbers so we can analyze 'first k games' etc. 
             # will also make adding in pitchfx data easier
    
    
    dfRegSeason = pd.read_csv(REG_SEASON_ALL)
    dfRegSeason['y'] = np.where(dfRegSeason['descr']=='Strikeout', 1, 0)
    dfRegSeason['year'] = pd.DatetimeIndex(dfRegSeason['date']).year 
    """
    dfRegSeason = dfRegSeason.sort_values(by=["date"],kind='mergesort')
    dfRegSeason.to_csv(DATA_PATH + "AtBats_RegularSeason_2012-2017_sorted.csv")
    dfPostSeason = pd.read_csv(POST_SEASON_ALL)
    dfPostSeason = dfPostSeason.sort_values(by=["date"],kind='mergesort')
    dfPostSeason.to_csv(DATA_PATH + "AtBats_PostSeason_2012-2017_sorted.csv")
    dfSpring = pd.read_csv(SPRING_TRN_ALL)
    dfSpring = dfSpring.sort_values(by=["date"],kind='mergesort')
    dfSpring.to_csv(DATA_PATH + "AtBats_SpringTraining_2012-2017_sorted.csv")
    """

    df_reg_2016 = season_subset(dfRegSeason,2016).head(15000) #Smaller set to make it run in time
    df_reg_2016['index'] = df_reg_2016.index #Adding an index column, which I needed to do for some reason


    num_train = int(np.round(.75*df_reg_2016.shape[0]))
    num_test = int(df_reg_2016.shape[0]-num_train)
    
    #Split test and train data
    train = df_reg_2016.head(num_train)
    test = df_reg_2016.tail(num_test)

    train['pkrate'] = train.apply(lambda row : Sliding_Window_Pitcher(train, row, 70),axis=1)
    train['bkrate'] = train.apply(lambda row : Sliding_Window_Batter(train, row, 70),axis=1)

    #remove beginningg of train data because those atbats don't have enough previous atbats for good features
    train = train.tail(int(np.round(.8*train.shape[0])))

    test['pkrate'] = test.apply(lambda row : Whole_Set_Pitcher(train, row),axis=1)
    test['bkrate'] = test.apply(lambda row : Whole_Set_Batter(train, row),axis=1)
    #with old version of sliding window. I think this would have been faster, but couldn't get to work
    #df_reg_2016['bkrate'] = Sliding_Window_Batter(df_reg_2016, df_reg_2016['batter'], df_reg_2016['index'], 80)

    #Turn train and test data into arrays that can be passed to the model
    trn_x = np.array(train[['pkrate','bkrate']]).reshape((train.shape[0],2))
    trn_y = np.array(train['y'])
    
    tst_x = np.array(test[['pkrate','bkrate']]).reshape((test.shape[0],2))
    tst_y = np.array(test['y'])

    lrm = LogisticModel(C=1000.0)

    lrm.fit(trn_x, trn_y)

    preds = lrm.predict_log_proba(tst_x)
    print (preds)

    score = sum([pred[0] * (1.0 - tst_y[ind]) + pred[1] * (tst_y[ind])   for ind, pred in enumerate(preds)])
    print(-score/tst_y.shape[0])

    



if __name__=="__main__":
    main()
