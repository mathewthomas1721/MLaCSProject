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
FIG_PATH = "../Figs/"
POST_SEASON_ALL = DATA_PATH +"AtBats_PostSeason_2012-2017_update.csv"
REG_SEASON_ALL = DATA_PATH +"AtBats_RegularSeason_2012-2017_update.csv"
SPRING_TRN_ALL = DATA_PATH + "AtBats_SpringTraining_2012-2017_update.csv"


def train_test_split(seas_df,t=.75):
    num_train = int(np.round(t*seas_df.shape[0]))
    num_test = int(seas_df.shape[0]-num_train)
    
    print("number at bats: \t",end="")
    print(seas_df.shape[0])
    print("number training : \t",end="")
    print(num_train)
    print("number test: \t",end="")
    print(num_test)
    print("total is\t:" +str(num_train+num_test))
    train = seas_df.head(num_train)
    trn_x = train[
    test = seas_df.tail(num_test)
    print(train.tail())
    print(test.head())
    sys.exit(0)

"""
returns a data frame with season 
"""
def season_subset(year):
    pass



"""
Runs majority classifier

"""

def Majority_Classifier(dfAtBats):
    Years=np.arange(2012,2018)
    performance = np.ones(len(Years)) 
    for i in range(len(Years)):
        year = Years[i]
        curr_seas = dfAtBats.loc[dfAtBats['year']==year]
        #print(curr_seas.head())
        Train_X, Test_X,Train_y,Test_y = train_test_split(curr_seas)
        plt.plot([year],[1+i],marker='o')
    
    ax = plt.subplot(111)
    ax.plot(x, y)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)   
    plt.show()

    
"""
    Tries dummy classifiers for all possible strategies

"""
def All_Dummy_Classifiers(AtBatSource):
    pass



def main():

    ### ---- Prepreocess Season Data --- ###
        # gives a column indexing season by year
        # binarizes the outcome of strikeouts to be 1 (yes strikeout) and 0 (not strikeout)
        #TODO: establish unique within-season game ID numbers so we can analyze 'first k games' etc. 
    
    
    dfRegSeason = pd.read_csv(REG_SEASON_ALL)
    dfRegSeason['y'] = np.where(dfRegSeason['descr']=='Strikeout', 1, 0)
    dfRegSeason['year'] = pd.DatetimeIndex(dfRegSeason['date']).year 
    
    ## commented out for runtime
    #dfPostSeason = pd.read_csv(POST_SEASON_ALL)
    #dfSpringTrn = pd.read_csv(SPRING_TRN_ALL)



    ### ---- Majority Classifier --- ###
    #print(dfRegSeason.head(20))
    Majority_Classifier(dfRegSeason)

    ### --- Logistic Regression --- ###


    ### --- Perceptron --- ###
    
        #TODO: decide if we want to use this as a baseline or as a first model in the actual project

if __name__=="__main__":
    main()
