"""
Python code to create a few baseline models for analyzing performance



@Authors:
    James Bannon

"""

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression as LogisticModel

DATA_PATH = "../Data/"
FIG_PATH = "../Figs/"
POST_SEASON_ALL = DATA_PATH +"AtBats_PostSeason_2012-2017_update.csv"
REG_SEASON_ALL = DATA_PATH +"AtBats_RegularSeason_2012-2017_update.csv"
SPRING_TRN_ALL = DATA_PATH + "AtBats_SpringTraining_2012-2017_update.csv"


def train_test_split(seas_df):
    pass


"""
returns a data frame with season 
"""
def season_subset(year):
    pass



"""
Runs majority classifier

"""

def Majority_Classifier(AtBatSource):
    Years=np.arange(2012,2018)
    performance = 
    for year in Years:
        df = 

"""
Tries dummy classifiers for all 

"""
def All_Dummy_Classifiers(AtBatSource):
    pass



def main():
    ### ---- Majority Classifier --- ###
    dfRegSeason = pd.read_csv(REG_SEASON_ALL)
    print(dfRegSeason.head())
    Years=np.arange(2012,2018)
    print(Years)


    ### --- Logistic Regression --- ###
    
if __name__=="__main__":
    main()
