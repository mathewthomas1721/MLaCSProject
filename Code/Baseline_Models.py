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
POST_SEASON_ALL = DATA_PATH +"AtBats_PostSeason_2012-2017_update.csv"
REG_SEASON_ALL = DATA_PATH +"AtBats_RegularSeason_2012-2017_update.csv"
SPRING_TRN_ALL = DATA_PATH + "AtBats_SpringTraining_2012-2017_update.csv"


def train_test_split(seas_df,t=.75,mode="dummy"):
   
    if mode=="dummy":
        num_train = int(np.round(t*seas_df.shape[0]))
        num_test = int(seas_df.shape[0]-num_train)
    
        train = seas_df.head(num_train)
        test = seas_df.tail(num_test)

    
        trn_x = np.array(train['inning']).reshape((num_train,1))
        trn_y = np.array(train['y'])
    
        tst_x = np.array(test['inning']).reshape((num_test,1))
        tst_y = np.array(test['y'])
        ## need to reshape so sklearn will work with it, otherwise throws a value error later
        return trn_x,tst_x,trn_y,tst_y
    else:
         ##TODO: alter to return useful info for logistic regression, SVC
        return 0

"""
returns a data frame with season 
"""
def season_subset(year):
    pass



def display_results(Years,performance,name):
    print("Year\t Score")
    of = open(WRITE_PATH+name+".txt","w")
    of.write("Year\t Score\n")
    for j in range(len(Years)):
        print(str(Years[j])+" \t "+ str(performance[j]))
        of.write(str(Years[j])+" \t "+ str(performance[j]))
    print("average score:" + str(np.mean(performance)))
    of.write("Average Score:\t"+ str(np.mean(performance)))
    of.write("\n")
    of.write("Score is percentage correct")
    of.close()
        

"""
Runs majority classifier

"""

def Majority_Classifier(dfAtBats):
    Years=np.arange(2012,2018)
    performance = np.ones(len(Years)) 
    for i in range(len(Years)):
        year = Years[i]
        curr_seas = dfAtBats.loc[dfAtBats['year']==year]
        X_train,X_test,y_train,y_test = train_test_split(curr_seas)
        clf = DummyClassifier(strategy='most_frequent',random_state=0)
        clf.fit(X_train,y_train)
        score=clf.score(X_train,y_train)
        plt.plot([year],[score],marker='o')
        performance[i]=score
    display_results(Years,performance,name="Majority_Classifier_Results")
    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.title("Percent Correct by Majority Classifier By Year")
    plt.xlabel("Year")
    plt.ylabel("Percent Correct")
    plt.savefig(FIG_PATH+"Majority_Classifier.pdf")
    plt.close()

    
"""
    Tries dummy classifiers for all possible strategies

"""
def All_Dummy_Classifiers(AtBatSource):
    strategies=["stratified","most_frequent","prior","uniform","constant"]
    pass



def main():

    ### ---- Prepreocess Season Data --- ###
        # gives a column indexing season by year
        # binarizes the outcome of strikeouts to be 1 (yes strikeout) and 0 (not strikeout)
        #TODO: establish unique within-season game ID numbers so we can analyze 'first k games' etc. 
             # will also make adding in pitchfx data easier
    
    
    dfRegSeason = pd.read_csv(REG_SEASON_ALL)
    dfRegSeason['y'] = np.where(dfRegSeason['descr']=='Strikeout', 1, 0)
    dfRegSeason['year'] = pd.DatetimeIndex(dfRegSeason['date']).year 
    
    ## commented out for runtime, should be pre-processed

    #dfPostSeason = pd.read_csv(POST_SEASON_ALL)
    #dfSpringTrn = pd.read_csv(SPRING_TRN_ALL)
    


    ### ---- Majority Classifier --- ###
    
    
    Majority_Classifier(dfRegSeason)


    All_Dummy_Classifiers(dfRegSeason)
    ### --- Logistic Regression --- ###
    
    ##TODO: write this code


    ### --- Perceptron --- ### 
    #TODO: decide if we want to use this as a baseline or as a first model in the actual project
    

    ### --- Basic Support Vector Classifier --- ###
    #TODO: decide if we want to use this as a baseline or as a first model in the actual project

if __name__=="__main__":
    main()
