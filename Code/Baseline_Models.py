import numpy as np
import pandas as pd
import scipy as sp
import sys
import matplotlib.pyplot as plt
from tt_split import *
import statsmodels.api as sm
import statsmodels.formula.api as smf
[from sklearn.metrics import roc_auc_score, roc_curve]
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression as LogisticModel
from sklearn.metrics import log_loss, make_scorer
FIG_PATH="../Figs/"
DATA_PATH="../Data/"


def plot_roc(y_true, y_hat,title,fname):
    roc_auc = roc_auc_score(y_true, y_hat)
    fpr, tpr ,thresh=  roc_curve(y_true, y_hat)
    plt.title(title)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(FIG_PATH+fname)
    plt.show()
    plt.close()

def plot_logloss(seasons,array):
    min_x = int(seasons[0])
    max_x=int(seasons[-1])
    plt.scatter(seasons,results)
    plt.title("Log Loss of Gradient Boosting Classifier Across Seasons")
    plt.xlabel("Season")
    plt.xticks(np.arange(min_x,max_x+1),seasons)
    plt.ylabel("Log Loss")
    plt.savefig("../Figs/GBM_results.pdf")
    plt.show()
    plt.close()
    pass

def basic_probability_guessing(y_train,y_test):
    p = np.mean(y_train)
    y_hat = [p]*len(y_test)
    return log_loss(y_test,y_hat), y_hat

def main():

    Seasons = ["2012","2013","2014","2015","2016","2017"]
    Seasons = ["2012"]

    for season in Seasons:
        fname = DATA_PATH+"RegularSeasonFeatures"+str(season)+".csv"
        df = pd.read_csv(fname,index_col=0)
        X_train,X_test,y_train,y_test = train_test_split_season(df)
        ll,y_hat = basic_probability_guessing(y_train,y_test)
        plot_roc(y_test,y_hat,"ROC Curve "+str(season)+" Dummy Classifier \n Log Loss:"+str(ll),"Dummy_ROC_Curve"+str(season)+".pdf")
        lme_file = DATA_PATH+"lme_"+str(season)+".csv"
        y_hat_lme = pd.read_csv(lme_file,index_col=0).as_matrix().reshape(-1,1)
        lme_ll=log_loss(y_test,y_hat_lme)
        plot_roc(y_test,y_hat,"ROC Curve "+str(season)+" LME\n Log Loss:"+str(lme_ll),"LME_ROC_Curve"+str(season)+".pdf")


if __name__=="__main__":
    main()
