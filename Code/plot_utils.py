import numpy as np 
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
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