import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
import numpy as np 
import pandas as pd
import pickle
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, RandomForestRegressor, DecisionTreeClassifier
from tt_split import *
from sklearn.model_selection import GridSearchCV, PredefinedSplit,RandomizedSearchCV, ParameterGrid
from sklearn.metrics import mean_squared_error, make_scorer,log_loss
from sklearn.feature_selection import SelectKBest,RFE
FIG_PATH="../Figs/"
DATA_PATH="../Data/"

def plot_roc(y_true, y_hats,title,fname):

	k_probs = np.zeros((len(y_true),len(y_hats)))
	roc_auc=[]
	
	for i in range(len(y_hats)):
		k_probs[:,i]=y_hats[i][:,1]
		roc_auc.append(roc_auc_score(y_true, k_probs[:,i]))
	plt.figure()
	colors = ['aqua', 'darkorange', 'cornflowerblue']
	classes = ['numeric','categorical','full']
	for k in range(len(y_hats)):
		fpr, tpr,thresh = roc_curve(y_true,k_probs[:,k])
		plt.plot(fpr, tpr, color=colors[k],label='{0} (area = {1:0.2f})'.format(classes[k], roc_auc[k]))
	plt.title(title)
	plt.legend(loc = 'lower right')
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.savefig(FIG_PATH+fname)
	#plt.show()
	plt.close()


def test_RF(year="2012"):
