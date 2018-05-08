import numpy as np 
import pandas as pd
import pickle
from sklearn.ensemble import AdaBoostClassifier

import time
import sys
from sklearn.model_selection import GridSearchCV, PredefinedSplit,ParameterGrid
from sklearn.metrics import mean_squared_error, make_scorer,log_loss
from sklearn.model_selection import GridSearchCV, PredefinedSplit,ParameterGrid
from sklearn.metrics import log_loss, make_scorer
from sklearn.preprocessing import StandardScaler
from tt_split import *
import time
import matplotlib.pyplot as plt
from sklearn.externals import joblib


MODEL_PATH = "./Models/"
""" zooms in on cross validation """


def cross_validate(X_train, X_val,y_train, y_val,param_grid):
	X_train_val = np.vstack((X_train, X_val))
	y_train_val = np.concatenate((y_train, y_val))
	val_fold = [-1]*len(X_train) + [0]*len(X_val) # 0 corresponds to validation
	estimator_ = AdaBoostClassifier()
	grid = GridSearchCV(estimator_,param_grid,return_train_score=True,
		cv = PredefinedSplit(test_fold=val_fold),refit = True,scoring = make_scorer(log_loss,greater_is_better = False))
	grid.fit(X_train_val, y_train_val)
	return grid.best_estimator_, grid.cv_results_['params'][grid.best_index_]


def plot_results(seasons, results):
	min_x = int(seasons[0])
	max_x=int(seasons[-1])
	plt.scatter(seasons,results)
	plt.title("Log Loss of Adaboost Classifier Across Seasons")
	plt.xlabel("Season")
	plt.xticks(np.arange(min_x,max_x+1),seasons)
	plt.ylabel("Log Loss")
	plt.savefig("../Figs/AdaBoost_results.pdf")
	plt.show()
	plt.close()

def make_models(Seasons=["2012","2013"],pgrid = {'learning_rate':[.01, .1],'n_estimators': [3,5],'max_leaf_nodes':[4]},scale=False):
	print("Making Adaboost Classifier Models.")
	results = []
	for season in Seasons:
		""" cross validate to find parameters"""
		df = pd.read_csv("../Data/RegularSeasonFeatures"+str(season)+".csv",index_col=0)
		X_train, X_val, X_test, y_train, y_val, y_test = train_test_split_season(df,validation=True)
		Best_Model, ParamDict = cross_validate(X_train,X_val, y_train, y_val,pgrid)
		
		print(" In year  "+str(season)+" the best params were:")
		print(ParamDict)

		X_train, X_test, y_train, y_test = train_test_split_season(df) 
		
		Best_Model.fit(X_train,y_train)
		joblib.dump(Best_Model,MODEL_PATH+"AdaBoost_"+str(season)+".pkl") # save the model 
		
		y_hat = Best_Model.predict_proba(X_test)
		results.append(log_loss(y_test,y_hat))
	plot_results(Seasons, results)

def main():
	Seasons = ["2012","2013"]#,"2014","2015","2016","2017"]
	pgrid_simple = {'n_estimators':np.arange(5,51,10),'learning_rate':[0.1, 0.5, 1]}

	make_models(Seasons,pgrid_simple)
	""" below are unit tests that can be run by commenting out the sys.exit(0) line below """

	sys.exit(0)
	for season in Seasons:
		file_name = "../Data/RegularSeasonFeatures"+season+".csv"
		print(file_name)
		df = pd.read_csv("../Data/RegularSeasonFeatures2012.csv",index_col=0)

		X_train, X_val, X_test, y_train, y_val, y_test = train_test_split_season(df,validation=True)
		s= time.time()
	
		
		
		BestAdaBoostEstimator,ParamDict=cross_validate(X_train, X_val, y_train, y_val,pgrid_simple)
		e=time.time()
		print("cross validating took "+ str( (e-s)/60) + " minutes.")
		X_train, X_test, y_train, y_test = train_test_split_season(df)
		BestAdaBoostEstimator.fit(X_train,y_train)
		y_hat = BestAdaBoostEstimator.predict_proba(X_test)
		print(log_loss(y_test,y_hat))
		print(ParamDict)
		sys.exit(0)

if __name__ == '__main__':
	main()