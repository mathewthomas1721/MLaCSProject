import numpy as np 
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit,ParameterGrid
from sklearn.metrics import log_loss, make_scorer
from sklearn.preprocessing import StandardScaler
from tt_split import *
import time
from sklearn.externals import joblib
MODEL_PATH = "./Models/"



def cross_validate(X_train, X_val,y_train, y_val,param_grid):
	X_train_val = np.vstack((X_train, X_val))
	y_train_val = np.concatenate((y_train, y_val))
	val_fold = [-1]*len(X_train) + [0]*len(X_val) # 0 corresponds to validation
	estimator_ = RandomForestClassifier()
	grid = GridSearchCV(estimator_,param_grid,return_train_score=True,
		cv = PredefinedSplit(test_fold=val_fold),refit = True,scoring = make_scorer(log_loss,greater_is_better = False))
	grid.fit(X_train_val, y_train_val)
	return grid.best_estimator_, grid.cv_results_['params'][grid.best_index_]


def plot_results(seasons, results):
	min_x = int(seasons[0])
	max_x=int(seasons[-1])

	plt.scatter(seasons,results)
	plt.title("Log Loss of Random Forest Classifier Across Seasons")
	plt.xlabel("Season")
	plt.xticks(np.arange(min_x,max_x+1),seasons)
	ax = plt.subplot(111)

	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	plt.ylabel("Log Loss")
	plt.savefig("../Figs/RanFor_results.pdf")
	plt.show()
	plt.close()

def make_models(Seasons=["2012","2013"],pgrid={'n_estimators':[5,10,50,100],'max_depth':[1,5,7,8]},scale=False):
	print("Making Random Forest Classifier Models.")
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
		joblib.dump(Best_Model,MODEL_PATH+"RanFor_"+str(season)+".pkl") # save the model 
		
		y_hat = Best_Model.predict_proba(X_test)
		results.append(log_loss(y_test,y_hat))
	plot_results(Seasons, results)

def main():


	pgrid = {'n_estimators':[5,10,50,75,100],'min_samples_split':[2,20], 'max_depth':[5,6,7,8]}
	seasons = ["2012","2013","2014","2015","2016","2017"]
	make_models(Seasons=seasons,pgrid=pgrid)
	sys.exit(0)

	""" unit tests """
	clf = RandomForestClassifier(n_estimators=50,max_depth = 7)
	df = pd.read_csv("../Data/RegularSeasonFeatures2012.csv",index_col=0)
	X_train, X_test, y_train, y_test = train_test_split_season(df) 
	
	clf.fit(X_train,y_train)
	y_hat = clf.predict_proba(X_test)
	print(y_hat)
	print(log_loss(y_test,y_hat))

	pass

if __name__ == '__main__':
	main()
