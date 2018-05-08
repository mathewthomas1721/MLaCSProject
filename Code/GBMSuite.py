import numpy as np 
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import sys
from sklearn.ensemble import GradientBoostingClassifier
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
	estimator_ = GradientBoostingClassifier()
	grid = GridSearchCV(estimator_,param_grid,return_train_score=True,
		cv = PredefinedSplit(test_fold=val_fold),refit = True,scoring = make_scorer(log_loss,greater_is_better = False))
	grid.fit(X_train_val, y_train_val)
	return grid.best_estimator_, grid.cv_results_['params'][grid.best_index_]


def plot_results(seasons, results):
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

def make_models(Seasons=["2012","2013"],pgrid = {'learning_rate':[.01, .1],'n_estimators': [3,5],'max_leaf_nodes':[4]},scale=False):
	print("Making Gradient Boosted Classifier Models.")
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
		joblib.dump(Best_Model,MODEL_PATH+"GBM_"+str(season)+".pkl") # save the model 
		
		y_hat = Best_Model.predict_proba(X_test)
		results.append(log_loss(y_test,y_hat))
	plot_results(Seasons, results)


def plot_split_results(seasons, qual, real):
	""" To Be Completed"""
	plt.scatter(seasons,real, color='red',label='real')
	plt.scatter(seasons,qual,color='blue',label='qual')
	plt.show()
	pass

def make_split_models(Seasons=["2012","2013"],pgrid = {'learning_rate':[.1],'n_estimators': [3,5],'max_leaf_nodes':[3,4]},scale=False):
	""" take in a list of seasons and spit out models per season.
			one trained on qualitative/one-hot features
			one trained on real features

		STILL TO COMPLETE
			needs robust way of getting scalar vs non-scalar columns.
	  """
	print("Making Split Gradient Boosted Classifier Models.")
	qual_results = []
	real_results = []
	for season in Seasons:
		df = pd.read_csv("../Data/RegularSeasonFeatures"+str(season)+".csv",index_col=0)
		X_train, X_val, X_test, y_train, y_val, y_test = train_test_split_season(df,validation=True)
		
		train_quals = X_train[:,14:77]
		train_reals = X_train[:,list(np.arange(14)) +[77]]


		val_quals = X_val[:,14:77]
		val_reals = X_val[:,list(np.arange(14)) +[77]]

		
		if scale:
			scaler=StandardScaler().fit(train_reals)
			train_reals=scaler.transform(train_reals)
			val_reals = scaler.transform(val_reals)
		Best_Model_qual, ParamDict_qual = cross_validate(train_quals,val_quals, y_train, y_val,pgrid)
		Best_Model_real, ParamDict_real = cross_validate(train_reals,val_reals, y_train, y_val,pgrid)


		X_train, X_test, y_train, y_test = train_test_split_season(df)
		
		train_quals = X_train[:,14:77]
		train_reals = X_train[:,list(np.arange(14)) +[77]]


		test_quals = X_test[:,14:77]
		test_reals = X_test[:,list(np.arange(14)) +[77]]
		
		if scale:
			scaler=StandardScaler().fit(train_reals)
			train_reals=scaler.transform(train_reals)
			test_reals = scaler.transform(test_reals)
		

		Best_Model_qual.fit(train_quals,y_train)
		Best_Model_real.fit(train_reals,y_train)

		joblib.dump(Best_Model_qual,MODEL_PATH+"GBM_"+str(season)+"_qual.pkl") # save the models
		joblib.dump(Best_Model_real,MODEL_PATH+"GBM_"+str(season)+"_qual.pkl")
		y_hat_quals = Best_Model_qual.predict_proba(test_quals)
		y_hat_reals = Best_Model_real.predict_proba(test_reals)
		qual_results.append(log_loss(y_test,y_hat_quals))
		real_results.append(log_loss(y_test,y_hat_quals))
	plot_split_results(Seasons,qual_results,real_results)

def main():
	pgrid = {'learning_rate':[.01, .1,1],
	'n_estimators': np.arange(5,51,5),
	'max_leaf_nodes':np.arange(2,9)}
	
	pgrid = {'learning_rate':[.1,1],
	'n_estimators': [5,20,50],
	'max_leaf_nodes':[4,6,8]}

	#make_split_models()
	make_models(Seasons=["2012","2013","2014","2015","2016","2017"],pgrid=pgrid)
	
	
	""" Below are unit tests that can be run by commenting out the sys.exit(0) line"""
	
	sys.exit(0)

	df = pd.read_csv("../Data/RegularSeasonFeatures2012.csv",index_col=0)
	X_train, X_val, X_test, y_train, y_val, y_test = train_test_split_season(df,validation=True)
	pgrid = {'learning_rate':[.1,1],'n_estimators': [3,10,20],'max_leaf_nodes':[4,5,6]}
	s=time.time()
	BestGBM, ParamDict = cross_validate(X_train, X_val, y_train, y_val,pgrid)
	e=time.time()
	print("took "+str( (e-s)/60) + " minutes to cross_validate ")
	X_train, X_test, y_train, y_test = train_test_split_season(df)
	BestGBM.fit(X_train,y_train)
	y_hat = BestGBM.predict_proba(X_test)
	print(log_loss(y_test,y_hat))
	



if __name__ == '__main__':
	main()