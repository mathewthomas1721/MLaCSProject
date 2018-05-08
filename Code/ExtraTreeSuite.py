import numpy as np 
import pandas as pd
import pickle
#import matplotlib.pyplot as plt
import sys
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit,ParameterGrid
from sklearn.metrics import log_loss, make_scorer
from tt_split import *
import time


MODEL_PATH = "./Models/"

def cross_validate(X_train, X_val,y_train, y_val,param_grid):
	X_train_val = np.vstack((X_train, X_val))
	y_train_val = np.concatenate((y_train, y_val))
	val_fold = [-1]*len(X_train) + [0]*len(X_val) # 0 corresponds to validation
	estimator_ = ExtraTreesClassifier()
	grid = GridSearchCV(estimator_,param_grid,return_train_score=True,
		cv = PredefinedSplit(test_fold=val_fold),refit = True,scoring = make_scorer(log_loss,greater_is_better = False))
	grid.fit(X_train_val, y_train_val)
	return grid.best_estimator_, grid.cv_results_['params'][grid.best_index_]



def make_models(Seasons):
	pass

Seasons = ["2012","2013","2014","2015","2016","2017"]
for season in Seasons:
	file_name = "../Data/RegularSeasonFeatures"+season+".csv"
	model_out = MODEL_PATH +str(season)+"/"
	print(model_out)
clf = ExtraTreesClassifier()