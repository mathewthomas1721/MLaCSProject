import numpy as np 
import pandas as pd
import pickle
#import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from boostedMLP import *
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from tt_split import *
import time
import sys


def pseudo_res_func(y_train, y_pred):
	return y_train-y_pred

def pseudo_res_log(y_train, y_pred):
	pseudo_res = np.multiply(1.0/(1+np.exp(y_pred)),np.multiply(y_pred,np.exp(y_pred)))	
	pseudo_res = y_train - pseudo_res
	return pseudo_res

def main():
	df = pd.read_csv("../Data/RegularSeasonFeatures2012.csv",index_col=0)
	
	X_train, X_val, X_test, y_train, y_val, y_test = train_test_split_season(df,validation=True)
	



	print("testing a boosted classifier")
	ensemble =[]
	learn_rate = 0.2
	n_estimators=20
	print("non-adaptive learn rate")
	print("num estimators = "+str(n_estimators))
	for i in range(n_estimators):
		print("i="+str(i+1))
		MLR = MLPRegressor(hidden_layer_sizes=(10,20))
		new_res = 0
		if len(ensemble)>0:
			for est in ensemble:
				new_res += learn_rate*est.predict(X_train)
		#print(new_res)
		MLR.fit(X_train,pseudo_res_log(y_train,new_res))
		ensemble.append(MLR)
	preds = np.array([est.predict(X_val) for est in ensemble])
	weights = learn_rate*np.ones(len(ensemble))#/(np.arange(len(ensemble))+1)
	preds = np.dot(preds.T,weights)
	print(X_val.shape)
	print(preds.shape)
	print(log_loss(y_val, 1.0/(1+np.exp(-preds))))
	print(log_loss(y_val, preds))
	


	
	




if __name__ == '__main__':
	main()