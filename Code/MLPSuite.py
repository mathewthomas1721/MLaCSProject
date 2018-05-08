import numpy as np 
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import sys
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from tt_split import *
import time


FILE_NAMES = ["../Data/RegularSeasonFeatures2012.csv","../Data/RegularSeasonFeatures2013.csv",
"../Data/RegularSeasonFeatures2014.csv","../Data/RegularSeasonFeatures2015.csv",
"../Data/RegularSeasonFeatures2016.csv","../Data/RegularSeasonFeatures2017.csv"]


def cross_validate(X_train, X_val, y_train, y_val, params):
	
	activations = ['identity','relu','tanh','logistic']
	architectures = [(10,20), (10,10), (10,10,10), (10,20,10,30), (2,2,10,2)]
	for i in range(len(architectures)):
		arch_id = i;
		for act in activations:
			res = []
			count = 0
			for a in params:
				#print(count+1)
				MLP = MLPClassifier(hidden_layer_sizes=architectures[i],activation=act,alpha=a,max_iter=500)
				s = time.time()
				MLP.fit(X_train, y_train)
				e = time.time()
				#print("took " + str((e-s)/60) + " seconds to fit.")
				y_hat=MLP.predict_proba(X_val)
				loss = log_loss(y_val,y_hat)
				res.append(loss)
				#count = count+1
			plt.plot(params, res)
			plt.title("MLP Regularization Tuning\n "+str(act)+" activation\n"+"architecture "+
				str(architectures[i]))
			plt.ylabel("log loss")
			plt.xlabel(r"$\alpha$")
			plt.savefig("../Figs/MLP_crossval_"+str(act)+"_"+str(i)+".pdf")
			plt.close()


def main():
	df = pd.read_csv("../Data/RegularSeasonFeatures2012.csv",index_col=0)
	print(df.columns)

	X_train, X_val, X_test, y_train, y_val, y_test = train_test_split_season(df,validation=True)
	
	first_col = df.columns.get_loc('pitcherkrate')
	last_col = df.columns.get_loc('wind5bkrate')
	scaler = StandardScaler()
	A = X_train[:,first_col:last_col]
	scaler.fit(A)
	X_train= scaler.transform(A)
	print(X_train.shape)

	X_val=scaler.transform(X_val[:,first_col:last_col])
	print(X_val.shape)
	alphas = np.logspace(-5,-1,100)

	#cross_validate(X_train, X_val, y_train, y_val,alphas)
	
	architectures = [ (10,20), (2,10,20),(100, ), (10,20,30,100), (100,100), (100, 50, 80, 1), (1,10, 20), (100, 500, 200),(1000,)]
	for architecture in architectures:
		print(architecture)
		MLP = MLPClassifier(hidden_layer_sizes=architecture,activation='relu')
		s = time.time()
		MLP.fit(X_train,y_train)
		e = time.time()
		print("took " + str((e-s)/60) + " seconds to fit.")
		y_hat = MLP.predict_proba(X_val)
		loss = log_loss(y_val, y_hat)
		print("log loss: " +str(loss))


if __name__ == '__main__':
	main()