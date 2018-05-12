import numpy as np 
import pandas as pd
import pickle
from AlternatingBoostingMachine import *
from GradientBoostingMachine import *
from tt_split import *
import time
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit,ParameterGrid

from sklearn.metrics import log_loss, make_scorer
from sklearn.preprocessing import StandardScaler
from tt_split import *
import time
from sklearn.externals import joblib
from cvxpy import *

def mle_pr(train_target,train_preds):
	
	train_preds=train_preds.flatten()
	train_target = train_target.flatten()
	denom = np.exp(train_preds)+1
	denom=denom.flatten()
	

	
	first_term = np.divide(train_target,denom)

	second_term = ((1-train_target)*np.exp(train_preds))/(np.exp(train_preds)+1)

	
	#res = first_term + second_term
	#return train_target-train_preds
	return -(first_term + second_term)


season="2012"
df = pd.read_csv("../Data/RegularSeasonFeatures"+str(season)+".csv",index_col=0)

X_train, X_val, X_test, y_train, y_val, y_test = train_test_split_season(df,validation=True)


quals_train = X_train[:,13:X_train.shape[1]-1]
quals_val = X_val[:,13:X_train.shape[1]-1]
X_train=X_train[:,0:13]
X_val = X_val[:,0:13]
print(quals_train.shape)




print("Centered, rates only")
scaler=StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)




print('perceptron')
clf = MLPClassifier(hidden_layer_sizes=(5,1000))

clf.fit(X_train,y_train)
y_hat = clf.predict_proba(X_val)
y_hat_real = y_hat
y_hat_real_train = clf.predict_proba(X_train)
print(log_loss(y_val,y_hat))



print("qualitative")

xt1 = X_train
xt2 = X_val

X_train=quals_train
X_val=quals_val


print('perceptron')
clf = MLPClassifier(hidden_layer_sizes=(5,))

clf.fit(X_train,y_train)
y_hat = clf.predict_proba(X_val)
y_hat_qual=y_hat
y_hat_qual_train = clf.predict_proba(X_train)
print(log_loss(y_val,y_hat))


print(y_hat_qual.shape)

X_f = np.hstack((y_hat_qual[:,1].reshape(-1,1),y_hat_real[:,1].reshape(-1,1)))
w = np.array([[.5],[.5]])
print(y_hat_qual[:,1].shape)
print(y_hat_real[:,1].shape)
print(X_f.shape)
av = np.matrix(X_f)*w
print("averaging")
print(log_loss(y_val,av))


print("convex optimized weighting")
X_f = np.hstack((y_hat_qual_train[:,1].reshape(-1,1),y_hat_real_train[:,1].reshape(-1,1)))
X_f=np.matrix(X_f)
theta = Variable(X_f.shape[1],1)
objective = Minimize(sum_entries(- (y_train*log(X_f*theta) +(1-y_train)*log(X_f*theta)))  )
constraints = [sum_entries(theta)==1,theta>=0]
prob = Problem(objective,constraints)
s = time.time()
prob.solve()
e = time.time()
print("took "+str( (e-s)/60)+ " minutes.")
print(theta.value)

X_f = np.matrix(np.hstack((y_hat_qual[:,1].reshape(-1,1),y_hat_real[:,1].reshape(-1,1))))
av = X_f*theta.value
print(av)
print(log_loss(y_val,av))





