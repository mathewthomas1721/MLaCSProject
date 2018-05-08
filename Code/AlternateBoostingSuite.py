import numpy as np 
import pandas as pd
import pickle
from AlternatingBoostingMachine import *
from GradientBoostingMachine import *
from tt_split import *
import time
import sys
from sklearn.model_selection import GridSearchCV, PredefinedSplit,ParameterGrid

from sklearn.metrics import log_loss, make_scorer
from sklearn.preprocessing import StandardScaler
from tt_split import *
import time
from sklearn.externals import joblib



def mle_pr(train_target,train_preds):
	
	train_preds=train_preds.flatten()
	train_target = train_target.flatten()
	denom = np.exp(train_preds)+1
	denom=denom.flatten()
	

	
	first_term = np.divide(train_target,denom)

	second_term = ((1-train_target)*np.exp(train_preds))/(np.exp(train_preds)+1)

	
	#res = first_term + second_term

	return -(first_term + second_term)


y = np.array([0,1,1,0,1])
preds = np.array([.5,.5,.25,.3,.9])

abc = AlternatingBoostingMachine(50,mle_pr)

#print(abc)
season="2012"
df = pd.read_csv("../Data/RegularSeasonFeatures"+str(season)+".csv",index_col=0)
X_train, X_val, X_test, y_train, y_val, y_test = train_test_split_season(df,validation=True)
#abc.fit(X_train,y_train)
#y_hat = abc.predict(X_val)

#print(log_loss(y_val,y_hat))


gbm = GradientBoostingMachine(10,mle_pr,learning_rate=0.1)
gbm.fit(X_train,y_train)
y_hat = gbm.predict(X_val)
print(log_loss(y_val,y_hat))