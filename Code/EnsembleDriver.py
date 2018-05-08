import ada_boost
import pandas as pd
from cvxpy import *
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
import numpy as np 
import pandas as pd
import pickle
from boostedMLP import *
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, RandomForestRegressor, DecisionTreeClassifier
from tt_split import *
import time
import sys
from sklearn.model_selection import GridSearchCV, PredefinedSplit,RandomizedSearchCV, ParameterGrid
from sklearn.metrics import mean_squared_error, make_scorer,log_loss
import AdaBoostSuite



def fit_ensemble_members():

def main():
	Seasons = ["2012","2013","2014","2015","2016","2017"]
	for season in Seasons:
		file_name = "../Data/RegularSeasonFeatures"+season+".csv"
		df = pd.read_csv(file_name,index_col=0)


if __name__ == '__main__':
	main()

from cvxpy import *
import numpy as np
import time



num_samples = 1000000
num_features = 5
y = np.random.choice([0,1],num_samples,replace=True)


X = np.matrix(np.random.rand(num_samples,num_features))

theta = Variable(num_features,1)

objective = Minimize(sum_entries(- (y*log(X*theta) +(1-y)*log(X*theta)))  )
constraints = [sum_entries(theta)==1,theta>=0]
prob = Problem(objective,constraints)
s = time.time()
prob.solve()
e = time.time()
print("took "+str( (e-s)/60)+ " minutes.")
print(theta.value)
