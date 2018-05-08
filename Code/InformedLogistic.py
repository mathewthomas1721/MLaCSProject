import ada_boost
import gradient_boost
from cvxpy import *
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostClassifier



def fit_ensemble():

def main():
	pass

"""
for each season: 
	fit each unique model & out put results into a picture
	check calibration probability
	store model in a season-specific ensemble  
	add to log reg 
	log reg to others (0/1, probs)



"""

if __name__ == '__main__':
	main()

from cvxpy import *
import numpy as np
import time



num_samples = 1000000
num_features = 8
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
