from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np 
import sys

class GradientBoostingMachine():

   

    def __init__(self, n_estimator, pseudo_residual_func, learning_rate=0.1, min_sample=5, max_depth=10):
        '''
        Initialize gradient boosting class
        
        :param n_estimator: number of estimators (i.e. number of rounds of gradient boosting)
        :pseudo_residual_func: function used for computing pseudo-residual
        :param learning_rate: step size of gradient descent
        :param min_sample: an internal node can be splitted only if it contains points more than min_smaple
        :param max_depth: restriction of tree depth.
        '''
        self.n_estimator = n_estimator
        self.pseudo_residual_func = pseudo_residual_func
        self.learning_rate = learning_rate
        self.min_sample = min_sample
        self.max_depth = max_depth
        #self.max_leaf_nodes= max_leaves

    def fit(self, train_data, train_target):


        predict_value = np.zeros((len(train_target), 1))
        pseudo_residual = self.pseudo_residual_func(train_target, predict_value)
        model_dict = {}
        for i in range(self.n_estimator):
            print("fitting estimator "+str(i+1))
            estimator = DecisionTreeRegressor(min_samples_leaf=self.min_sample, max_depth=self.max_depth)
            estimator.fit(train_data, pseudo_residual)
            model_dict[i] = estimator
            predict_value += self.learning_rate *estimator.predict(train_data).reshape(-1,1)
            pseodo_residual = self.pseudo_residual_func(train_target, predict_value)
        self.model_dict = model_dict
    
    def predict(self, test_data):

        predict_vector = np.zeros((len(test_data), 1))
        for i in range(self.n_estimator):
            predict_temp = self.model_dict[i].predict(test_data).reshape(-1,1)
            predict_vector += self.learning_rate * predict_temp
        return 1/(1+np.exp(-1*predict_vector))


