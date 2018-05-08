from sklearn.neural_network import MLPRegressor
import numpy as np 
import pandas as pd
import pickle



class GradientBoostingMachine():

    def __init__(self, n_estimator, pseudo_residual_func, est_type="DT",learning_rate=0.1, params = {'min_sample':5, 'max_depth'=3}):
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
    
    def fit(self, train_data, train_target):
        '''
        Fit gradient boosting model
        '''
        
     
        self.ensemble=[]
        for i in range(1,self.n_estimator+1):
            estimator = DecisionTreeRegressor(criterion="mse",max_depth=self.max_depth,min_samples_split=self.min_sample)
            ## assume i-1 fit estimators
            new_targ =0
            for j in range(len(self.ensemble)):
                preds = self.ensemble[j].predict(train_data)
                #z = self.pseudo_residual_func(train_target,new_targ)
                new_targ +=self.learning_rate*self.ensemble[j].predict(train_data).reshape((train_target.shape[0],1))
            estimator.fit(train_data,self.pseudo_residual_func(train_target,new_targ))
            self.ensemble.append(estimator)
            
            
            
        
        ## have a function that takes in x and returns a weighted linear combo of the fit estimators.
    
    def predict(self, test_data):
        '''
        Predict value
        '''
    
        preds = np.array([estimator.predict(test_data) for estimator in self.ensemble])
        weights = self.learning_rate*np.ones(preds.shape[0])
        return np.dot(weights, preds)