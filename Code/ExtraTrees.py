from sklearn.metrics import log_loss,make_scorer
from tt_split import train_test_split_season
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import export_graphviz
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import pydot
import matplotlib.pyplot as plt
import datetime

def ETreesRegressor(fname):
    # Read in data and display first 5 rows
    features = pd.read_csv(fname,index_col = 0)

    feature_list = list(features.columns)

    train_features, val_features, test_features, train_labels, val_labels, test_labels = train_test_split_season(features,validation = True)

    X_train_val = np.vstack((train_features, val_features))
    y_train_val = np.concatenate((train_labels, val_labels))
    val_fold = [-1]*len(train_features) + [0]*len(val_features) #0 corresponds to validation

    # Now we set up and do the grid search over l2reg. The np.concatenate
    # command illustrates my search for the best hyperparameter. In each line,
    # I'm zooming in to a particular hyperparameter range that showed promise
    # in the previous grid. This approach works reasonably well when
    # performance is convex as a function of the hyperparameter, which it seems
    # to be here.
    param_grid = [{'max_features' : np.arange(1,len(feature_list))}]
    #'max_depth' : np.arange(1,100,20)}],
    #'min_samples_split' :  np.arange(2,10,2),
    #'min_samples_leaf' : np.arange(2,10,2),
    #'max_leaf_nodes' : np.arange(2,100,20)}]

    ridge_regression_estimator = ExtraTreesRegressor()
    grid = GridSearchCV(ridge_regression_estimator,
                        param_grid,
                        return_train_score=True,
                        cv = PredefinedSplit(test_fold=val_fold),
                        refit = True,
                        scoring = make_scorer(log_loss,
                                              greater_is_better = False))
    grid.fit(X_train_val, y_train_val)

    df = pd.DataFrame(grid.cv_results_)
    # Flip sign of score back, because GridSearchCV likes to maximize,
    # so it flips the sign of the score if "greater_is_better=FALSE"
    df['mean_test_score'] = -df['mean_test_score']
    df['mean_train_score'] = -df['mean_train_score']
    cols_to_keep = ["param_max_features","mean_test_score","mean_train_score"]
    #cols_to_keep = ["param_max_features", "param_max_depth","param_min_samples_split","param_min_samples_leaf", "param_max_leaf_nodes","mean_test_score","mean_train_score"]
    df_toshow = df[cols_to_keep].fillna('-')
    df_toshow = df_toshow.sort_values(by=["param_l2reg"])
    print(df_toshow)


    '''
    train_features, test_features, train_labels, test_labels = train_test_split_season(features,validation = False)

    max_features = 25
    max_depth = 15
    min_samples_split =  2
    min_samples_leaf = 1
    max_leaf_nodes = 206
    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)

    # Instantiate model with 1000 decision trees
    rf = ExtraTreesRegressor(n_estimators = 100, random_state = 42, max_features = max_features, max_depth = max_depth, min_samples_leaf = min_samples_leaf, min_samples_split = min_samples_split, max_leaf_nodes = max_leaf_nodes)

    # Train the model on training data
    rf.fit(train_features, train_labels);

    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    print(log_loss(test_labels,predictions))
    '''


ETreesRegressor('../Data/RegularSeasonFeatures2012.csv')
