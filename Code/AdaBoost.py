from sklearn.metrics import log_loss,make_scorer
from tt_split import train_test_split_season
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import export_graphviz
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import pydot
import matplotlib.pyplot as plt
import datetime

def ABoostRegressor(fname):
    features = pd.read_csv(fname,index_col = 0)

    feature_list = list(features.columns)

    train_features, val_features, test_features, train_labels, val_labels, test_labels = train_test_split_season(features,validation = True)

    X_train_val = np.vstack((train_features, val_features))
    y_train_val = np.concatenate((train_labels, val_labels))
    val_fold = [-1]*len(train_features) + [0]*len(val_features) #0 corresponds to validation


    param_grid = [{'n_estimators' : 10**np.arange(0,3), 'learning_rate':10.0**np.arange(-2,1,1)}]


    ridge_regression_estimator = AdaBoostRegressor()
    grid = GridSearchCV(ridge_regression_estimator,
                        param_grid,
                        return_train_score=True,
                        cv = PredefinedSplit(test_fold=val_fold),
                        refit = True,
                        scoring = make_scorer(log_loss,
                                              greater_is_better = False))
    grid.fit(X_train_val, y_train_val)

    df = pd.DataFrame(grid.cv_results_)
    df['mean_test_score'] = -df['mean_test_score']
    df['mean_train_score'] = -df['mean_train_score']
    cols_to_keep = ['n_estimators','learning_rate', 'loss']
    df_toshow = df[cols_to_keep].fillna('-')
    df_toshow = df_toshow.sort_values(by=["mean_test_score"])
    print(df_toshow[0])


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


    rf = AdaBoostRegressor(n_estimators=50, learning_rate=1.0, loss='linear', random_state=42)

    # Train the model on training data
    rf.fit(train_features, train_labels);

    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    print(log_loss(test_labels,predictions))
    '''
ABoostRegressor('../Data/RegularSeasonFeatures2012.csv')
