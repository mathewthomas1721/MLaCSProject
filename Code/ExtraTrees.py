from tt_split import train_test_split_season
# Pandas is used for data manipulation
import pandas as pd

# Use numpy to convert to arrays
import numpy as np

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Import the model we are using
from sklearn.ensemble import ExtraTreesClassifier

# Import matplotlib for plotting and use magic command for Jupyter Notebooks
import matplotlib.pyplot as plt

# Use datetime for creating date objects for plotting
import datetime

def compute_log_loss(y_true, pred_probs,eps=10**-4):
    pred_probs[np.where(pred_probs==0)]= eps
    pred_probs[np.where(pred_probs==1)]= 1 - eps
    res = (-1/y_true.shape[0])*np.sum(np.dot(y_true,np.log(pred_probs)) +
    np.dot(1-y_true,np.log(1-pred_probs)))
    return res

def ETreesRegressor(fname):
    # Read in data and display first 5 rows
    features = pd.read_csv(fname,index_col = 0)
    #features= features.drop('id',axis = 1)
    #print(features.shape[0])
    #print(features.isnull().any())
    #features = features.dropna(how='any')
    #print(features.shape[0])

    #features.head(5)

    #print('The shape of our features is:', features.shape)

    # Descriptive statistics for each column
    #features.describe()

    # One-hot encode the data using pandas get_dummies
    #features = pd.get_dummies(features)

    # Display the first 5 rows of the last 12 columns
    #features.iloc[:,5:].head(5)

    # Labels are the values we want to predict
    #labels = np.array(features['y'])

    # Remove the labels from the features
    # axis 1 refers to the columns
    #features= features.drop('y', axis = 1)

    # Saving feature names for later use
    feature_list = list(features.columns)
    #print(feature_list)
    # Convert to numpy array
    #features = np.array(features)

    # Split the data into training and testing sets
    #train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, shuffle = False)

    train_features, val_features, test_features, train_labels, val_labels, test_labels = train_test_split_season(features,validation = True)

    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Val Features Shape:', val_features.shape)
    print('Val Labels Shape:', val_labels.shape)


    max_features = None
    max_depth = None
    min_samples_split =  2
    min_samples_leaf = 1
    max_leaf_nodes = None

    rf = ExtraTreesClassifier(n_estimators = 100, random_state = 42)#, max_features = max_features, max_depth = max_depth, min_samples_leaf = min_samples_leaf, min_samples_split = min_samples_split, max_leaf_nodes = max_leaf_nodes)
    # Train the model on training data
    rf.fit(train_features, train_labels);

    # Use the forest's predict method on the test data
    predictions = rf.predict_proba(val_features)
    print(compute_log_loss(val_labels,predictions), max_features)
    '''
    for i in range(0,1000):
        max_features = i+1
        # Instantiate model with 1000 decision trees
        rf = ExtraTreesClassifier(n_estimators = 100, random_state = 42)#, max_features = max_features, max_depth = max_depth, min_samples_leaf = min_samples_leaf, min_samples_split = min_samples_split, max_leaf_nodes = max_leaf_nodes)
        # Train the model on training data
        rf.fit(train_features, train_labels);

        # Use the forest's predict method on the test data
        predictions = rf.predict_proba(val_features)
        print(compute_log_loss(val_labels,predictions), max_features)
        '''
    '''
    train_features, test_features, train_labels, test_labels = train_test_split_season(features,validation = False)

    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)

    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators = 100, random_state = 42, max_features = max_features, max_depth = max_depth, min_samples_leaf = min_samples_leaf, min_samples_split = min_samples_split, max_leaf_nodes = max_leaf_nodes)

    # Train the model on training data
    rf.fit(train_features, train_labels);

    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    print(compute_log_loss(test_labels,predictions))
    '''


ETreesRegressor('../Data/RegularSeasonFeatures2012.csv')
