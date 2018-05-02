from tt_split import train_test_split_season
# Pandas is used for data manipulation
import pandas as pd

# Use numpy to convert to arrays
import numpy as np

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor


# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot

# Import matplotlib for plotting and use magic command for Jupyter Notebooks
import matplotlib.pyplot as plt

# Use datetime for creating date objects for plotting
import datetime

# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot

def compute_log_loss(y_true, pred_probs,eps=10**-4):
    pred_probs[np.where(pred_probs==0)]= eps
    pred_probs[np.where(pred_probs==1)]= 1 - eps
    res = (-1/y_true.shape[0])*np.sum(np.dot(y_true,np.log(pred_probs)) +
    np.dot(1-y_true,np.log(1-pred_probs)))
    return res

def RFRegressor(fname):
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

    # Convert to numpy array
    #features = np.array(features)

    # Split the data into training and testing sets
    #train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, shuffle = False)
    '''
    train_features, val_features, test_features, train_labels, val_labels, test_labels = train_test_split_season(features,validation = True)

    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Val Features Shape:', val_features.shape)
    print('Val Labels Shape:', val_labels.shape)
    '''
    
    max_features = 21
    max_depth = 7
    min_samples_split =  2
    min_samples_leaf = 1
    max_leaf_nodes = 206
    '''
    for i in range(1,1000):
        max_leaf_nodes = i+1
        # Instantiate model with 1000 decision trees
        rf = RandomForestRegressor(n_estimators = 100, random_state = 42, max_features = max_features, max_depth = max_depth, min_samples_leaf = min_samples_leaf, min_samples_split = min_samples_split, max_leaf_nodes = max_leaf_nodes)
        # Train the model on training data
        rf.fit(train_features, train_labels);

        # Use the forest's predict method on the test data
        predictions = rf.predict(test_features)
        print(compute_log_loss(test_labels,predictions), max_leaf_nodes)
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
    # Pull out one tree from the forest
    tree = rf.estimators_[5]

    # Export the image to a dot file
    export_graphviz(tree, out_file = "tree.dot", feature_names = feature_list, rounded = True, precision = 1)

    # Use dot file to create a graph
    (graph, ) = pydot.graph_from_dot_file("tree.dot")

    # Write graph to a png file
    graph.write_png('tree.png')

    # Limit depth of tree to 3 levels
    rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3)
    rf_small.fit(train_features, train_labels)

    # Extract the small tree
    tree_small = rf_small.estimators_[5]

    # Save the tree as a png image
    export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = feature_list, rounded = True, precision = 1)

    (graph, ) = pydot.graph_from_dot_file('small_tree.dot')

    graph.write_png('small_tree.png');

    # Get numerical feature importances
    importances = list(rf.feature_importances_)

    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

    # Print out the feature and importances
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

    # New random forest with only the two most important variables
    rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=42)

    # Extract the two most important features
    important_indices = [feature_list.index('temp_1'), feature_list.index('average')]
    train_important = train_features[:, important_indices]
    test_important = test_features[:, important_indices]

    # Train the random forest
    rf_most_important.fit(train_important, train_labels)

    # Make predictions and determine the error
    predictions = rf_most_important.predict(test_important)

    errors = abs(predictions - test_labels)

    # Display the performance metrics
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

    mape = np.mean(100 * (errors / test_labels))
    accuracy = 100 - mape

    print('Accuracy:', round(accuracy, 2), '%.')

    # Set the style
    plt.style.use('fivethirtyeight')

    # list of x locations for plotting
    x_values = list(range(len(importances)))

    # Make a bar chart
    plt.bar(x_values, importances, orientation = 'vertical')

    # Tick labels for x axis
    plt.xticks(x_values, feature_list, rotation='vertical')

    # Axis labels and title
    plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');



    # Dates of training values
    months = features[:, feature_list.index('month')]
    days = features[:, feature_list.index('day')]
    years = features[:, feature_list.index('year')]

    # List and then convert to datetime object
    dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
    dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

    # Dataframe with true values and dates
    true_data = pd.DataFrame(data = {'date': dates, 'actual': labels})

    # Dates of predictions
    months = test_features[:, feature_list.index('month')]
    days = test_features[:, feature_list.index('day')]
    years = test_features[:, feature_list.index('year')]

    # Column of dates
    test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]

    # Convert to datetime objects
    test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]

    # Dataframe with predictions and dates
    predictions_data = pd.DataFrame(data = {'date': test_dates, 'prediction': predictions})

    # Plot the actual values
    plt.plot(true_data['date'], true_data['actual'], 'b-', label = 'actual')

    # Plot the predicted values
    plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label = 'prediction')
    plt.xticks(rotation = '60');
    plt.legend()

    # Graph labels
    plt.xlabel('Date'); plt.ylabel('Maximum Temperature (F)'); plt.title('Actual and Predicted Values');

    # Make the data accessible for plotting
    true_data['temp_1'] = features[:, feature_list.index('temp_1')]
    true_data['average'] = features[:, feature_list.index('average')]
    true_data['friend'] = features[:, feature_list.index('friend')]

    # Plot all the data as lines
    plt.plot(true_data['date'], true_data['actual'], 'b-', label  = 'actual', alpha = 1.0)
    plt.plot(true_data['date'], true_data['temp_1'], 'y-', label  = 'temp_1', alpha = 1.0)
    plt.plot(true_data['date'], true_data['average'], 'k-', label = 'average', alpha = 0.8)
    plt.plot(true_data['date'], true_data['friend'], 'r-', label = 'friend', alpha = 0.3)

    # Formatting plot
    plt.legend(); plt.xticks(rotation = '60');

    # Lables and title
    plt.xlabel('Date'); plt.ylabel('Maximum Temperature (F)'); plt.title('Actual Max Temp and Variables');
    '''

RFRegressor('../Data/RegularSeasonFeatures2012.csv')
