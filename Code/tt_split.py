import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



def train_test_split_season(df,ts=0.2,validation=False,val_ts=0.1):
	"""
	Performs train/test split on a single season of data.

	Inputs:
			df - the data frame with a featurized season.
				Must have a column 'y' with the strikeout results.
				assumes 'y' is the first column of the data frame (i.e col 0)
				and that the rest of the features are in columns 1:

			ts  - the fraction of the data frame that should be in the test set

			validation - whether or not a validation set should be returned

			val_ts - the fraction of the training data that should be set aside
					in the validation set

	Outputs:
		A tuple with results.

		If validation is false:
			(X_train, X_test, y_train, y_test)
		else:
			(X_train, X_val, X_test, y_train, y_val, y_test)
	"""

	y = df['y']  # necessary preprocessing into numpy arrays
	y=y.as_matrix()
	X = df[df.columns[1:]]
	X = X.as_matrix()

	if not validation:
		X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=ts,shuffle=False)
		return X_train, X_test, y_train, y_test
	else:
		X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=ts,shuffle=False)
		X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ts, shuffle=False)
		return X_train, X_val, X_test,y_train, y_val, y_test

def main():

	""" unit tests"""
	test_file = "../Data/RegularSeasonFeatures2012.csv"
	df = pd.read_csv(test_file,index_col=0)
	print(df.head())
	(a,b,c,d) =train_test_split_season(df)
	print(a.shape)
	print(b.shape)
	print(c.shape)
	print(d.shape)


if __name__ == '__main__':
	main()
