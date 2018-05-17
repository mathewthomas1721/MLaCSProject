import numpy as np
import pandas as pd
import scipy as sp
import sys
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression as LogisticModel


DATA_PATH = "../Data/"
WRITE_PATH= "../Data/OutData/"
FIG_PATH = "../Figs/"
PLAYERS_ALL = DATA_PATH + "MLB_Players.csv"
PITCHERS_ALL = DATA_PATH + "MLB_Pitchers.csv"


for i in range(2,8):
	REG_SEASON_YEAR = DATA_PATH + "MLB_AtBats_RegularSeason_201" + str(i) + ".csv"
	df_season = pd.read_csv(REG_SEASON_YEAR)
	df_season['y'] = np.where(df_season['descr']=='Strikeout', 1, 0)
	df_season.bases = df_season.bases.fillna("E")
	#dfRegSeason = dfRegSeason.sort_values(by=["date"],kind='mergesort')
	df_season = df_season.replace('Globe Life Park in Arlington', 'Rangers Ballpark in Arlington')
	df_season = df_season.replace('O.co Coliseum', 'Oakland Coliseum')
	df_season = df_season.replace('Guaranteed Rate Field', 'U.S. Cellular Field')
	df_season['lenbases'] = df_season.bases.apply(len)	
	df_season['outsscore'] = df_season['outs'] + df_season['home_score'] + df_season['away_score']	
	df_season = df_season.sort_values(by=["lenbases"], kind='mergesort')	
	df_season = df_season.sort_values(by=["outsscore"], kind='mergesort')	
 	df_season = df_season.sort_values(by=["side"],kind='mergesort')	 	df_season = df_season.sort_values(by=["side"],kind='mergesort')
 	df_season = df_season.sort_values(by=["inning"],kind='mergesort')	 	df_season = df_season.sort_values(by=["inning"],kind='mergesort')
 	df_season = df_season.sort_values(by=["stadium"],kind='mergesort')	 	df_season = df_season.sort_values(by=["stadium"],kind='mergesort')
 	df_season = df_season.sort_values(by=["date"],kind='mergesort')	 	df_season = df_season.sort_values(by=["date"],kind='mergesort')
 	df_season = df_season.reset_index()	 	df_season = df_season.reset_index()
 	df_season = df_season.drop('index', axis=1)	 	df_season = df_season.drop('index', axis=1)
	df_season = df_season.drop(['lenbases', 'outsscore'], axis=1)
	df_season = df_season.drop('index', axis=1)
	df_season  = df_season.drop(df_season.columns[df_season.columns.str.contains('unnamed',case = False)],axis = 1)
	df_season.to_csv(DATA_PATH + "MLB_AtBats_RegularSeason_201" + str(i) + "_sorted.csv")
	print(i)