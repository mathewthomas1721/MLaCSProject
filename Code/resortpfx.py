import numpy as np
import pandas as pd
import scipy as sp
import sys
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression as LogisticModel



DATA_PATH = "../Data/MLB_PitchFX_2012/"
WRITE_PATH= "../Data/OutData/"
FIG_PATH = "../Figs/"
year = sys.argv[1]



REG_SEASON_YEAR = DATA_PATH + "MLB_PitchFX_RegularSeason_" +year +".csv"
df_season = pd.read_csv(REG_SEASON_YEAR)


#dfRegSeason = dfRegSeason.sort_values(by=["date"],kind='mergesort')
df_season = df_season.sort_values(by=["tfs_zulu"], kind='mergesort')
df_season = df_season.sort_values(by=["stadium"],kind='mergesort')
df_season = df_season.sort_values(by=["date"],kind='mergesort')
df_season = df_season.reset_index()
df_season = df_season.drop('index', axis=1)
df_season  = df_season.drop(df_season.columns[df_season.columns.str.contains('unnamed',case = False)],axis = 1)
df_season.to_csv(DATA_PATH + "MLB_PitchFX_RegularSeason_" + year +"_sorted.csv")