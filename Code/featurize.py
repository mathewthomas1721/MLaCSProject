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
POST_SEASON_ALL = DATA_PATH + "AtBats_PostSeason_2012-2017_sorted.csv"
REG_SEASON_ALL = DATA_PATH + "AtBats_RegularSeason_2012-2017_sorted.csv"
SPRING_TRN_ALL = DATA_PATH + "AtBats_SpringTraining_2012-2017_sorted.csv"
PLAYERS_ALL = DATA_PATH + "MLB_Players.csv"
PITCHERS_ALL = DATA_PATH + "MLB_Pitchers.csv"


def getRatesPitcher(df):
	dfslim  = df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1)
	dfslim = dfslim.drop(['home_score', 'away_score', 'stadium', 'outs', 'inning', 'descr', 'side', 'bases', 'date', 'year'], axis=1)
	pitchers = dfslim['pitcher'].unique()
	#print(dfslim.info())
	pitcherdfs = []
	for i, p in enumerate(pitchers):
		dfpitcher = dfslim[dfslim.pitcher == p]
		dfpitcher = dfpitcher.reset_index()
		#dfpitcher = dfpitcher.drop('index', axis=1)
		dfpitcher['cumpitcherk'] = dfpitcher['y'].expanding(2).sum()
		dfpitcher.cumpitcherk = dfpitcher.cumpitcherk.fillna(0)
		dfpitcher['pitcherkrate'] = dfpitcher['cumpitcherk'] / dfpitcher.index
		pitcherdfs.append(dfpitcher)
	allpitchers = pd.concat(pitcherdfs, axis=0)
	allpitchers = allpitchers.drop(['y', 'pitcher', 'batter'],axis=1)
	return allpitchers


def getRatesBatter(df):
	dfslim  = df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1)
	dfslim = dfslim.drop(['home_score', 'away_score', 'stadium', 'outs', 'inning', 'descr', 'side', 'bases', 'date', 'year'], axis=1)
	batters = dfslim['batter'].unique()
	#print(dfslim.info())
	batterdfs = []
	for i, b in enumerate(batters):
		dfbatter = dfslim[dfslim.batter == b]
		dfbatter = dfbatter.reset_index()
		#dfbatter = dfbatter.drop('index', axis=1)
		dfbatter['cumbatterk'] = dfbatter['y'].expanding(2).sum()
		dfbatter.cumbatterk = dfbatter.cumbatterk.fillna(0)
		dfbatter['batterkrate'] = dfbatter['cumbatterk'] / dfbatter.index
		batterdfs.append(dfbatter)
	allbatters = pd.concat(batterdfs, axis=0)
	allbatters = allbatters.drop(['y', 'pitcher', 'batter'],axis=1)
	return allbatters





def main():
	dfRegSeason = pd.read_csv(REG_SEASON_ALL)
	dfRegSeason['y'] = np.where(dfRegSeason['descr']=='Strikeout', 1, 0)
	dfRegSeason['year'] = pd.DatetimeIndex(dfRegSeason['date']).year
	dfPlayers = pd.read_csv(PLAYERS_ALL)
	dfPitchers = pd.read_csv(PITCHERS_ALL)

	dfRegSeason  = dfRegSeason.drop(dfRegSeason.columns[dfRegSeason.columns.str.contains('unnamed',case = False)],axis = 1)


	ballparks = dfRegSeason['stadium'].unique()
	dfRegSeason.bases = dfRegSeason.bases.fillna("E")
	allps = getRatesPitcher(dfRegSeason)
	allbs = getRatesBatter(dfRegSeason)
	dfRegSeason['index'] = dfRegSeason.index
	#print(dfRegSeason.head(50))
	dfRegSeason = pd.merge(dfRegSeason, allps, how = 'left', left_on = 'index', right_on = 'index')
	dfRegSeason = pd.merge(dfRegSeason, allbs, how = 'left', left_on = 'index', right_on = 'index')


	oneHotstad = pd.get_dummies(dfRegSeason['stadium'])
	oneHotbase = pd.get_dummies(dfRegSeason['bases'])
	oneHotside = pd.get_dummies(dfRegSeason['side'])
	oneHotinning = pd.get_dummies(dfRegSeason['inning'], prefix = 'inn')
	oneHotouts = pd.get_dummies(dfRegSeason['outs'], prefix = 'outs')
	dfPlayers.bats = dfPlayers.bats.fillna("NA")
	dfPitchers.pitches = dfPitchers.pitches.fillna("NA")
	#print(dfPitchers.head(1))

	dfbatters = pd.merge(dfRegSeason, dfPlayers, how = 'left', left_on = 'batter', right_on = 'bref_id')
	dfboth = pd.merge(dfbatters, dfPitchers, how = 'left', left_on = 'pitcher', right_on = 'bref_id')
	pd.options.display.max_columns = 999
	#print(dfboth.head(1))
	dfboth.bats = dfboth.bats.fillna("NA")
	dfboth.pitches = dfboth.pitches.fillna("NA")
	dfboth['matchup'] = dfboth['bats'] + dfboth['pitches']
	#print(dfboth.head(1))
	oneHotHand = pd.get_dummies(dfboth['matchup'])
	#print(oneHotHand.head(1))


	dfRegSeasonfeat = pd.concat([dfRegSeason, oneHotstad, oneHotbase, oneHotside, oneHotinning, oneHotHand, oneHotouts], axis=1)
	#print(dfRegSeasonfeat.info)
	
	dfRegSeasonfeat['score'] = ((-1) ** dfRegSeasonfeat['top']) * (dfRegSeasonfeat['home_score'] - dfRegSeasonfeat['away_score'])
	dfRegSeasonfeat =  dfRegSeasonfeat.drop(['cumbatterk','cumpitcherk','index', 'side', 'stadium', 'bases', 'home_score', 'away_score', 'descr', 'date', 'inning', 'outs', 'year', 'batter', 'pitcher'], axis=1)
	print(dfRegSeasonfeat)
	dfRegSeasonfeat  = dfRegSeasonfeat.drop(dfRegSeasonfeat.columns[dfRegSeasonfeat.columns.str.contains('unnamed',case = False)],axis = 1)
	dfRegSeasonfeat = dfRegSeasonfeat.head(1025296)
	dfRegSeasonfeat.to_csv(DATA_PATH + "RegularSeasonFeatures2012-2017.csv")

	


if __name__=="__main__":
	main()