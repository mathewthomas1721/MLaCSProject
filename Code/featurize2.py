import numpy as np
import pandas as pd
import scipy as sp
import sys
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression as LogisticModel

import featurizepfx


year = sys.argv[1]

N = float(sys.argv[2])
DATA_PATH = "../Data/"
WRITE_PATH= "../Data/OutData/"
FIG_PATH = "../Figs/"
#POST_SEASON_ALL = DATA_PATH + "AtBats_PostSeason_2012-2017_sorted.csv"
REG_SEASON_ALL = DATA_PATH + "MLB_AtBats_RegularSeason_" + year + "_sorted.csv"
#SPRING_TRN_ALL = DATA_PATH + "AtBats_SpringTraining_2012-2017_sorted.csv"
PLAYERS_ALL = DATA_PATH + "MLB_Players_new.csv"
PITCHERS_ALL = DATA_PATH + "MLB_Pitchers_new.csv"


def getLeagueAvg(df):
	tot_k = df['y'].sum()
	tot_pa = df.shape[0]
	return N * tot_k / tot_pa

def getRatesPitcher(df, league_avg):
	dfslim  = df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1)
	dfslim = dfslim.drop(['home_score', 'away_score', 'stadium', 'outs', 'inning', 'descr', 'side', 'bases', 'date', 'visitor', 'home'], axis=1)
	pitchers = dfslim['pitcher'].unique()
	#print(dfslim.info())
	pitcherdfs = []
	for i, p in enumerate(pitchers):
		print(i)
		dfpitcher = dfslim[dfslim.pitcher == p]
		dfpitcher = dfpitcher.reset_index()
		#dfpitcher = dfpitcher.drop('index', axis=1)
		dfpitcher['cumpitcherk'] = dfpitcher['y'].expanding(2).sum()
		dfpitcher.cumpitcherk = dfpitcher.cumpitcherk.fillna(0)
		dfpitcher['pitcherkrate'] = (dfpitcher['cumpitcherk'] - dfpitcher['y'] + league_avg ) / (dfpitcher.index + N)
		pitcherdfs.append(dfpitcher)
		for wind, window_size in enumerate([151, 71, 26]):
			dfpitcher['wind' + str(wind) + 'pkrate'] = (dfpitcher.y.rolling(window_size).sum() - dfpitcher['y']) / (window_size - 1) - dfpitcher['pitcherkrate']
			dfpitcher['wind' + str(wind) + 'pkrate'] = dfpitcher['wind' + str(wind) + 'pkrate'].fillna(0.0)
			
	allpitchers = pd.concat(pitcherdfs, axis=0)
	allpitchers = allpitchers.drop(['y', 'pitcher', 'batter'],axis=1)
	return allpitchers


def getRatesBatter(df, league_avg):
	dfslim  = df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1)
	dfslim = dfslim.drop(['home_score', 'away_score', 'stadium', 'outs', 'inning', 'descr', 'side', 'bases', 'date', 'visitor', 'home'], axis=1)
	batters = dfslim['batter'].unique()
	#print(dfslim.info())
	batterdfs = []
	for i, b in enumerate(batters):
		print(i)
		dfbatter = dfslim[dfslim.batter == b]
		dfbatter = dfbatter.reset_index()
		#dfbatter = dfbatter.drop('index', axis=1)
		dfbatter['cumbatterk'] = dfbatter['y'].expanding(2).sum()
		dfbatter.cumbatterk = dfbatter.cumbatterk.fillna(0)
		dfbatter['batterkrate'] = (dfbatter['cumbatterk'] - dfbatter['y'] + league_avg) / (dfbatter.index + N)
		batterdfs.append(dfbatter)
		for wind, window_size in enumerate([151, 61, 26]):
			dfbatter['wind' + str(wind) + 'bkrate'] = (dfbatter.y.rolling(window_size).sum() - dfbatter['y']) / (window_size - 1) - dfbatter['batterkrate']
			dfbatter['wind' + str(wind) + 'bkrate'] = dfbatter['wind' + str(wind) + 'bkrate'].fillna(0.0)
	allbatters = pd.concat(batterdfs, axis=0)
	allbatters = allbatters.drop(['y', 'pitcher', 'batter'],axis=1)
	return allbatters





def main():
	dfRegSeason = pd.read_csv(REG_SEASON_ALL)
	dfRegSeason['y'] = np.where(dfRegSeason['descr']=='Strikeout', 1, 0)
	dfPlayers = pd.read_csv(PLAYERS_ALL)
	dfPitchers = pd.read_csv(PITCHERS_ALL)

	dfRegSeason  = dfRegSeason.drop(dfRegSeason.columns[dfRegSeason.columns.str.contains('unnamed',case = False)],axis = 1)
	league_avg = getLeagueAvg(dfRegSeason)

	ballparks = dfRegSeason['stadium'].unique()
	dfRegSeason.bases = dfRegSeason.bases.fillna("E")
	allps = getRatesPitcher(dfRegSeason, league_avg)
	allbs = getRatesBatter(dfRegSeason, league_avg)
	dfRegSeason['index'] = dfRegSeason.index
	#print(dfRegSeason.head(50))
	dfRegSeason = pd.merge(dfRegSeason, allps, how = 'left', left_on = 'index', right_on = 'index')
	dfRegSeason = pd.merge(dfRegSeason, allbs, how = 'left', left_on = 'index', right_on = 'index')


	oneHotstad = pd.get_dummies(dfRegSeason['stadium'])
	oneHotstad = oneHotstad.drop(['Williamsport Little League Classic', 'Sydney Cricket Ground', 'Fort Bragg Field', 'Tokyo Dome'], axis=1, errors='ignore')
	oneHotbase = pd.get_dummies(dfRegSeason['bases'])
	oneHotside = pd.get_dummies(dfRegSeason['side'])
	dfRegSeason['inning'] = np.minimum(dfRegSeason['inning'], 12)
	oneHotinning = pd.get_dummies(dfRegSeason['inning'], prefix = 'inn')
	oneHotouts = pd.get_dummies(dfRegSeason['outs'], prefix = 'outs')
	dfPlayers.bats = dfPlayers.bats.fillna("NA")
	dfPitchers.pitches = dfPitchers.pitches.fillna("NA")
	#print(dfPitchers.head(1))

	dfbatters = pd.merge(dfRegSeason, dfPlayers, how = 'left', left_on = 'batter', right_on = 'player')
	dfboth = pd.merge(dfbatters, dfPitchers, how = 'left', left_on = 'pitcher', right_on = 'player')
	pd.options.display.max_columns = 999
	#print(dfboth.head(1))
	dfboth.pitches = dfboth.pitches.fillna("NA")
	dfboth.throws = dfboth.throws.fillna("NA")
	dfboth.bats.fillna(dfboth.throws, inplace=True)
	dfboth['matchup'] = dfboth['bats'] + dfboth['pitches']
	#print(dfboth.head(1))
	oneHotHand = pd.get_dummies(dfboth['matchup'])
	#print(oneHotHand.head(1))

	dfRegSeasonfeat = pd.concat([dfRegSeason, oneHotstad, oneHotbase, oneHotside, oneHotinning, oneHotouts, oneHotHand], axis=1)
	#print(dfRegSeasonfeat.info)


	dfPFX, LAs = featurizepfx.computepfx(dfRegSeason, year)
	dfRegSeasonfeat['off_score'] = dfRegSeasonfeat['top'] * dfRegSeasonfeat['away_score'] + dfRegSeasonfeat['bottom'] * dfRegSeasonfeat['home_score']
	dfRegSeasonfeat['abhash'] = dfRegSeasonfeat['date'].apply(str) + "_" + dfRegSeasonfeat['pitcher'] + dfRegSeasonfeat['batter'] + dfRegSeasonfeat['inning'].apply(str) + "_" + dfRegSeasonfeat['off_score'].apply(str)
	dfRegSeasonfeat = pd.merge(dfRegSeasonfeat, dfPFX, how='left', left_on = 'abhash', right_on = 'abhash')
	dfRegSeasonfeat['spinvsavg'] = dfRegSeasonfeat.spinvsavg.fillna(0.0)
	dfRegSeasonfeat['windnasty'] = dfRegSeasonfeat.windnasty.fillna(0.0)
	dfRegSeasonfeat['windss'] = dfRegSeasonfeat.windss.fillna(0.0)
	dfRegSeasonfeat['windcs'] = dfRegSeasonfeat.windcs.fillna(0.0)
	dfRegSeasonfeat['windO_swing'] = dfRegSeasonfeat.windO_swing.fillna(0.0)
	dfRegSeasonfeat['windO_swing_batter'] = dfRegSeasonfeat.windO_swing_batter.fillna(0.0)
	dfRegSeasonfeat['windss_batter'] = dfRegSeasonfeat.windss_batter.fillna(0.0)
	dfRegSeasonfeat['windcs_batter'] = dfRegSeasonfeat.windcs_batter.fillna(0.0)
	dfRegSeasonfeat['windfbsspeed'] = dfRegSeasonfeat.windfbsspeed.fillna(0.0)
	dfRegSeasonfeat['windfbespeed'] = dfRegSeasonfeat.windfbespeed.fillna(0.0)
	dfRegSeasonfeat['windfbspin'] = dfRegSeasonfeat.windfbspin.fillna(0.0)
	dfRegSeasonfeat['windfbpfx_x'] = dfRegSeasonfeat.windfbpfx_x.fillna(0.0)
	dfRegSeasonfeat['windfbpfx_z'] = dfRegSeasonfeat.windfbpfx_z.fillna(0.0)




	for key in LAs:
		dfRegSeasonfeat['cum' + key] = dfRegSeasonfeat['cum' + key].fillna(LAs[key])



	
	dfRegSeasonfeat['score'] = ((-1) ** dfRegSeasonfeat['top']) * (dfRegSeasonfeat['home_score'] - dfRegSeasonfeat['away_score'])
	dfRegSeasonfeat =  dfRegSeasonfeat.drop(['abhash','off_score','visitor','home', 'cumbatterk','cumpitcherk','index', 'side', 'bases', 'home_score', 'away_score', 'descr', 'date', 'inning', 'outs'], axis=1)
	dfRegSeasonfeat  = dfRegSeasonfeat.drop(dfRegSeasonfeat.columns[dfRegSeasonfeat.columns.str.contains('unnamed',case = False)],axis = 1)
	#dfRegSeasonfeat = dfRegSeasonfeat.head(1025296)
	print(dfRegSeasonfeat)

	cols = ['y', 'pitcher', 'batter', 'stadium', 'score', 'cumnasty', 'cumss', 'cumcs','cumO_swing', 'cumfbsspeed', 'cumfbespeed', 'cumfbspin', 'cumfbpfx_x', 'cumfbpfx_z','cumss_batter', 'cumcs_batter', 'cumO_swing_batter',  'spinvsavg', 'windnasty', 'windss', 'windcs', 'windO_swing', 'windfbespeed', 'windfbsspeed', 'windfbspin', 'windfbpfx_x', 'windfbpfx_z', 'windO_swing', 'windO_swing_batter', 'windss_batter', 'windcs_batter'] 
	cols += [col for col in dfRegSeasonfeat if not (col in cols)]
	dfRegSeasonfeat = dfRegSeasonfeat[cols]
	print(dfRegSeasonfeat)

	dfRegSeasonfeat.to_csv(DATA_PATH + "RegularSeasonFeatures" + year + ".csv")

	


if __name__=="__main__":
	main()