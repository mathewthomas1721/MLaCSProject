import numpy as np
import pandas as pd
import scipy as sp
import sys
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
import datetime


DATA_PATH = "../Data/"
WRITE_PATH= "../Data/OutData/"
FIG_PATH = "../Figs/"
#POST_SEASON_ALL = DATA_PATH + "AtBats_PostSeason_2012-2017_sorted.csv"
REG_SEASON_PFX = DATA_PATH + "MLB_PitchFX_2012/MLB_PitchFX_RegularSeason_2012_sorted.csv"
REG_SEASON_AB = DATA_PATH + "MLB_AtBats_RegularSeason_2012_sorted.csv"
#SPRING_TRN_ALL = DATA_PATH + "AtBats_SpringTraining_2012-2017_sorted.csv"
PFXN = 25.0

def getLeagueAvg(df, trait):
	tot_k = df[trait].sum()
	tot_pa = df.shape[0]
	return  1.0 * tot_k / tot_pa


def accumulate_Pitcher(df, lavg):
	dfslim = df[['pitcher', 'nasty', 'ss', 'cs']]
	pitchers = dfslim.pitcher.unique()
	pitcherdfs = []
	for i,p in enumerate(pitchers):
		print(i)
		dfp = dfslim[dfslim.pitcher == p]
		dfp = dfp.reset_index()
		dfp['cumnasty'] = (dfp['nasty'].expanding().sum() + lavg['nasty'] * PFXN  - dfp['nasty']) / (dfp.index + PFXN)
		dfp['cumnasty'] = dfp.cumnasty.fillna(0)
		dfp['cumss'] = (dfp['ss'].expanding().sum() + lavg['ss'] * PFXN  - dfp['ss']) / (dfp.index + PFXN)
		dfp['cumss'] = dfp.cumss.fillna(0)
		dfp['cumcs'] = (dfp['cs'].expanding().sum() + lavg['cs'] * PFXN  - dfp['cs']) / (dfp.index + PFXN)
		dfp['cumcs'] = dfp.cumcs.fillna(0)

		pitcherdfs.append(dfp)


	allpitchers = pd.concat(pitcherdfs, axis=0)
	allpitchers = allpitchers.drop(['pitcher', 'nasty', 'ss', 'cs'],axis=1)
	return allpitchers


def computepfx(dfAtBat):
	dfPFX = pd.read_csv(REG_SEASON_PFX)
	dfPFX  = dfPFX.drop(dfPFX.columns[dfPFX.columns.str.contains('unnamed',case = False)],axis = 1)
	#
	#dfPFX = dfPFX.head(50)
	#

	dfPFX['date'] = dfPFX.date.apply(lambda dt: "20" + datetime.datetime.strptime(dt, '%m/%d/%y').strftime('%y-%m-%d'))


	dfPFX['ss'] = np.where(dfPFX['descr']=='Swinging Strike', 1, 0) + np.where(dfPFX['descr']=='Foul Tip', 1, 0)
	dfPFX['cs'] = np.where(dfPFX['descr']=='Called Strike', 1, 0)

	League_Avgs = {}
	League_Avgs['ss'] =  getLeagueAvg(dfPFX, 'ss')
	League_Avgs['cs'] = getLeagueAvg(dfPFX, 'cs')
	League_Avgs['nasty'] = getLeagueAvg(dfPFX, 'nasty')

	dfAtBat = pd.read_csv(REG_SEASON_AB)

	dfAtBat = pd.concat([dfAtBat, pd.get_dummies(dfAtBat['side'])], axis=1)
	dfAtBat['off_score'] = dfAtBat['top'] * dfAtBat['away_score'] + dfAtBat['bottom'] * dfAtBat['home_score']

	dfAtBat['abhash'] = dfAtBat['date'].apply(str) + "_" + dfAtBat['pitcher'] + dfAtBat['batter'] + dfAtBat['inning'].apply(str) + "_" + dfAtBat['off_score'].apply(str)
	dfPFX['abhash'] = dfPFX['date'].apply(str) + "_" +  dfPFX['pitcher'] + dfPFX['batter'] + dfPFX['inning'].apply(str) + "_" + dfPFX['offense_score'].apply(str)


	dfPFX['index'] = dfPFX.index
	allps = accumulate_Pitcher(dfPFX, League_Avgs)
	dfPFX = pd.merge(dfPFX, allps, how = 'left', left_on = 'index', right_on = 'index')

	dfPFX = dfPFX[~dfPFX.abhash.duplicated(keep='first')]
	dfPFX = dfPFX[['abhash', 'cumnasty', 'cumss', 'cumcs']]
	print(dfPFX)
	return dfPFX, League_Avgs


if __name__=="__main__":
	computepfx()