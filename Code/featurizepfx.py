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
#SPRING_TRN_ALL = DATA_PATH + "AtBats_SpringTraining_2012-2017_sorted.csv"
PFXN = 25.0

def getLeagueAvg(df, trait):
	tot_k = df[trait].sum()
	tot_pa = df.shape[0]
	return  1.0 * tot_k / tot_pa


def accumulate_Pitcher(df, lavg):
	dfslim = df[['pitcher', 'nasty', 'ss', 'cs', 'Ozone_swing']]
	pitchers = dfslim.pitcher.unique()
	pitcherdfs = []
	for i,p in enumerate(pitchers):
		print(i)
		dfp = dfslim[dfslim.pitcher == p]
		dfp = dfp.reset_index()
		dfp['cumnasty'] = (dfp['nasty'].expanding().sum() + lavg['nasty'] * PFXN  - dfp['nasty']) / (dfp.index + PFXN)
		dfp['cumss'] = (dfp['ss'].expanding().sum() + lavg['ss'] * PFXN  - dfp['ss']) / (dfp.index + PFXN)
		dfp['cumcs'] = (dfp['cs'].expanding().sum() + lavg['cs'] * PFXN  - dfp['cs']) / (dfp.index + PFXN)
		dfp['cumO_swing'] = (dfp['Ozone_swing'].expanding().sum() + lavg['O_swing'] * PFXN  - dfp['Ozone_swing']) / (dfp.index + PFXN)

		dfp['windnasty'] = (dfp['nasty'].rolling(51).sum() - dfp['nasty']) / (50.0) - dfp['cumnasty']
		dfp['windss'] = (dfp['ss'].rolling(51).sum() - dfp['ss']) / (50.0) - dfp['cumss']
		dfp['windcs'] = (dfp['cs'].rolling(51).sum() - dfp['cs']) / (50.0) - dfp['cumcs']
		dfp['windO_swing'] = (dfp['Ozone_swing'].rolling(51).sum() - dfp['Ozone_swing']) / (50.0) - dfp['cumO_swing']

		pitcherdfs.append(dfp)


	allpitchers = pd.concat(pitcherdfs, axis=0)
	allpitchers = allpitchers.drop(['pitcher', 'nasty', 'ss', 'cs', 'Ozone_swing'],axis=1)
	return allpitchers


def accumulate_Batter(df, lavg):
	dfslim = df[['batter', 'ss', 'cs', 'Ozone_swing']]
	batters = dfslim.batter.unique()
	batterdfs = []
	for i,b in enumerate(batters):
		dfb = dfslim[dfslim.batter == b]
		dfb = dfb.reset_index()

		dfb['cumss_batter'] = (dfb['ss'].expanding().sum() + lavg['ss'] * PFXN - dfb['ss']) / (dfb.index + PFXN)
		dfb['cumcs_batter'] = (dfb['cs'].expanding().sum() + lavg['cs'] * PFXN - dfb['cs']) / (dfb.index + PFXN)
		dfb['cumO_swing_batter'] = (dfb['Ozone_swing'].expanding().sum() + lavg['O_swing'] * PFXN - dfb['Ozone_swing']) / (dfb.index + PFXN)

		dfb['windss_batter'] = (dfb['ss'].rolling(51).sum() - dfb['ss']) / (50.0) - dfb['cumss_batter']
		dfb['windcs_batter'] = (dfb['cs'].rolling(51).sum() - dfb['cs']) / (50.0) - dfb['cumcs_batter']
		dfb['windO_swing_batter'] = (dfb['Ozone_swing'].rolling(51).sum() - dfb['Ozone_swing']) / (50.0) - dfb['cumO_swing_batter']

		batterdfs.append(dfb)

	allbatters = pd.concat(batterdfs, axis=0)
	allbatters = allbatters.drop(['batter', 'ss', 'cs', 'Ozone_swing'], axis=1)
	return allbatters




def accumulate_fastball_data(df, lavg):
	dffb = df[df.pitch_type.isin(['FF', 'FT', 'SI'])]
	dffb = dffb[['pitcher', 'start_speed', 'end_speed', 'spin_rate', 'pfx_z', 'pfx_x']]

	pitchers = dffb.pitcher.unique()
	pitcherdfs = []
	for i,p in enumerate(pitchers):
		print(i)
		dfp = dffb[dffb.pitcher == p]
		dfp = dfp.reset_index()
		dfp['cumfbsspeed'] = (dfp['start_speed'].expanding().sum() - dfp['start_speed']) / dfp.index
		dfp['cumfbespeed'] = (dfp['end_speed'].expanding().sum() - dfp['end_speed']) / dfp.index
		dfp['cumfbspin'] = (dfp['spin_rate'].expanding().sum() - dfp['spin_rate']) / dfp.index
		dfp['cumfbpfx_z'] = (dfp['pfx_z'].expanding().sum() - dfp['pfx_z']) / dfp.index
		dfp['cumfbpfx_x'] = (dfp['pfx_x'].expanding().sum() - dfp['pfx_x']) / dfp.index
		dfp['spinvsavg'] = abs(dfp['cumfbspin'] - lavg['fbspin'])

		dfp['windfbsspeed'] = (dfp['start_speed'].rolling(21).sum() - dfp['start_speed']) / (20.0) - dfp['cumfbsspeed']
		dfp['windfbespeed'] = (dfp['end_speed'].rolling(21).sum() - dfp['end_speed']) / (20.0) - dfp['cumfbespeed']
		dfp['windfbspin'] = (dfp['spin_rate'].rolling(21).sum() - dfp['spin_rate']) / (20.0) - dfp['cumfbspin']
		dfp['windfbpfx_x'] = (dfp['pfx_x'].rolling(21).sum() - dfp['pfx_x']) / (20.0) - dfp['cumfbpfx_x']
		dfp['windfbpfx_z'] = (dfp['pfx_z'].rolling(21).sum() - dfp['pfx_z']) / (20.0) - dfp['cumfbpfx_z']


		pitcherdfs.append(dfp)

	allpitchers = pd.concat(pitcherdfs, axis = 0)
	allpitchers = allpitchers.drop(['pitcher', 'start_speed', 'end_speed', 'spin_rate', 'pfx_x', 'pfx_z'],axis=1)
	return allpitchers


def computepfx(dfAtBat, year):

	REG_SEASON_PFX = DATA_PATH + "MLB_PitchFX_" + year +"/MLB_PitchFX_RegularSeason_" + year +"_sorted.csv"
	REG_SEASON_AB = DATA_PATH + "MLB_AtBats_RegularSeason_" + year +"_sorted.csv"

	dfPFX = pd.read_csv(REG_SEASON_PFX)
	dfPFX  = dfPFX.drop(dfPFX.columns[dfPFX.columns.str.contains('unnamed',case = False)],axis = 1)
	#
	#dfPFX = dfPFX.head(50)
	#

	#dfPFX['date'] = dfPFX.date.apply(lambda dt: "20" + datetime.datetime.strptime(dt, '%m/%d/%y').strftime('%y-%m-%d'))


	dfPFX['ss'] = np.where(dfPFX['descr']=='Swinging Strike', 1, 0) + np.where(dfPFX['descr']=='Foul Tip', 1, 0)
	dfPFX['cs'] = np.where(dfPFX['descr']=='Called Strike', 1, 0)
	dfPFX['zone'] = (dfPFX['pz'] > dfPFX['sz_bot']) & (dfPFX['px'] > -.7084) & (dfPFX['px'] < .7084) & (dfPFX['pz'] < dfPFX['sz_top'])
	dfPFX['Ozone_swing'] = ~dfPFX['zone'] & (dfPFX['umpcall'] != 'B')

	League_Avgs = {}
	League_Avgs['ss'] =  getLeagueAvg(dfPFX, 'ss')
	League_Avgs['ss_batter'] = League_Avgs['ss']
	League_Avgs['cs'] = getLeagueAvg(dfPFX, 'cs')
	League_Avgs['cs_batter'] = League_Avgs['cs']
	League_Avgs['nasty'] = getLeagueAvg(dfPFX, 'nasty')
	League_Avgs['O_swing'] = getLeagueAvg(dfPFX, 'Ozone_swing')
	League_Avgs['O_swing_batter'] = League_Avgs['O_swing']
	dffb = dfPFX[dfPFX.pitch_type.isin(['FF', 'FT', 'SI'])]
	League_Avgs['fbsspeed'] = getLeagueAvg(dffb, 'start_speed')
	League_Avgs['fbespeed'] = getLeagueAvg(dffb, 'end_speed')
	League_Avgs['fbspin'] = getLeagueAvg(dffb, 'spin_rate')
	League_Avgs['fbpfx_z'] = getLeagueAvg(dffb, 'pfx_z')
	League_Avgs['fbpfx_x'] = getLeagueAvg(dffb, 'pfx_x')

	dfAtBat = pd.read_csv(REG_SEASON_AB)

	dfAtBat = pd.concat([dfAtBat, pd.get_dummies(dfAtBat['side'])], axis=1)
	dfAtBat['off_score'] = dfAtBat['top'] * dfAtBat['away_score'] + dfAtBat['bottom'] * dfAtBat['home_score']

	dfAtBat['abhash'] = dfAtBat['date'].apply(str) + "_" + dfAtBat['pitcher'] + dfAtBat['batter'] + dfAtBat['inning'].apply(str) + "_" + dfAtBat['off_score'].apply(str)
	dfPFX['abhash'] = dfPFX['date'].apply(str) + "_" +  dfPFX['pitcher'] + dfPFX['batter'] + dfPFX['inning'].apply(str) + "_" + dfPFX['offense_score'].apply(str)


	dfPFX['index'] = dfPFX.index
	allps = accumulate_Pitcher(dfPFX, League_Avgs)
	dfPFX = pd.merge(dfPFX, allps, how = 'left', left_on = 'index', right_on = 'index')


	allps = accumulate_fastball_data(dfPFX, League_Avgs)
	dfPFX = pd.merge(dfPFX, allps, how = 'left', left_on = 'index', right_on = 'index')

	allbs = accumulate_Batter(dfPFX, League_Avgs)
	dfPFX = pd.merge(dfPFX, allbs, how = 'left', left_on = 'index', right_on = 'index')

	dfPFX['windfbspin'] = dfPFX.windfbspin.fillna(method='ffill')
	dfPFX['windfbsspeed'] = dfPFX.windfbsspeed.fillna(method='ffill')
	dfPFX['windfbespeed'] = dfPFX.windfbespeed.fillna(method='ffill')
	dfPFX['windfbpfx_z'] = dfPFX.windfbpfx_z.fillna(method='ffill')
	dfPFX['windfbpfx_x'] = dfPFX.windfbpfx_x.fillna(method='ffill')


	dfPFX['cumfbspin'] = dfPFX.cumfbspin.fillna(method='ffill')
	dfPFX['cumfbsspeed'] = dfPFX.cumfbsspeed.fillna(method='ffill')
	dfPFX['cumfbespeed'] = dfPFX.cumfbespeed.fillna(method='ffill')
	dfPFX['cumfbpfx_z'] = dfPFX.cumfbpfx_z.fillna(method='ffill')
	dfPFX['cumfbpfx_x'] = dfPFX.cumfbpfx_x.fillna(method='ffill')




	dfPFX = dfPFX[~dfPFX.abhash.duplicated(keep='first')]
	dfPFX = dfPFX[['abhash', 'cumnasty', 'cumss', 'cumcs','cumO_swing', 'cumfbsspeed', 'cumfbespeed', 'cumfbspin', 'cumfbpfx_x', 'cumfbpfx_z','cumss_batter', 'cumcs_batter', 'cumO_swing_batter',  'spinvsavg', 'windnasty', 'windss', 'windcs', 'windO_swing', 'windfbespeed', 'windfbsspeed', 'windfbspin', 'windfbpfx_x', 'windfbpfx_z', 'windO_swing', 'windO_swing_batter', 'windss_batter', 'windcs_batter']]
	print(dfPFX)
	return dfPFX, League_Avgs


if __name__=="__main__":
	computepfx()