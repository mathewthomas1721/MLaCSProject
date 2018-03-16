import pandas as pd
import numpy as np

def cleanAtBatsFile(fName):
  f = open(fName+".csv",'r')
  i = 0
  g = open(fName+"_update.csv",'w')
  for line in f:

      if '"date","stadium","pitcher","batter","outs","home_score","away_score","descr","inning","side","bases"' not in line and "date,stadium,pitcher,batter,outs,home_score,away_score,descr,inning,side,bases" not in line:
          g.write(line.replace('"', ""))
      else:
          #print(line)
          if i == 0:
               g.write(line.replace('"', ""))
      i = i+1

def cleanPitchFXFile(fName):
  f = open(fName+".csv",'r')
  i = 0
  g = open(fName+"_update.csv",'w')
  for line in f:

      if "date,stadium,inning,side,pitcher,pitch_count,batter,balls,strikes,ay,px,ax,sz_bot,vz0,vy0,pfx_x,type_confidence,z0,tfs,pz,start_speed,az,zone,break_angle,spin_dir,end_speed,vx0,sz_top,nasty,descr,pfx_z,break_y,pitch_type,tfs_zulu,x,spin_rate,y0,break_length,y,x0,on_1b,on_2b,on_3b,umpcall,outcome,offense_score,defense_score" not in line :
          g.write(line)
      else:
          #print(line)
          if i == 0:
              g.write(line)
      i = i+1

def checkPitchFXFile(fName):
    f = open(fName+".csv",'r')

    for line in f:

        if "date,stadium,inning,side,pitcher,pitch_count,batter,balls,strikes,ay,px,ax,sz_bot,vz0,vy0,pfx_x,type_confidence,z0,tfs,pz,start_speed,az,zone,break_angle,spin_dir,end_speed,vx0,sz_top,nasty,descr,pfx_z,break_y,pitch_type,tfs_zulu,x,spin_rate,y0,break_length,y,x0,on_1b,on_2b,on_3b,umpcall,outcome,offense_score,defense_score" in line :
            print(line)

def checkAtBatsFile(fName):
    f = open(fName+".csv",'r')

    for line in f:
        if '"date","stadium","pitcher","batter","outs","home_score","away_score","descr","inning","side","bases"' in line or "date,stadium,pitcher,batter,outs,home_score,away_score,descr,inning,side,bases" in line:
            print(line)


cleanAtBatsFile("AtBats_PostSeason_2012-2017")
cleanAtBatsFile("AtBats_RegularSeason_2012-2017")
cleanAtBatsFile("AtBats_SpringTraining_2012-2017")
cleanPitchFXFile("PitchFX_PostSeason_2012-2017")
cleanPitchFXFile("PitchFX_RegularSeason_2012-2017")
cleanPitchFXFile("PitchFX_SpringTraining_2012-2017")


checkAtBatsFile("AtBats_PostSeason_2012-2017_update")
checkAtBatsFile("AtBats_RegularSeason_2012-2017_update")
checkAtBatsFile("AtBats_SpringTraining_2012-2017_update")
checkPitchFXFile("PitchFX_PostSeason_2012-2017_update")
checkPitchFXFile("PitchFX_RegularSeason_2012-2017_update")
checkPitchFXFile("PitchFX_SpringTraining_2012-2017_update")
'''
#pd.set_option('display.height', 500)
pd.set_option('display.max_rows', 40)
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 500)
df = pd.read_csv('PitchFX_SpringTraining_2012-2016_update.csv')
#df['date'] = df['date'].astype('datetime64[ns]')
print(df.dtypes)
'''
