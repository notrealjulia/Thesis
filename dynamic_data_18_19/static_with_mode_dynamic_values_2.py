# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 17:18:59 2020

@author: JULSP
"""

import pandas as pd
from os.path import dirname
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.mstats import mode


path = dirname(__file__)
dynamic_data_path = "//garbo/Afd-681/05-Technical Knowledge/05-03-Navigational Safety/00 Udviklingsprojekter/01 ML - Missing properties/02 Work/01 IWRAP/ML South Baltic Sea vA/export"

static_data = pd.read_csv(path + "/static_with_ROT_Speed_1464.csv")

example_df =pd.read_csv(path + '/245176000.csv', sep='	', index_col=None, error_bad_lines=False)

#static_data = static_data[:5]
example_df['speed_rounded'] = np.nan
example_df['speed_rounded'] = example_df.iloc[:,3].round(0)

mode_of_mean_speed = example_df.mode()['speed_rounded'][0]

#%%

"""
Getting mode speed and ROT
"""
#For ROT drop 0 values
static_data['Speed_mode'] = np.nan
static_data['ROT_mode'] = np.nan
#static_data['Speed_mode'] = static_data['Speed_mode'].astype(object)

f = lambda x: mode(x, axis=None)[0]

for i in range(len(static_data)):
    try:
        print("getting dynamic data for ship number", i+1)
    #getting mmsi name of the ship from the static data 
        mmsi = static_data.mmsi[i]
    #loading the dynamic file for that specific ship
        dynamic = pd.read_csv(dynamic_data_path +"/{}.csv".format(mmsi), sep='	', index_col=None, error_bad_lines=False)
    #calculating mean, max, std speed for that specific ship and saving it in the static DF
        dynamic['rounded_speed'] = np.nan
        dynamic['rounded_speed'] = dynamic.iloc[:,3].round(0) #rounding up to integers
        #print(dynamic['rounded_speed'][0])
        static_data['Speed_mode'].loc[i] = dynamic.mode()['rounded_speed'][0]
        
        dynamic['rounded_ROT'] = np.nan
        dynamic['rounded_ROT'] = dynamic.iloc[:,7].round(0)
        dynamic = dynamic[(dynamic[['rounded_ROT']] != 0).all(axis=1)] #dropping all 0 values
        static_data['ROT_mode'].loc[i] = dynamic.mode()['rounded_ROT'][0]
        static_data['ROT_median'].loc[i] = dynamic.iloc[:,13].median()
#        static_data['ROT_median'].loc[i] = dynamic.iloc[:,7].median()
#        static_data["ROT_min"].loc[i] = dynamic.iloc[:,7].min()
#        static_data["Speed_median"].loc[i] = dynamic.iloc[:,3].median()
    except:
        print("couldn't get data on", i+1)