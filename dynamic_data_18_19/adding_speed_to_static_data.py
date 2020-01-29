# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:27:49 2020

@author: JULSP
"""


import pandas as pd
from os.path import dirname
import matplotlib.pyplot as plt
import numpy as np

path = dirname(__file__)
static_data_full = pd.read_csv(path + "/static_ship_data.csv", sep='	', index_col=None, error_bad_lines=False)

static_small = static_data_full[:5]
static_small['mean_speed'] = np.nan
static_small['top_speed'] = np.nan
static_small['std_speed'] = np.nan

dynamic_data_path = "//garbo/Afd-681/05-Technical Knowledge/05-03-Navigational Safety/00 Udviklingsprojekter/01 ML - Missing properties/02 Work/01 IWRAP/ML South Baltic Sea vA/export"


#%%
"""
For testing stuff
"""
#test_csv_load = pd.read_csv(dynamic_data_path + "/477102200.csv", sep='	', index_col=None, error_bad_lines=False)

#%%
"""
Adding data when it is local
"""

for i in range(len(static_small)):
    #getting mmsi name of the ship from the static data 
    mmsi = static_small.mmsi[i]
    #loading the dynamic file for that specific ship
    dynamic = pd.read_csv(path +"/dynamic_sample/{}.csv".format(mmsi), sep='	', index_col=None, error_bad_lines=False)
    #calculating mean, max, std speed for that specific ship and saving it in the static DF
    static_small['mean_speed'].loc[i] = dynamic.iloc[:,3].mean()
    static_small["top_speed"].loc[i] = dynamic.iloc[:,3].max()
    static_small["std_speed"].loc[i] = dynamic.iloc[:,3].std()
    
#%%
"""
Getting a sample to work with
rename the variable before running
""" 
types = static_data_full.iloc[:,5].unique()

sample_static3 = static_data_full.loc[(static_data_full.iloc[:,5] == 'General cargo ship') | (static_data_full.iloc[:,5] == 'Oil products tanker') | (static_data_full.iloc[:,5] == 'Fishing ship')]

sample_static3['mean_speed'] = np.nan
sample_static3['top_speed'] = np.nan
sample_static3['std_speed'] = np.nan

sample_static3 = sample_static3[1000:]

sample_static3 = sample_static3.reset_index(drop =True)

#%%
"""
Remote data - dies after 3
"""

for i in range(len(sample_static3)):
    try:
        print("getting dynamic data for ship number", i+1)
    #getting mmsi name of the ship from the static data 
        mmsi = sample_static3.mmsi[i]
    #loading the dynamic file for that specific ship
        dynamic = pd.read_csv(dynamic_data_path +"/{}.csv".format(mmsi), sep='	', index_col=None, error_bad_lines=False)
    #calculating mean, max, std speed for that specific ship and saving it in the static DF
        sample_static3['mean_speed'].loc[i] = dynamic.iloc[:,3].mean()
        sample_static3["top_speed"].loc[i] = dynamic.iloc[:,3].max()
        sample_static3["std_speed"].loc[i] = dynamic.iloc[:,3].std()
    except:
        print("couldn't get data on", i)
    
#%%
frames = [sample_static, sample_static2, sample_static3]
        
static_final = pd.concat(frames, ignore_index = True)

static_final.to_csv('static_with_mean_speed.csv', encoding ='utf-8', index=False)
#ToDo calculate averages per types of ships

#%%
static_final.plot.scatter(y="mean_speed", x="length_from_data_set", alpha = 0.3)

static_final.plot.scatter(y="top_speed", x="length_from_data_set", alpha = 0.3)

static_final.plot.scatter(y="std_speed", x="length_from_data_set", alpha = 0.3)

static_final.plot.scatter(y="length_from_data_set", x="width", alpha = 0.3)
