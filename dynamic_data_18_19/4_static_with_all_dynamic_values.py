# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 10:39:07 2020

@author: JULSP

Adding mode ROT and mode Speed to small subsection of vessels that have ROT values
"""


import pandas as pd
from os.path import dirname
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.mstats import mode


path = dirname(__file__)
dynamic_data_path = "//garbo/Afd-681/05-Technical Knowledge/05-03-Navigational Safety/00 Udviklingsprojekter/01 ML - Missing properties/02 Work/01 IWRAP/ML South Baltic Sea vA/export"

static_data = pd.read_csv(path + "/static_with_mean_maxROT.csv")

mode_ship_length = static_data["mean_speed"].mode()

#%%
"""
Getting 
mode speed 
mode ROT
min ROT
"""
static_data['ROT_median'] = np.nan
static_data['ROT_min'] = np.nan
static_data['Speed_median'] = np.nan

#static_data['ROT_mode'] = static_data['ROT_mode'].astype(object)
#static_data['Speed_mode'] = static_data['Speed_mode'].astype(object)

for i in range(len(static_data)):
    try:
        print("getting dynamic data for ship number", i+1)
    #getting mmsi name of the ship from the static data 
        mmsi = static_data.mmsi[i]
    #loading the dynamic file for that specific ship
        dynamic = pd.read_csv(dynamic_data_path +"/{}.csv".format(mmsi), sep='	', index_col=None, error_bad_lines=False)
    #calculating mean, max, std speed for that specific ship and saving it in the static DF
        static_data['ROT_median'].loc[i] = dynamic.iloc[:,7].median()
        static_data["ROT_min"].loc[i] = dynamic.iloc[:,7].min()
        static_data["Speed_median"].loc[i] = dynamic.iloc[:,3].median()
    except:
        print("couldn't get data on", i+1)
        
#%%
        
"""
Making a correlation matrix
"""
import seaborn as sn

static_data['length/width'] = static_data['length_from_data_set'] / static_data['width']

list_of_columns = [9,15,29,16,17,18,19,21,28,22,23,24,26,25,27]
correlation_df = static_data.iloc[:,list_of_columns]

correlation_df.rename({'length_from_data_set':'vessel_length', 'width':'vessel_width', 'mean_speed':'Speed_mean', 'top_speed':'Speed_max','std_speed':'Speed_std'}, axis = 1, inplace = True)

corrMatrix = correlation_df.corr()
sn.heatmap(corrMatrix,center = 0, annot=True)

static_data.to_csv(path+'static_with_ROT_Speed_1464.csv', encoding ='utf-8', index=False)
