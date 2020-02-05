# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 12:35:07 2020
FAILED
@author: JULSP
"""

import pandas as pd
from os.path import dirname
import matplotlib.pyplot as plt
import numpy as np

path = dirname(__file__)
dynamic_data_path = "//garbo/Afd-681/05-Technical Knowledge/05-03-Navigational Safety/00 Udviklingsprojekter/01 ML - Missing properties/02 Work/01 IWRAP/ML South Baltic Sea vA/export"

#static_data_full = pd.read_csv(path + "/static_ship_data.csv", sep='	', index_col=None, error_bad_lines=False)

static_data_full = pd.read_csv(path + "/static_with_mean_speed.csv")

#static_small = static_data_full[:5]
#static_small['ROT'] = np.nan
#static_small['COG'] = np.nan
#static_small['Heading'] = np.nan
#static_small['Timestamps'] = np.nan

#to add values that are in a list
#static_small['COG'] = static_small['COG'].astype(object)
#static_small['ROT'] = static_small['ROT'].astype(object)
#static_small['Heading'] = static_small['Heading'].astype(object)
#static_small['Timestamps'] = static_small['Timestamps'].astype(object)

#%%
#Adding dynamic Orientation values to the entire dataset

static_data_full['ROT'] = np.nan
static_data_full['COG'] = np.nan
static_data_full['Heading'] = np.nan
static_data_full['Timestamps'] = np.nan
static_data_full['Speed'] = np.nan

#to add values that are in a list need to do this 
static_data_full['COG'] = static_data_full['COG'].astype(object)
static_data_full['ROT'] = static_data_full['ROT'].astype(object)
static_data_full['Heading'] = static_data_full['Heading'].astype(object)
static_data_full['Timestamps'] = static_data_full['Timestamps'].astype(object)
static_data_full['Speed'] = static_data_full['Speed'].astype(object)

for i in range(len(static_data_full)):
    try:
        print("getting dynamic data for ship number", i+1)
    #getting mmsi name of the ship from the static data 
        mmsi = static_data_full.mmsi[i]
    #loading the dynamic file for that specific ship based on the mmsi number
        dynamic = pd.read_csv(dynamic_data_path +"/{}.csv".format(mmsi), sep='	', index_col=None, error_bad_lines=False)
    #calculating mean, max, std speed for that specific ship and saving it in the static DF
        static_data_full.at[i,'ROT'] = list(dynamic.iloc[:,7])
        static_data_full.at[i,'Heading'] = list(dynamic.iloc[:,5])
        static_data_full.at[i,'COG'] = list(dynamic.iloc[:,6])
        static_data_full.at[i,'Timestamps'] = list(dynamic.iloc[:,0])
        static_data_full.at[i,'Speed'] = list(dynamic.iloc[:,3])
    except:
        print("couldn't get data on", i)
#%%

static_data_full.to_csv('static_with_rot.csv', encoding ='utf-8', index=False)

static_small_2 = static_data_full[:5]        

#%%
#For testing things - don't run
dynamic_test = pd.read_csv(path + "/245176000.csv", sep='	', index_col=None, error_bad_lines=False)

static_small['COG'] = static_small['COG'].astype(object)
static_small['ROT'] = static_small['ROT'].astype(object)
static_small['Heading'] = static_small['Heading'].astype(object)

print(dynamic_test.iloc[:,5]) #heading
print(list(dynamic_test.iloc[:,6])) #COG

this_list =list(dynamic_test.iloc[:,6])
rot_list = list(dynamic_test.iloc[:,7])
static_small.at[0,'COG'] = this_list
static_small.at[0,'ROT'] = list(dynamic_test.iloc[:,7])
#%%
#Testing on a smaller sample
for i in range(len(static_small)):
    #getting mmsi name of the ship from the static data 
    mmsi = static_small.mmsi[i]
    #loading the dynamic file for that specific ship
    dynamic = pd.read_csv(path +"/dynamic_sample/{}.csv".format(mmsi), sep='	', index_col=None, error_bad_lines=False)
    #calculating mean, max, std speed for that specific ship and saving it in the static DF
    static_small.at[i,'ROT'] = list(dynamic_test.iloc[:,7])
    static_small.at[i,'Heading'] = list(dynamic_test.iloc[:,5])
    static_small.at[i,'COG'] = list(dynamic_test.iloc[:,6])
    static_small.at[i,'Timestamps'] = list(dynamic_test.iloc[:,0])