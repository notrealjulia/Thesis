# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 19:19:01 2020

@author: JULSP

Comapring ROT to ship lenght 
"""

import pandas as pd
from os.path import dirname
import matplotlib.pyplot as plt
import numpy as np

path = dirname(__file__)
dynamic_data_path = "//garbo/Afd-681/05-Technical Knowledge/05-03-Navigational Safety/00 Udviklingsprojekter/01 ML - Missing properties/02 Work/01 IWRAP/ML South Baltic Sea vA/export"

static_data_full = pd.read_csv(path + "/static_with_mean_speed.csv")

#%%

"""
Getting ROT values from dynamic data
"""

static_data_full['ROT_mean'] = np.nan
static_data_full['ROT_max'] = np.nan


for i in range(len(static_data_full)):
    try:
        print("getting dynamic data for ship number", i+1)
    #getting mmsi name of the ship from the static data 
        mmsi = static_data_full.mmsi[i]
    #loading the dynamic file for that specific ship
        dynamic = pd.read_csv(dynamic_data_path +"/{}.csv".format(mmsi), sep='	', index_col=None, error_bad_lines=False)
    #calculating mean, max, std speed for that specific ship and saving it in the static DF
        static_data_full['ROT_mean'].loc[i] = dynamic.iloc[:,7].mean()
        static_data_full["ROT_max"].loc[i] = dynamic.iloc[:,7].max()
    except:
        print("couldn't get data on", i+1)
        
#%%

cargo = static_data_full.loc[static_data_full.iloc[:,6] == "Cargo"]
tanker = static_data_full.loc[static_data_full.iloc[:,6] == "Tanker"]
fishing = static_data_full.loc[static_data_full.iloc[:,6] == "Fishing"]

#%%

print(len(cargo.loc[cargo.iloc[:,25] == 0])/len(cargo), "% of all cargo ships have no value for Rate of Turn")
print(len(tanker.loc[tanker.iloc[:,25] == 0])/len(tanker), "% of all tanker ships have no value for Rate of Turn")
print(len(fishing.loc[fishing.iloc[:,25] == 0])/len(fishing), "% of all fishing ships have no value for Rate of Turn")

Cargo_ROT = cargo.loc[cargo.iloc[:,25] != 0]

Vessels_ROT = static_data_full.loc[static_data_full.iloc[:,25] != 0]

print(len(Vessels_ROT)/len(static_data_full), "% of all selected vessels have a ROT value")

#%%
ax = Vessels_ROT.plot.scatter(x = 'length_from_data_set', y = 'ROT_max', alpha=0.3)        
ax.set_xlabel("length of vessel")
ax.set_ylabel("Max ROT of vessel")

#%%

Vessels_ROT = Vessels_ROT.reset_index(drop = True)

Vessels_ROT.to_csv(path+'static_with_mean_maxROT.csv', encoding ='utf-8', index=False)
