# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 16:26:13 2020

@author: KORAL
"""

"""

Extracting dynamic data for September 2018 (timestamp, LAT, LON, COG and SOG values) 
for all ACTIVE vessels that have => 5000 signals and merging them into one file

"""
import os
import glob
import pandas as pd 
from tqdm import tqdm
import time
from datetime import datetime



static = pd.read_csv(r'C:\Users\KORAL\OneDrive - Ramboll\Documents\GitHub\Thesis\Thesis\dynamic_data_18_19\Trips_and_VesselStatus_Static.csv', sep = '\t') 
dynamic_path= r"C:\Users\KORAL\OneDrive - Ramboll\Documents\GitHub\Thesis\Thesis\dynamic_data_18_19\export"'\\'

activeDF = static.loc[static['status'] == 'active']
activeDF = activeDF.loc[activeDF['signals'] >= 5000]
activeDF = activeDF.loc[activeDF['length_from_data_set'] <= 400]
mmsis = [str(mmsi) for mmsi in activeDF['mmsi']]

#%%

activeVessels  = []
for mmsi in tqdm(mmsis):
    df  = pd.read_csv(dynamic_path + mmsi + '.csv', usecols=["timestamp", "lat [deg]", "lon [deg]", "sog [kn]", "cog [deg]"], sep = '\t')
    
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df[df['timestamp'].dt.month == 9]
        df['mmsi'] = mmsi
        activeVessels.append(df)
    except:
        print (str(mmsi), ' No September')
    
    
activeVesselsDF = pd.concat(activeVessels)
activeVesselsDF.to_parquet('ActiveDynamicALL.parquet')


