# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:49:43 2020

@author: KORAL
"""



# ALL vessels 

from datetime import datetime
import pandas as pd
from os.path import dirname
import matplotlib.pyplot as plt
import numpy as np

path = dirname(__file__)
dynamic_path= "//garbo/Afd-681/05-Technical Knowledge/05-03-Navigational Safety/00 Udviklingsprojekter/01 ML - Missing properties/02 Work/01 IWRAP/ML South Baltic Sea vA/export"
dynamic = pd.read_csv(dynamic_path + "/245176000.csv", sep='	', index_col=None, error_bad_lines=False)
static = pd.read_csv(r"C:\Users\KORAL\OneDrive - Ramboll\Documents\GitHub\Thesis\Thesis\Static Data 2018-2019/static_ship_data.csv", sep='	', index_col=None, error_bad_lines=False)
static['trips'] = np.nan
dictionary = static.groupby('iwrap_type_from_dataset')['mmsi'].apply(lambda g: g.values.tolist()).to_dict()

for mmsi in dictionary.get('Fast ferry'): 
    
    try:
        dynamic = pd.read_csv(dynamic_path +"/{}.csv".format(str(mmsi)), sep='	', index_col=None, error_bad_lines=False)    
        idleTimes = []
        stops = []
        trips = 0
        
                   
        for i, speed in enumerate(dynamic['sog [kn]']):
            try:
                if speed < 0.1:
                    stops.append(dynamic['timestamp'].iloc[i])
                        
                elif speed > 0.1:
                    firstStop = datetime.strptime(stops[0], "%Y-%m-%dT%H:%M:%S")
                    lastStop = datetime.strptime(stops[-1], "%Y-%m-%dT%H:%M:%S")
                    idleTime = (lastStop - firstStop)
                    idleTimes.append(idleTime)
                    stops = []
            except:
                pass
        
        for time in idleTimes:
            if time.total_seconds() > 900:
                
                trips += 1

        static['trips'][(static['mmsi'] == mmsi)] = trips  
    except MemoryError as error:
        memoryErrors.append(error)    
#%%

### TESTING 
print(static.loc[static['mmsi'] == 235101881])
dynamic = pd.read_csv(dynamic_path + "/235101881.csv", sep='	', index_col=None, error_bad_lines=False)
plt.plot(dynamic['lat [deg]'], dynamic['lon [deg]'])

#%%

# 1 VESSEL 
idleTimes = []
stops = []
trips = 0
static['trips'] = np.nan
          
for i, speed in enumerate(dynamic['sog [kn]']):
    try:
        if speed < 0.1:
            stops.append(dynamic['timestamp'].iloc[i])
                
        elif speed > 0.1:
            firstStop = datetime.strptime(stops[0], "%Y-%m-%dT%H:%M:%S")
            lastStop = datetime.strptime(stops[-1], "%Y-%m-%dT%H:%M:%S")
            idleTime = (lastStop - firstStop)
            idleTimes.append(idleTime)
            
            stops = []
    except:
        pass

for time in idleTimes:
    if time.total_seconds() > 1800:
        print(time)
        
        trips += 1

static['trips'][(static['mmsi'] == mmsi)] = trips      
