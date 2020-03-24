# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 09:39:25 2020

@author: KORAL
"""

import numpy as np
import pandas as pd



active = pd.read_parquet(r'C:\Users\KORAL\OneDrive - Ramboll\Documents\GitHub\Thesis\Thesis\dynamic_data_18_19\organized_code\active_dynamic_September.parquet')
activeSpeed = active.loc[active['sog [kn]'] <= 35]  ## only around 1500 signals were more than 35 kn, average speed usualy does not exceed 30 

#%%


active['day'] = pd.to_datetime(active['timestamp']).dt.day
#%%


decription = active.describe()


#%%
daily_activity = active['day'].value_counts()
boat_activity = active['mmsi'].value_counts()
type_activity = active['iwrap_type_from_dataset'].value_counts()
#%%
speed_activity = active['sog [kn]'].value_counts()




ax = daily_activity.plot.bar(x = 'Index', y = 'timestamp', rot=90)

