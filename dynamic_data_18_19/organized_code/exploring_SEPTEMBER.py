# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 09:39:25 2020

@author: KORAL
"""

import numpy as np
import pandas as pd



active = pd.read_parquet(r'C:\Users\KORAL\OneDrive - Ramboll\Documents\GitHub\Thesis\Thesis\dynamic_data_18_19\organized_code\active_dynamic_September.parquet')


#%%


active['day'] = pd.to_datetime(active['timestamp']).dt.day
#%%
daily_activity = active['day'].value_counts()
boat_activity = active['mmsi'].value_counts()
type_activity = active['iwrap_type_from_dataset'].value_counts()


ax = daily_activity.plot.bar(x = 'Index', y = 'timestamp', rot=90)

