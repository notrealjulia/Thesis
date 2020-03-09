# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:08:49 2020

@author: JULSP
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname

path = dirname(__file__)
#dynamic_data_path = "//garbo/Afd-681/05-Technical Knowledge/05-03-Navigational Safety/00 Udviklingsprojekter/01 ML - Missing properties/02 Work/01 IWRAP/ML South Baltic Sea vA/export"


data = pd.read_csv(path + "/Trips_and_VesselStatus_Static.csv", sep='	', index_col=None, error_bad_lines=False)


d = {'ship type': ['Fast ferry', 'Fishing ship', 'General cargo ship', 'Oil products tanker','Other ship','Passenger ship','Pleasure boat', 'Support ship'], 
     'number of trips': [4986, 18170, 35384, 8372, 78411, 28558, 179456, 53594]}
df = pd.DataFrame(data=d)
df = df.set_index('ship type')

#df = df.drop(['Pleasure boat', 'Other ship'])

df.plot.pie(y = 'number of trips', autopct='%1.0f%%', pctdistance=0.8, labeldistance=1.2)
#ax.title('sdknfls')


#%%

data.drop(data[ (data['signals'] <= 1000) | (data['status'] == 'not active')].index, inplace = True)



trips_for_types = data.groupby('iwrap_type_from_dataset')['trips'].sum() 

ax2 = trips_for_types.plot.pie(y = 'trips', autopct='%1.0f%%', pctdistance=0.8, labeldistance=1.2)
#%%

data_selected = trips_for_types.drop(['Pleasure boat', 'Other ship'])
data_selected.plot.pie(y = 'trips', autopct='%1.0f%%', pctdistance=0.8, labeldistance=1.2)
