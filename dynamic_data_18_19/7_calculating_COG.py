# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:47:59 2020

@author: JULSP

THIS IS A MESS, WILL HAVE TO REMAKE
"""

import pandas as pd
from os.path import dirname
import matplotlib.pyplot as plt

path = dirname(__file__)
data_dynamic_test = pd.read_csv(path + "/246669000.csv", sep='	', index_col=None, error_bad_lines=False)

#%%

data = data_dynamic_test.set_index('timestamp')
#transforming index into resampable form 
data.index = pd.to_datetime(data.index)
#resampling for every 10minutes and dropping empty values, taking mean value of the new sample bin
data_resample = data.resample('T').mean()
data_resample = data_resample.dropna()
data_resample['Cog_diff'] = data_resample['cog [deg]'].diff()
data_resample['Cog_diff_abs'] = data_resample['Cog_diff'].abs()
#DList = [group[1] for group in data_resample.groupby(data_resample.index.day)
#data_cog_dif = data_resample['cog [deg]'].diff()
#max_rot = data_resample['cog [deg]'].max()

#find difference between time stamps, to locate information holes
time_difference = data_resample.index.to_series().diff()

#add differences in timestamps to the original df
data_with_time_difference = pd.concat([data_resample, time_difference], axis=1, sort=False)

#drop all lines that do not deal with 1 minute intervals
data_with_time_difference_1min = data_with_time_difference[data_with_time_difference.timestamp == '00:01:00']

#calculate maximum COG change per minute
max_value_of_ship_ROT = data_with_time_difference_1min.Cog_diff_abs.max()
median_value_of_ship_ROT = data_with_time_difference_1min.Cog_diff_abs.median()
mean_value_of_ship_ROT = data_with_time_difference_1min.Cog_diff_abs.mean()

#%%

#data_with_time_difference_1min.boxplot(column = 'Cog_diff_abs' )
data_dynamic_test.hist(column = 'cog [deg]')

#%%
#split it up in days 

#one_day = data_cog_dif.loc['2018-12-22']

#DList = [group[1] for group in data_cog_dif.groupby(data_cog_dif.index.day)]

max_list = []
for day in DList:
    max_list.append(day['cog [deg]'].diff())
    
#%%
data_2 = pd.to_datetime(data_dynamic_test.timestamp)
print(data_2.head())
time_list = data_2.diff()


#%%

#print(data_resample.head())
time_difference = data_resample.index.to_series().diff()

#%%
for index, row in data_resample.iterrows():
    print (index)
#%%

import seaborn as sn

corr_test_df = data_dynamic_test.drop(columns = 'rot [deg/min]')
ax = plt.axes()
corrMatrix = corr_test_df.corr()
sn.heatmap(corrMatrix, ax = ax, center = 0, annot=True)
ax.set_title('Correlation Matrix of Dynamic Variables from one vessel')
plt.show()
