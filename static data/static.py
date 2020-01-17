# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd
import numpy as np
import glob

"""

A few notes:
•	Min and max draught are the min and max values seen in the corresponding dynamic data file.
•	Min and max type are the min and max values seen for the ship. Type is entered manually and navigators are not always to strict on this.
•	AIS type is 1 for type A and 2 for type B. The type relates to the transmitter type and the amount of information it relays. Type A is used on commercial vessels. Type B is more often used on pleasure crafts (I assume it to be cheaper!).
•	a, b, c and d are the geometry for where the AIS GPS antenna is placed on the vessel. As with all other values in the data files, there can be missing values here.

There are about 21,000 vessels in one file. A number of the vessels will be repeats from one your to the next.

AIS ship types: 
    https://api.vtexplorer.com/docs/ref-aistypes.html

"""



path = r'C:\Users\KORAL\Documents\GitHub\Thesis\static data' # use your path
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None,
                     error_bad_lines=False,
                     names = ['mmsi', 'length', 'width', 'minDraught', 'maxDraught', 'typeMin', 'typeMax', 'imo', 'shipName', 'aisType', 'shipName', 'a', 'b', 'c', 'd'])
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)
frame.drop_duplicates(subset=['mmsi'], keep='last', inplace=True)

aisTypes = frame['typeMax'].value_counts(dropna=False) # number of AIS types in the dataset
aisTypes_normalized = frame['typeMax'].value_counts(normalize=True, dropna=False) # % of AIS types in the dataset
missingTypes = frame.loc[frame['typeMax'] == 0.0]
missingSize = frame.loc[(frame['length'] == 0) & (frame['width'] == 0)]
missingSizeAndType = missingSize.loc[missingSize['typeMax'] == 0.0]

print('missing types: ', len(missingTypes),  len(missingTypes)*100/len(frame),'%')
print('missing size: ', len(missingSize), len(missingSize)*100/len(frame),'%')
print('missing size and type: ', len(missingSizeAndType), len(missingSizeAndType)*100/len(frame),'%')
#%%


frameCopy = frame.loc[(frame['length'] != 0) | (frame['width'] !=0)]


meansAIS = frameCopy.groupby(['typeMax'])['length', 'width'].mean()
stdAIS = frameCopy.groupby(['typeMax'])['length', 'width'].std()


#%%



path = r'C:\Users\KORAL\Documents\GitHub\Thesis\static data' # use your path
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None,
                     error_bad_lines=False,
                     names = ['mmsi', 'length', 'width', 'minDraught', 'maxDraught', 'typeMin', 'typeMax', 'imo', 'shipName', 'aisType', 'shipName', 'a', 'b', 'c', 'd'])
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)


#%%

smt = frame.loc[(frame['typeMax'] == 70.0) & (frame['length'] != 0) & (frame['width'] != 0)]


import seaborn as sns
sns.boxplot(x=smt['length'])


#%%

from scipy import stats
import numpy as np
z = np.abs(stats.zscore(smt['length']))
print(z)
indexesO = np.where(z > 3)
smt.drop(smt.index[[indexesO]], inplace=True)

