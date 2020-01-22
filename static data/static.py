# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from os.path import dirname
import pandas as pd
import numpy as np
import glob
from scipy import stats
import numpy as np
import seaborn as sns



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



path = dirname(__file__)
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    print(path + "/*.csv")
    df = pd.read_csv(filename, index_col=None,
                     error_bad_lines=False,
                     names = ['mmsi', 'length', 'width', 'minDraught', 'maxDraught', 'typeMin', 'typeMax', 'imo', 'shipName', 'aisType', 'callSign', 'a', 'b', 'c', 'd'])
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)
#frame.drop_duplicates(subset=['callSign'], keep='last', inplace=True) # TO DO: Find on which identifier to drop the duplicates


aisTypes = frame['typeMax'].value_counts(dropna=False) # number of AIS types in the dataset
aisTypes_normalized = frame['typeMax'].value_counts(normalize=True, dropna=False) # % of AIS types in the dataset

#%%
"""

MISSING VALUES

Missing values are those, that are equal to  0 or NaN.

•	First loop goes trough variable names and appends missing values to a list
•	Second loop goes trough missing values and finds counts/percentages relative to all data

"""


names = ['mmsi', 'length', 'width', 'minDraught', 'maxDraught', 'typeMin', 'typeMax', 'imo', 'shipName', 'aisType', 'callSign', 'a', 'b', 'c', 'd']
missingValues = []
missingCounts = []
missingPercentages = []

for el in names:
    missingVal=frame.loc[(frame[el]==0)|(frame[el].isna())]
    missingValues.append(missingVal)

for el in missingValues:
    missingCounts.append(len(el))
    missingPercentages.append(len(el)*100/len(frame))

resultsMissingValues = pd.DataFrame({'Variable': names,
                       'Number of missing':missingCounts,
                       '% of missing': np.round_(missingPercentages, decimals = 3)})


#%%

"""

Means, STDs and KDE, outliers

"""


# TO DO: remove missing values such as zeroes and >400m from dataset
frameCopy = frame.loc[(frame['length'] != 0) | (frame['width'] != 0)]
frameCopy = frameCopy.loc[(frame['length'] < 400)]


meansAIS = frameCopy.groupby(['typeMax'])['length', 'width'].mean()
stdAIS = frameCopy.groupby(['typeMax'])['length', 'width'].std()



## experimenting with AIS type = 70 (all Cargo ships)
smt = frame.loc[(frame['typeMax'] == 70.0) & (frame['length'] != 0) & (frame['width'] != 0)] #removing O width and length
sns.boxplot(x=smt['length']) # outliers visualisation



## z-score for removing outliers, to be finished

z = np.abs(stats.zscore(smt['length']))
print(z)
indexesO = np.where(z > 3)
#smt.drop(smt.index[[indexesO]], inplace=True)

#%%

example = frame.loc[(['mmsi'] == 273310950)]

