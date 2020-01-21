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
import matplotlib.pyplot as plt  

path = dirname(__file__)
frame = pd.read_csv(path + "/static_ship_data.csv", sep='	', index_col=None, error_bad_lines=False)
#frame.drop_duplicates(subset=['imo'], keep='last', inplace=True) # TO DO: Find on which identifier to drop the duplicates



#%%
    
"""

PRE-PREOCESSING

•	Removing irrelevant columns (100% missing values)
•	Renaming columns
•	Separating 'Undefined' category from 'Other' category in iwrap column using mask and org_type column
•	Finding type values

"""

mask = (frame['org_type_info_from_data'] == 'Undefined') & (frame['iwrap_type_from_dataset'] == 'Other ship')
frame['iwrap_type_from_dataset'][mask] = 'Undefined'

frame = frame[['mmsi', 'imo_from_data_set', 'name_from_data_set', 'iwrap_type_from_dataset', 'org_type_info_from_data', 'length_from_data_set', 'width', 'size_a','size_b','size_c','size_d']]
frame = frame.rename(columns={"imo_from_data_set":"imo", "name_from_data_set": "name", "iwrap_type_from_dataset": "iwrapType", "org_type_info_from_data": "orgType", "length_from_data_set": "length", "iwrap_type_from_dataset": "iwrapType"})

orgTypes = frame['orgType'].value_counts(dropna=False) # number of AIS (org) types in the dataset
orgTypes_normalized = frame['orgType'].value_counts(normalize=True, dropna=False) # % of AIS (org) types in the dataset

iwrapTypes = frame['iwrapType'].value_counts(dropna=False) # number of AIS (iwrap) types in the dataset
iwrapTypes_normalized = frame['iwrapType'].value_counts(normalize=True, dropna=False) # % of AIS (iwrap) types in the dataset
#%%


"""

MISSING VALUES

Missing values are those, that are equal to  0, -1 or NaN.

•	First loop goes trough variable names and appends missing values to a list
•	Second loop goes trough missing values and finds counts/percentages relative to all data

"""


names = list(frame)

missingValues = []
missingCounts = []
missingPercentages = []

for el in names:
    missingVal=frame.loc[(frame[el]==0)|(frame[el]==-1)|(frame[el].isna())]
    missingValues.append(missingVal)

for el in missingValues:
    missingCounts.append(len(el))
    missingPercentages.append(len(el)*100/len(frame))

resultsMissingValues = pd.DataFrame({'Variable': names,
                       'Number of missing':missingCounts,
                       '% of missing': np.round_(missingPercentages, decimals = 2)})

    
iwrapUndefined  = frame.loc[(frame['iwrapType']=='Undefined')]
#%%

"""

ANALYSIS


•	Finding/Removing outliers
•	Finding Means, STD, Variance
•	Kernel Density Estimation


"""

types = list(iwrapTypes.index.values)


frameCopy = frame.loc[(frame['length'] != -1) | (frame['width'] != -1)] ##  a copy without missing length and width properties and lengths > 400 
frameCopy = frameCopy.loc[(frameCopy['length'] <= 400)]

## OUTLIERS: 
plt.figure(figsize=(10,5))    
L = sns.boxplot(x="iwrapType", y="length",
                # hue="smoker", col="time",
                 data=frameCopy)
L.set_xticklabels(L.get_xticklabels(), rotation=45)


plt.figure(figsize=(10,5))    
W = sns.boxplot(x="iwrapType", y="width",
                # hue="smoker", col="time",
                 data=frameCopy)
W.set_xticklabels(W.get_xticklabels(), rotation=45)

# To do: find z scores, decide the threshold for removing outliers if any




## MEAN, STD
meansAIS = frameCopy.groupby(['iwrapType'])['length', 'width'].mean()
stdAIS = frameCopy.groupby(['iwrapType'])['length', 'width'].std()


## KDE (Kernel Density Estimation of size by vessel type)

types = list(iwrapTypes.index.values)
cols =  []

for el in types:
    cols.append(frameCopy[frameCopy.iwrapType == el])

for col in cols:
    plt.figure(figsize=(5,5))
    ax = sns.kdeplot(col.length, col.width,
                      cmap="Blues", shade=True, shade_lowest=False)
    sns.despine()
    ax.set(xlabel='Length', ylabel='Width')
    ax.set_title(col.iwrapType.iloc[0])
    #plt.savefig('kde_{}.svg'.format(col.iwrap_type_from_dataset.iloc[0]), format='svg', dpi=1000)
    


#%%
"""

VISUALISATION

"""

## VESSEl TYPES (IWRAP)

types=frame.iwrapType.value_counts()
fig, ax = plt.subplots()    
width = 0.75 # the width of the bars 
ind = np.arange(len(types))  # the x locations for the groups
ax.barh(ind, types, width)
ax.set_yticks(ind+width/2)
ax.set_yticklabels(types.index, minor=False)
plt.title('title')
for i, v in enumerate(types):
    plt.text(v, i, " "+str(v), va='center', fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


plt.tick_params(axis='both', which='both', bottom='off',labelbottom='off')
plt.title('iWrap vessel types, \n South Baltic Sea 2018-2019', fontsize=12)

