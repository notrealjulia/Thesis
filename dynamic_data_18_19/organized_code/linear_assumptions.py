# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 18:12:04 2020

@author: JULSP
"""

import pandas as pd
from os.path import dirname
from sklearn.utils import shuffle
import numpy as np
from sklearn.preprocessing import LabelEncoder
import mglearn
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split 
import seaborn as sns  
import matplotlib.pyplot as plt

"""
Loading Data
"""
path = dirname(__file__)
# data without "Unidetified" vessels
data_all = pd.read_csv(path + "/data_jun_jul_aug_sep_oct.csv")

#%%
"""
0.
Pre-Processing
"""
data_clean = data_all.drop(['trips'], axis=1)
#data_clean = data_clean[data_clean.iwrap_type_from_dataset != 'Other ship']
data_clean = data_clean.dropna() #drop nan values, our models can't handle this
data_processed = data_clean[data_clean.length_from_data_set < 400] #vessels that are over 400m do not exist
data_processed = data_processed[data_processed.Speed_max <65] #boats that go faster than that are helicopters
data_processed = data_processed[data_processed.Speed_max >0] #boats that stand still

data_processed = shuffle(data_processed)

data_processed = data_processed.reset_index(drop =True) #reset index

types_of_vessels = data_processed.iwrap_type_from_dataset.value_counts()
print(types_of_vessels)


amount_limit = 20

df1 = data_processed[data_processed['iwrap_type_from_dataset'] == 'General cargo ship'][:amount_limit]
df2 = data_processed[data_processed['iwrap_type_from_dataset'] == 'Oil products tanker'][:amount_limit]
df3 = data_processed[data_processed['iwrap_type_from_dataset'] == 'Passenger ship'][:amount_limit]
df4 = data_processed[data_processed['iwrap_type_from_dataset'] == 'Pleasure boat'][:amount_limit]
df5 = data_processed[data_processed['iwrap_type_from_dataset'] == 'Other ship'][:amount_limit]
df6 = data_processed[data_processed['iwrap_type_from_dataset'] == 'Fishing ship'][:amount_limit]
df7 = data_processed[data_processed['iwrap_type_from_dataset'] == 'Support ship'][:5]
df8 = data_processed[data_processed['iwrap_type_from_dataset'] == 'Fast ferry'][:5]

data_processed = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8]) #put it all together
data_processed = data_processed.sample(frac=1).reset_index(drop=True) #shuffle
data_processed = data_processed.reset_index(drop =True) #reset index


#%%
"""
0.
Ecoding the labels
Picking X and y variables
Splitting train/test/validation
Scaling
"""

#Ecoding the labels
lb = LabelEncoder()
data_processed['iwrap_cat'] = lb.fit_transform(data_processed['iwrap_type_from_dataset'])
#picking X and y and splitting
X = data_processed[['Speed_mean', 'Speed_median', 'Speed_min', 'Speed_max', 'Speed_std', 'ROT_mean', 'ROT_median', 'ROT_min', 'ROT_max', 'ROT_std']]

#X = data_processed[['Speed_median', 'ROT_mean', 'ROT_median', 'ROT_min', 'ROT_max', 'ROT_std']]

y = data_processed[['iwrap_cat']]
y = y.values.ravel() #somethe model wants this to be an array


#split into test and train+val 
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2,  random_state=0, stratify = y)
# split train+validation set into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=1, stratify = y_trainval)

print("\nSize of training set: {}   size of validation set: {}   size of test set:" " {}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))

#Scaling the data
scaler = RobustScaler() #accounts for outliers
scaler.fit(X_trainval)
X_trainval_scaled = scaler.transform(X_trainval)
X_valid_scaled = scaler.transform(X_valid)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#make sure that we have all classes represented
print(np.unique(y_trainval))
print(np.unique(y_test))
print(np.unique(y_train))
print(np.unique(y_valid))

#%%
"""
0.
Retrieving vessel names for Classification report
LabelEncoder splits them up differently each time 
There is probably an automatic way of doing this
"""
class_0 = data_processed.loc[data_processed['iwrap_cat'] == 0, 'iwrap_type_from_dataset'].iloc[0]
class_1 = data_processed.loc[data_processed['iwrap_cat'] == 1, 'iwrap_type_from_dataset'].iloc[0]
class_2 = data_processed.loc[data_processed['iwrap_cat'] == 2, 'iwrap_type_from_dataset'].iloc[0]
class_3 = data_processed.loc[data_processed['iwrap_cat'] == 3, 'iwrap_type_from_dataset'].iloc[0]
class_4 = data_processed.loc[data_processed['iwrap_cat'] == 4, 'iwrap_type_from_dataset'].iloc[0]
class_5 = data_processed.loc[data_processed['iwrap_cat'] == 5, 'iwrap_type_from_dataset'].iloc[0]
class_6 = data_processed.loc[data_processed['iwrap_cat'] == 6, 'iwrap_type_from_dataset'].iloc[0]
class_7 = data_processed.loc[data_processed['iwrap_cat'] == 7, 'iwrap_type_from_dataset'].iloc[0]

target_names = [class_0, class_1, class_2, class_3, class_4, class_5, class_6, class_7]


#%%
"""
1.
Linear relationship and homoscadacity
"""

LR_1_data = data_processed


# ax1 = LR_1_data.plot.scatter(y = 'Speed_mean', x ='length_from_data_set', alpha = 0.3, s = 100, c = 'iwrap_type_from_dataset' )
# ax1.set_xlabel('Length of Vessel in meters')
# ax1.set_ylabel('Mean speed of Vessel in knots')
# ax1.set_title('Relationship between the Speed and the Length of the vessel \n on a subset of dynamic data')

"""
SPEED
TODO save the plots
"""

LR_1_data = LR_1_data.rename({'iwrap_type_from_dataset': 'Vessel Type'}, axis=1)

sns.set(font_scale=1.2)
sns.set_style("white")

ax = sns.scatterplot(x="length_from_data_set", y="Speed_mean", hue="Vessel Type", style = "Vessel Type", data=LR_1_data)
ax.set_xlabel('Length of Vessel in m')
ax.set_ylabel('Mean speed of vessel in kn')
# ax.set_title('Relationship between Speed and Length of vessels \n on a subset of dynamic data')
# ax.show()

#%%

data_processed['Speed_median_norm']=(data_processed['Speed_median']-data_processed['Speed_median'].min())/(data_processed['Speed_median'].max()-data_processed['Speed_median'].min())
data_processed['ROT_mean_norm']=(data_processed['ROT_mean']-data_processed['ROT_mean'].min())/(data_processed['ROT_mean'].max()-data_processed['ROT_mean'].min())

#%%
 
LR_1_data = data_processed
LR_1_data = LR_1_data.rename({'iwrap_type_from_dataset': 'Vessel Type'}, axis=1)

sns.set(font_scale=1.2)
sns.set_style("white")

ax2 = sns.scatterplot(y="ROT_mean_norm", x="Speed_median_norm", hue="Vessel Type", style = "Vessel Type",  data=LR_1_data)
ax2.set_xlabel('Mean rotation of vessel in deg/min')
ax2.set_ylabel('Median speed of vessel in kn')
# ax2.set_title('Relationship between Speed and Length of vessels \n on a subset of dynamic data')
# ax2.show()

#%%
ax3 = sns.scatterplot(x="length_from_data_set", y="Speed_std", hue="Vessel Type", style = "Vessel Type",  data=LR_1_data, legend = False)
ax3.set_xlabel('Length of Vessel in m')
ax3.set_ylabel('Standard deviation vessels speed in kn')

#%%
ax4 = sns.scatterplot(x="length_from_data_set", y="Speed_min", hue="Vessel Type", style = "Vessel Type",  data=LR_1_data, legend = False)
ax4.set_xlabel('Length of Vessel in m')
ax4.set_ylabel('Minimum speed of vessel in kn')
# ax4.set_title('Relationship between Speed and Length of vessels \n on a subset of dynamic data')
# ax4.show()
#%%
ax5 = sns.scatterplot(x="length_from_data_set", y="Speed_max", hue="Vessel Type", style = "Vessel Type",  data=LR_1_data, legend = False)
ax5.set_xlabel('Length of Vessel in m')
ax5.set_ylabel('Maximum speed of vessel in kn')
# ax5.set_title('Relationship between Speed and Length of vessels \n on a subset of dynamic data')
# ax5.show()
#%%
"""
ROTATION
"""
ax6 = sns.scatterplot(x="length_from_data_set", y="ROT_mean", hue="Vessel Type", style = "Vessel Type",  data=LR_1_data)
ax6.set_xlabel('Length of Vessel in m')
ax6.set_ylabel('Mean Rate of Turn of Vessel in deg/min')
# ax6.set_title('Relationship between Rate of Turn and Length of vessels \n on a subset of dynamic data')
ax6.show()
#%%
ax7 = sns.scatterplot(x="length_from_data_set", y="ROT_median", hue="Vessel Type", style = "Vessel Type",  data=LR_1_data, legend = False)
ax7.set_xlabel('Length of Vessel in m')
ax7.set_ylabel('Median Rate of Turn of Vessel in deg/min')
# ax7.set_title('Relationship between Rate of Turn and Length of vessels \n on a subset of dynamic data')
ax7.show()
#%%
ax8 = sns.scatterplot(x="length_from_data_set", y="ROT_std", hue="Vessel Type", style = "Vessel Type",  data=LR_1_data, legend = False)
ax8.set_xlabel('Length of Vessel in m')
ax8.set_ylabel('Standard deviation vessels ROT in deg/min')
# ax8.set_title('Relationship between Rate of Turn and Length of vessels \n on a subset of dynamic data')
ax8.show()

#%%
ax9 = sns.scatterplot(x="length_from_data_set", y="ROT_min", hue="Vessel Type", style = "Vessel Type",  data=LR_1_data, legend = False)
ax9.set_xlabel('Length of Vessel in m')
ax9.set_ylabel('Minimum Rate of Turn of Vessel in deg/min')
# ax9.set_title('Relationship between Rate of Turn and Length of vessels \n on a subset of dynamic data')
ax9.show()

#%%
ax10 = sns.scatterplot(x="length_from_data_set", y="ROT_max", hue="Vessel Type", style = "Vessel Type",  data=LR_1_data, legend = False)
ax10.set_xlabel('Length of Vessel in m')
ax10.set_ylabel('Maximum Rate of Turn of Vessel in deg/min')
# ax10.set_title('Relationship between Rate of Turn and Length of vessels \n on a subset of dynamic data')
ax10.show()


#%%
"""
2.
Normal Distribution
"""

from scipy.stats import norm
LR_2_data = data_processed
sns.set_color_codes()
sns.set_style("white")

x = LR_2_data.Speed_mean
ax11 = sns.distplot(x, kde_kws = {"lw": 2, "label" : "KDE"})
ax11.set_title("Distribution of Mean Speed values of a vessel subset")
ax11.set_xlabel("Mean Speed in knots")


#%%
x = LR_2_data.Speed_median
ax12 = sns.distplot(x, kde_kws = {"lw": 2, "label" : "KDE"})
ax12.set_title("Distribution of Median Speed values of a vessel subset")
ax12.set_xlabel("Median Speed in knots")

#%%

x = LR_2_data.Speed_std
ax13 = sns.distplot(x, kde_kws = {"lw": 2, "label" : "KDE"})
ax13.set_title("Distribution of Speed Standard Deviation values")
ax13.set_xlabel("Standard deviation values of Speed in knots")


#%%

x = LR_2_data.Speed_min
ax14 = sns.distplot(x, kde_kws = {"lw": 2, "label" : "KDE"})
ax14.set_title("Distribution Minimum Speed values of a vessel subset")
ax14.set_xlabel("Minimal Speed in knots")


#%%

x = LR_2_data.Speed_max
ax15 = sns.distplot(x, kde_kws = {"lw": 2, "label" : "KDE"})
ax15.set_title("Distribution Maximum Speed values of a vessel subset")
ax15.set_xlabel("Maximum Speed in knots")


#%%

x = LR_2_data.ROT_mean
ax16 = sns.distplot(x,kde_kws = {"lw": 2, "label" : "KDE"})
ax16.set_title("Mean Rate of Turn values of a vessel subset")
ax16.set_xlabel("Mean Rate of Turn in deg/min")


#%%

x = LR_2_data.ROT_median
ax17 = sns.distplot(x,  kde_kws = {"lw": 2, "label" : "KDE"})
ax17.set_title("Median Rate of Turn values of a vessel subset")
ax17.set_xlabel("Median Rate of Turn in deg/min")
#%%

x = LR_2_data.ROT_std
ax18 = sns.distplot(x, kde_kws = {"lw": 2, "label" : "KDE"})
ax18.set_title("Standard deviation values of a vessel subset")
ax18.set_xlabel("Minimal Rate of Turn in deg/min")

#%%

x = LR_2_data.ROT_max
ax19 = sns.distplot(x, bins = 15, kde_kws = {"lw": 2, "label" : "KDE"})
ax19.set_title("Maximum Rate of Turn values of a vessel subset")
ax19.set_xlabel("Maximum Rate of Turn in deg/min")

#%%

x = LR_2_data.ROT_min
ax20 = sns.distplot(x,  kde_kws = {"lw": 2, "label" : "KDE"})
ax20.set_title("Minimum Rate of Turn values of a vessel subset")
ax20.set_xlabel("Minimal Rate of Turn in deg/min")


#%%

"""
3. 
Multicolinearity
correlation matrix
"""

ax = plt.axes()
data_for_corr = X
corrMatrix = data_for_corr.corr()
sns.heatmap(corrMatrix, ax = ax, center = 0, annot=True, vmin = -1, vmax = 1, )
ax.set_title('Correlation Matrix of Dynamic Variables from 4,000 daily vessel movements')
plt.show()

