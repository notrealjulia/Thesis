# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 17:12:10 2020

@author: JULSP
"""

import pandas as pd
from os.path import dirname
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
# import mglearn #for visualising grid search - comment out if can't import
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

#%%
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
data_clean = data_clean.dropna()
# data_clean = data_clean[data_clean.iwrap_type_from_dataset != 'Other ship'] 
# data_clean = data_clean[data_clean.iwrap_type_from_dataset != 'Oil products tanker'] 
# data_clean = data_clean[data_clean.iwrap_type_from_dataset != 'Support ship'] 
# data_clean = data_clean[data_clean.iwrap_type_from_dataset != 'Fast ferry'] 


data_processed = data_clean[data_clean.length_from_data_set < 400] #vessels that are over 400m do not exist
data_processed = data_processed[data_processed.Speed_max <65] #boats that go faster than that are helicopters
data_processed = data_processed[data_processed.Speed_max >0] #boats that stand still


from sklearn.utils import shuffle
data_processed = shuffle(data_processed)

data_processed = data_processed.reset_index(drop =True) #reset index

types_of_vessels = data_processed.iwrap_type_from_dataset.value_counts()
print(types_of_vessels)

#%%
"""
0.
Limiting each amount of vessels 
NEURAL NETWORK doesn't need this step
"""

amount_limit = 50

df1 = data_processed[data_processed['iwrap_type_from_dataset'] == 'General cargo ship'][:amount_limit]
df2 = data_processed[data_processed['iwrap_type_from_dataset'] == 'Oil products tanker'][:amount_limit]
df3 = data_processed[data_processed['iwrap_type_from_dataset'] == 'Passenger ship'][:amount_limit]
df4 = data_processed[data_processed['iwrap_type_from_dataset'] == 'Pleasure boat'][:amount_limit]
df5 = data_processed[data_processed['iwrap_type_from_dataset'] == 'Other ship'][:amount_limit]
df6 = data_processed[data_processed['iwrap_type_from_dataset'] == 'Fishing ship'][:amount_limit]
df7 = data_processed[data_processed['iwrap_type_from_dataset'] == 'Support ship'][:amount_limit]
df8 = data_processed[data_processed['iwrap_type_from_dataset'] == 'Fast ferry'][:amount_limit]

data_processed = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8]) #put it all together
data_processed = data_processed.sample(frac=1).reset_index(drop=True) #shuffle
data_processed = data_processed.reset_index(drop =True) #reset index

data_processed = data_processed.reset_index(drop =True) #reset index
#%%
"""
0.
Picking the x and y variables
"""
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()

#Ecoding the labels
# lb = LabelEncoder()
labels = lb.fit_transform(data_processed['iwrap_type_from_dataset'])
#%%
#picking X and y and splitting
X = data_processed[['Speed_median', 'Speed_max', 'Speed_std', 'ROT_mean', 'ROT_std']]
y = labels
# y = y.values.ravel() #somethe model wants this to be an array

#%%
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


#need this later for visualisation
feature_names = ['Speed_median', 'Speed_max', 'Speed_std', 'ROT_mean', 'ROT_std']

#%%

"""
For classification visualize scatter plots
"""
data_frame = pd.DataFrame(X_train_scaled, columns = feature_names)
scatter_matrix = pd.plotting.scatter_matrix(data_frame, c= y_train, cmap = 'Dark2', s = 50, alpha = 0.9)
