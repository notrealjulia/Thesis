# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:24:54 2020

@author: JULSP
"""

import pandas as pd
from os.path import dirname
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.mstats import mode
import seaborn as sn
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing
from sklearn import metrics

#%%
"""
Loading Data
& Pre-Processing
"""

path = dirname(__file__)
data = pd.read_csv(path + "/static_all_with_speed_rot.csv") #column 3 to 12 is dynamic data

data_processed = data[data.length_from_data_set < 400] #vessels that are over 400m are usually helicopters
data_processed = data_processed[data_processed.length_from_data_set > 10]
data_processed = data_processed[data_processed.Speed_mean != 0]
data_processed = data_processed[data_processed.Speed_max <65]

#data_processed = data_processed[data_processed.Speed_std.dropna()]
#%%
"""
Picking the x and y variables
"""
X = data_processed.iloc[:,[21,23,24,25,26]]
y = data_processed.iloc[:,9]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#%%
"""
Normalizing and Scaling
"""

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
#scaler = MinMaxScaler()
scaler = RobustScaler()
#scaler = StandardScaler()

scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


#%%
"""
Linear Regression
"""
#standard linear regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr.predict(X_test_scaled)
print("Training set score: {:.2f}".format(lr.score(X_train_scaled, y_train)))
print("Test set score: {:.2f}\n".format(lr.score(X_test_scaled, y_test)))

#linear regression on scaled data - should not make a difference
lr2 = LinearRegression().fit(X_train, y_train)
print("Unscaled Training set score: {:.2f}".format(lr2.score(X_train, y_train)))
print("Unscaled Test set score: {:.2f}\n".format(lr2.score(X_test, y_test)))

#linear regression with regularization of w megnitude
ridge01 = Ridge(alpha=0.01).fit(X_train, y_train)
print("Ridge Training set score: {:.2f}".format(ridge01.score(X_train, y_train)))
print("Ridge Test set score: {:.2f}\n".format(ridge01.score(X_test, y_test)))

#linear regression with regularization og w amount
lasso = Lasso(alpha = 0.5).fit(X_train, y_train)
print("Lasso Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
print("Lasoo Test set score: {:.2f}\n".format(lasso.score(X_test, y_test)))


#%%
"""
Ensamble models
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

regr = RandomForestRegressor(n_estimators=100, max_depth=25, max_features =5, random_state=0, ccp_alpha =5)
                            #max features log2(number of features)
regr.fit(X_train_scaled, y_train)
print("Random Forest Training set score: {:.2f}".format(regr.score(X_train_scaled, y_train)))
print("Random Forest Test set score: {:.2f}\n".format(regr.score(X_test_scaled, y_test)))

#ignores certain features, should outpreform Random Forest
regr2 = GradientBoostingRegressor( )
regr2.fit(X_train_scaled, y_train)
print("Gradient Boosting Training set score: {:.2f}".format(regr2.score(X_train_scaled, y_train)))
print("Gradient Boosting Test set score: {:.2f}\n".format(regr2.score(X_test_scaled, y_test)))

#%%
"""
Neural Network
"""

from sklearn.neural_network import MLPRegressor

nn = MLPRegressor(max_iter = 10000, random_state=0, alpha = 0.01)
nn.fit(X_train, y_train)
print("Neural Network Training set score: {:.2f}".format(nn.score(X_train, y_train)))
print("Neural Network Test set score: {:.2f}\n".format(nn.score(X_test, y_test)))
