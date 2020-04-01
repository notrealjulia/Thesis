# -*- coding: utf-8 -*-
"""
Created on Thu Mar 5 11:24:54 2020

@author: JULSP

All Regression models used for predicting length based on dynamic data
1.Linear models
2.Ensamble models
3.and a Neural Network

With cross validation for all models
Train/test evaluation for all models
MAE based on r^2 for all models
Grid search for 2. ensamble and 3. neural networks
Visualisation of coeffiecient magnitudes for different features

4. Mean absolute error calculated seperately for all ship types for the best preforming model (Random Forest Regressor)
"""

import pandas as pd
from os.path import dirname
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
import mglearn #for visualising grid search - comment out if can't import
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

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
data_clean = data_clean[data_clean.iwrap_type_from_dataset != 'Other ship'] 
data_processed = data_clean[data_clean.length_from_data_set < 400] #vessels that are over 400m do not exist
data_processed = data_processed[data_processed.Speed_max <65] #boats that go faster than that are helicopters
data_processed = data_processed[data_processed.Speed_max >0] #boats that stand still

data_processed = data_processed.reset_index(drop =True) #reset index
#%%
"""
0.
Picking the x and y variables
"""
X = data_processed[['Speed_mean', 'Speed_median', 'Speed_min', 'Speed_max', 'Speed_std', 'ROT_mean', 'ROT_median', 'ROT_min', 'ROT_max', 'ROT_std']]
y = data_processed[['length_from_data_set']]
y = y.values.ravel() #somethe model wants this to be an array

#need this later for visualisation
feature_names = ['Speed_mean', 'Speed_median', 'Speed_min', 'Speed_max', 'Speed_std', 'ROT_mean', 'ROT_median', 'ROT_min', 'ROT_max', 'ROT_std']

#%%
"""
0.
Splitting the data
"""
#split into test and train+val 
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2,  random_state=0)
# split train+validation set into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=1)

print("\nSize of training set: {}   size of validation set: {}   size of test set:" " {}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))
#%%
"""
0.
Normalizing and Scaling
"""
scaler = RobustScaler() #accounts for outliers unlike Standard Scaler

scaler.fit(X_trainval)
X_trainval_scaled = scaler.transform(X_trainval) #TODO Double check if this is correct scaling or if it should be scaled on training data only
X_valid_scaled = scaler.transform(X_valid)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
#%%
"""
1.
Linear Models
Linear Regression
Scaling is not necessary
"""
lr = LinearRegression()

#Cross validation
cross_val_scores_lr = cross_val_score(lr, X =X_valid_scaled, y =y_valid, cv = 5)
print("Cross-validation scores for linear regression:\n {}\n".format(cross_val_scores_lr))
print("Mean cross validation for LR {:.3f}".format(np.mean(cross_val_scores_lr)))

#Train and test 
lr.fit(X_train_scaled, y_train)
lr.predict(X_test_scaled)
print("Training set score: {:.3f}".format(lr.score(X_train_scaled, y_train)))
print("Test set score: {:.3f}\n".format(lr.score(X_test_scaled, y_test)))

#Mean absolute error (to interpret MSqR)
y_pred = lr.predict(X_test_scaled)
MAE = mean_absolute_error(y_test, y_pred)
print("Absolute mean error for LR is {:.3f}m\n".format(MAE))

###RESULTS without Other ship:
# Cross-validation scores for linear regression:
#  [0.5552764  0.60799746 0.58145962 0.59658759 0.59179572]

# Mean cross validation for LR 0.587
# Training set score: 0.600
# Test set score: 0.593

# Absolute mean error for LR is 34.506m


###RESULTS with Other ship:
# Cross-validation scores for linear regression:
#  [0.58261925 0.59685726 0.57082665 0.57861807 0.56089348]

# Mean cross validation for LR 0.578
# Training set score: 0.562
# Test set score: 0.565

# Absolute mean error for LR is 35.383m
#%%

"""
2.
Visulising w's magnitude in LR
"""

print("Features sorted by their score:")

lr_feature = sorted(zip(map(lambda x: round(x, 4), lr.coef_), feature_names), reverse=False)
print(sorted(zip(map(lambda x: round(x, 4), lr.coef_), feature_names), 
              reverse=True))

number_list_2 = []
name_list_2 = []
i = 0
for i in range(10):
    number_list_2.append(lr_feature[i][0]) 
    name_list_2.append(lr_feature[i][1])

plt.barh(name_list_2, number_list_2)
plt.title('Weights assigned to different features by Linear Regression')
plt.xlabel('weight')

#%%
"""
1.
Linear Regression with L2 and L1 Regularization
This improved nothing, LR doesn't overfit - can ignore
"""

#linear regression with regularization of w megnitude
ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
print("Ridge Training set score: {:.3f}".format(ridge01.score(X_train, y_train)))
print("Ridge Test set score: {:.3f}\n".format(ridge01.score(X_test, y_test)))

#linear regression with regularization of w amount
lasso = Lasso(alpha = 0.01, max_iter = 1000).fit(X_train, y_train)
print("Lasso Training set score: {:.3f}".format(lasso.score(X_train, y_train)))
print("Lasoo Test set score: {:.3f}\n".format(lasso.score(X_test, y_test)))

#combining l1 and l2
lasso = ElasticNet().fit(X_train, y_train)
print("ElasticNet Training set score: {:.3f}".format(lasso.score(X_train, y_train)))
print("ElasticNet Test set score: {:.3f}\n".format(lasso.score(X_test, y_test)))

#%%
"""
2.
Ensamble models
Random Forest - Grid search and Cross Validation
Use scaled data for Tree based models
!!! TAKES A WHILE TO RUN - USES ALL CORES !!!
"""
forest_regr = RandomForestRegressor(random_state = 42)

#parameteres for RF
param_grid = {'n_estimators': [ 70, 80, 90],
              'max_depth': [15, 20, 35],
              'max_features': [2, 3, 5],
              'ccp_alpha': [2, 3, 5]}

grid_search = GridSearchCV(forest_regr, param_grid, cv=5, n_jobs =-1)
grid_search.fit(X_valid_scaled, y_valid)
print("Best parameters: {}".format(grid_search.best_params_))
best_parameters = grid_search.best_params_
print("Mean cross-validated score of the best_estimator: {:.2f}".format(grid_search.best_score_))

### RESULTS:
# Best parameters: {'ccp_alpha': 2, 'max_depth': 35, 'max_features': 5, 'n_estimators': 90}
# Mean cross-validated score of the best_estimator: 0.79

#%%
"""
2.
Random Forest model
Random Grid Search continued - visualising results
#only works with 2 hyper parameters therefore commented out
"""

# #convert to DF
# results = pd.DataFrame(grid_search.cv_results_)
# scores = np.array(results.mean_test_score).reshape(3,3)

# # plot the mean cross-validation scores
# mglearn.tools.heatmap(scores, xlabel='n_estimators', xticklabels=param_grid['n_estimators'],
#                       ylabel='max_depth', yticklabels=param_grid['max_depth'], cmap="viridis")

#CV_scores = cross_val_score(grid_search)
#print("Cross-validation scores: ", CV_scores)
#print("Mean cross-validation score: ", CV_scores.mean())

##“RF can achieve 76% mean cross-validation accuracy on the Dynamic AIS dataset”—nothing more and nothing less.
#%%
"""
2.
Random Forest model
Encorporate the best hyper parameters from Grid Search
Train/test
MAE
"""
regr = RandomForestRegressor(**best_parameters, random_state = 42)
# regr = RandomForestRegressor(n_estimators=60, max_depth=10, max_features =3, random_state=0, 
#                              ccp_alpha =5) #max features log2(number of features)

regr.fit(X_train_scaled, y_train)
print("Random Forest Training set score: {:.2f}".format(regr.score(X_train_scaled, y_train)))
print("Random Forest Test set score: {:.2f}\n".format(regr.score(X_test_scaled, y_test)))

#mean absolute error
y_pred = regr.predict(X_test_scaled)
MAE_RF = mean_absolute_error(y_test, y_pred)
print("absolute mean error for RF is {:.3f}m\n".format(MAE_RF))


#RESULTS for Random Forest: without the 'Other ships'
# Random Forest Training set score: 0.84
# Random Forest Test set score: 0.82

# absolute mean error for RF is 20.680m

#RESULTS for Random Forest with "Other ships"

# Random Forest Training set score: 0.82
# Random Forest Test set score: 0.78

# absolute mean error for RF is 21.821m

#%%
"""
2.
Visulising w's magnitude in RF for different features
"""
print("Features sorted by their score:")

fe = sorted(zip(map(lambda x: round(x, 4), regr.feature_importances_), feature_names), reverse=False)

print(fe[0][1])
number_list = []
name_list = []
i = 0
for i in range(10):
    number_list.append(fe[i][0]) 
    name_list.append(fe[i][1])

plt.barh(name_list, number_list, orientation = 'horizontal')
plt.title('Weights assigned to different features by Random Forest')
plt.xlabel('weight')
#%%
"""
2.
Gradient Boosting model
Very much like Random forest - preforms slightly better
!!! TAKES A WHILE TO RUN - USES ALL CORES !!!
"""

regr2 = GradientBoostingRegressor(random_state=42)

#parameteres for RF
param_grid2 = {'n_estimators': [ 55, 60, 65, 70],
              'max_depth': [7,8, 9, 10],
              'max_features': [2, 3, 5],
              'ccp_alpha': [0.1,0.5, 1, ]}

#Best parameters: {'ccp_alpha': 0, 'max_depth': 10, 'max_features': 3, 'n_estimators': 70}

grid_search2 = GridSearchCV(regr2, param_grid2, cv=5, n_jobs =-1)
grid_search2.fit(X_valid_scaled, y_valid) #on validation set
print("Best parameters: {}".format(grid_search2.best_params_))
best_parameters2 = grid_search2.best_params_
print("Mean cross-validated score of the best_estimator: {:.2f}".format(grid_search2.best_score_))


#ignores certain features, should outpreform Random Forest
regr2 = GradientBoostingRegressor(**best_parameters2, random_state=42)
regr2.fit(X_train_scaled, y_train)
print("Gradient Boosting Training set score: {:.2f}".format(regr2.score(X_train_scaled, y_train)))
print("Gradient Boosting Test set score: {:.2f}\n".format(regr2.score(X_test_scaled, y_test)))

#mean absolute error
y_pred = regr2.predict(X_test_scaled)
MAE_GB = mean_absolute_error(y_test, y_pred)
print("Absolute mean error for GB is {:.3f}m\n".format(MAE_GB))

#RESULTS for Gradient Boosting without "Other" vessels:
# Best parameters: {'ccp_alpha': 0.1, 'max_depth': 8, 'max_features': 3, 'n_estimators': 70}
# Mean cross-validated score of the best_estimator: 0.82
# Gradient Boosting Training set score: 0.90
# Gradient Boosting Test set score: 0.84

# Absolute mean error for GB is 19.224m

#RESULTS for GB with OTHER vessels:
# Best parameters: {'ccp_alpha': 1, 'max_depth': 8, 'max_features': 5, 'n_estimators': 70}
# Mean cross-validated score of the best_estimator: 0.79
# Gradient Boosting Training set score: 0.83
# Gradient Boosting Test set score: 0.79

# Absolute mean error for GB is 21.424m
#%%
"""
2.
Visulising w's magnitude in GB
"""

print("Features sorted by their score:")

fe2 = sorted(zip(map(lambda x: round(x, 4), regr2.feature_importances_), feature_names), reverse=False)
print(sorted(zip(map(lambda x: round(x, 4), regr2.feature_importances_), feature_names), 
              reverse=True))

number_list_2 = []
name_list_2 = []
i = 0
for i in range(10):
    number_list_2.append(fe2[i][0]) 
    name_list_2.append(fe2[i][1])

plt.barh(name_list_2, number_list_2)
plt.title('Weights assigned to different features by Gradient Boosting')
plt.xlabel('weight')
#%%
"""
3.
Neural Network
Grid Search - for Neural Network

!!! TAKES A WHILE TO RUN - USES ALL CORES !!!

"""
nn = MLPRegressor(max_iter = 20000, random_state=0)

param_grid = {'hidden_layer_sizes': [(100,50,10), (100,50), (100,100)],
              'alpha': [0.0001, 0.001, 0.01],}

grid_search = GridSearchCV(nn, param_grid, cv=5, n_jobs =-1) #incorporates Cross validation, , n_jobs =-1 uses all PC cores
grid_search.fit(X_valid_scaled, y_valid)
print("Best parameters: {}".format(grid_search.best_params_))
print("Mean cross-validated score of the best_estimator: {:.2f}".format(grid_search.best_score_))
best_parameters3 = grid_search.best_params_

# Best parameters: {'alpha': 0.001, 'hidden_layer_sizes': (100, 50, 10)}
# Mean cross-validated score of the best_estimator: 0.78
#%%
""" 
3.
Neural Network
Grid Search continued - visualising results
"""

#convert to DF
results = pd.DataFrame(grid_search.cv_results_)
scores = np.array(results.mean_test_score).reshape(3, 3)

# plot the mean cross-validation scores
mglearn.tools.heatmap(scores, xlabel='hidden_layer_sizes', xticklabels=param_grid['hidden_layer_sizes'],
                      ylabel='alpha', yticklabels=param_grid['alpha'], cmap="viridis")

#%%
"""
3.
Neural Network
Train Test
MAE
"""

# nn = MLPRegressor(max_iter = 20000, random_state=0, **best_parameters3)  #
nn = MLPRegressor(max_iter = 20000, random_state=0, hidden_layer_sizes = (10,100,50,10))
nn.fit(X_train_scaled, y_train)
print("Neural Network Training set score: {:.2f}".format(nn.score(X_train_scaled, y_train)))
print("Neural Network Test set score: {:.2f}\n".format(nn.score(X_test_scaled, y_test)))

#mean absolute error
y_pred = nn.predict(X_test_scaled)
MAE_NN = mean_absolute_error(y_test, y_pred)
print("Absolute mean error for GB is {:.3f}m\n".format(MAE_NN))

###RESLUTS:
# Neural Network Training set score: 0.81
# Neural Network Test set score: 0.80

# Absolute mean error for GB is 22.123m

####RESULTS with Other ship types
# Neural Network Training set score: 0.79
# Neural Network Test set score: 0.77

# Absolute mean error for GB is 22.135m
#%%
"""
3. Neural Network

Visualisaing w's magnitude in the first layer
"""


import matplotlib.pyplot as plt

plt.figure(figsize=(20, 5))
plt.imshow(nn.coefs_[0], interpolation='none', cmap='viridis')
plt.yticks(range(10), feature_names)
plt.xlabel("Columns in weight matrix")
plt.ylabel("Input feature")
plt.colorbar()

#%%

"""
4. 
How well does the best model predict for individual ship types?
Not sure if this part is right 
"""

tanker = data_processed[data_processed['iwrap_type_from_dataset'] == 'Oil products tanker']
cargo = data_processed[data_processed['iwrap_type_from_dataset'] == 'General cargo ship']
passenger = data_processed[data_processed['iwrap_type_from_dataset'] == 'Passenger ship']
support = data_processed[data_processed['iwrap_type_from_dataset'] == 'Support ship']
fast_ferry = data_processed[data_processed['iwrap_type_from_dataset'] == 'Fast ferry']
fishing = data_processed[data_processed['iwrap_type_from_dataset'] == 'Fishing ship']
boat = data_processed[data_processed['iwrap_type_from_dataset'] == 'Pleasure boat']


tanker_y = tanker[['length_from_data_set']].values.ravel()
tanker_X =  tanker[['Speed_mean', 'Speed_median', 'Speed_min', 'Speed_max', 'Speed_std', 'ROT_mean', 'ROT_median', 'ROT_min', 'ROT_max', 'ROT_std']]
tanker_X_scaled = scaler.transform(tanker_X)

cargo_y = cargo[['length_from_data_set']].values.ravel()
cargo_X =  cargo[['Speed_mean', 'Speed_median', 'Speed_min', 'Speed_max', 'Speed_std', 'ROT_mean', 'ROT_median', 'ROT_min', 'ROT_max', 'ROT_std']]
cargo_X_scaled = scaler.transform(cargo_X)

passenger_y = passenger[['length_from_data_set']].values.ravel()
passenger_X =  passenger[['Speed_mean', 'Speed_median', 'Speed_min', 'Speed_max', 'Speed_std', 'ROT_mean', 'ROT_median', 'ROT_min', 'ROT_max', 'ROT_std']]
passenger_X_scaled = scaler.transform(passenger_X)

support_y = support[['length_from_data_set']].values.ravel()
support_X =  support[['Speed_mean', 'Speed_median', 'Speed_min', 'Speed_max', 'Speed_std', 'ROT_mean', 'ROT_median', 'ROT_min', 'ROT_max', 'ROT_std']]
support_X_scaled = scaler.transform(support_X)

fast_ferry_y = fast_ferry[['length_from_data_set']].values.ravel()
fast_ferry_X =  fast_ferry[['Speed_mean', 'Speed_median', 'Speed_min', 'Speed_max', 'Speed_std', 'ROT_mean', 'ROT_median', 'ROT_min', 'ROT_max', 'ROT_std']]
fast_ferry_X_scaled = scaler.transform(fast_ferry_X)

fishing_y = fishing[['length_from_data_set']].values.ravel()
fishing_X =  fishing[['Speed_mean', 'Speed_median', 'Speed_min', 'Speed_max', 'Speed_std', 'ROT_mean', 'ROT_median', 'ROT_min', 'ROT_max', 'ROT_std']]
fishing_X_scaled = scaler.transform(fishing_X)

boat_y = boat[['length_from_data_set']].values.ravel()
boat_X =  boat[['Speed_mean', 'Speed_median', 'Speed_min', 'Speed_max', 'Speed_std', 'ROT_mean', 'ROT_median', 'ROT_min', 'ROT_max', 'ROT_std']]
boat_X_scaled = scaler.transform(boat_X)

#%%
"""
4.
Gradient Boost evaluation for different ship types
"""

rf_predicted_tanker_y = regr2.predict(tanker_X_scaled)
rf_MAE_tanker = mean_absolute_error(tanker_y, rf_predicted_tanker_y)
print("GB absolute mean error for Tankers is {:.3f}m".format(rf_MAE_tanker))
print("mean length of a Tanker is {:.3f}m\n".format(np.mean(tanker_y)))

rf_predicted_cargo_y = regr2.predict(cargo_X_scaled)
rf_MAE_cargo = mean_absolute_error(cargo_y, rf_predicted_cargo_y)
print("GB absolute mean error for cargo is {:.3f}m".format(rf_MAE_cargo))
print("mean length of a cargo vessel is {:.3f}m\n".format(np.mean(cargo_y)))

rf_predicted_passenger_y = regr2.predict(passenger_X_scaled)
rf_MAE_passenger = mean_absolute_error(passenger_y, rf_predicted_passenger_y)
print("GB absolute mean error for passenger is {:.3f}m".format(rf_MAE_passenger))
print("mean length of a passenger vessel is {:.3f}m\n".format(np.mean(passenger_y)))

rf_predicted_support_y = regr2.predict(support_X_scaled)
rf_MAE_support = mean_absolute_error(support_y, rf_predicted_support_y)
print("GB absolute mean error for support is {:.3f}m".format(rf_MAE_support))
print("mean length of a support vessel is {:.3f}m\n".format(np.mean(support_y)))

rf_predicted_fast_ferry_y = regr2.predict(fast_ferry_X_scaled)
rf_MAE_fast_ferry = mean_absolute_error(fast_ferry_y, rf_predicted_fast_ferry_y)
print("GB absolute mean error for fast_ferry is {:.3f}m".format(rf_MAE_fast_ferry))
print("mean length of a fast_ferry vessel is {:.3f}m".format(np.mean(fast_ferry_y)))
print("max length of a fast_ferry vessel is {:.3f}m".format(np.max(fast_ferry_y)))
print("min length of a fast_ferry vessel is {:.3f}m\n".format(np.min(fast_ferry_y)))


rf_predicted_fishing_y = regr2.predict(fishing_X_scaled)
rf_MAE_fishing = mean_absolute_error(fishing_y, rf_predicted_fishing_y)
print("GB absolute mean error for fishing is {:.3f}m".format(rf_MAE_fishing))
print("mean length of a fishing is {:.3f}m\n".format(np.mean(fishing_y)))

rf_predicted_boat_y = regr2.predict(boat_X_scaled)
rf_MAE_boat = mean_absolute_error(boat_y, rf_predicted_boat_y)
print("GB absolute mean error for boat is {:.3f}m".format(rf_MAE_boat))
print("mean length of a boat is {:.3f}m\n".format(np.mean(boat_y)))

#RESULTS with Other ships:
# GB absolute mean error for Tankers is 34.428m
# mean length of a Tanker is 155.097m

# GB absolute mean error for cargo is 24.577m
# mean length of a cargo vessel is 128.980m

# GB absolute mean error for passenger is 18.857m
# mean length of a passenger vessel is 125.975m

# GB absolute mean error for support is 16.701m
# mean length of a support vessel is 27.683m

# GB absolute mean error for fast_ferry is 23.540m
# mean length of a fast_ferry vessel is 20.345m
# max length of a fast_ferry vessel is 103.000m
# min length of a fast_ferry vessel is 13.000m

# GB absolute mean error for fishing is 18.852m
# mean length of a fishing is 18.649m

# GB absolute mean error for boat is 10.953m
# mean length of a boat is 15.540m


#RESULTS without Other Ships:

# GB absolute mean error for Tankers is 29.859m
# mean length of a Tanker is 155.097m

# GB absolute mean error for cargo is 21.716m
# mean length of a cargo vessel is 128.980m

# GB absolute mean error for passenger is 15.160m
# mean length of a passenger vessel is 125.975m

# GB absolute mean error for support is 15.701m
# mean length of a support vessel is 27.683m

# GB absolute mean error for fast_ferry is 20.308m
# mean length of a fast_ferry vessel is 20.345m
# max length of a fast_ferry vessel is 103.000m
# min length of a fast_ferry vessel is 13.000m

# GB absolute mean error for fishing is 15.876m
# mean length of a fishing is 18.649m

# GB absolute mean error for boat is 8.306m
# mean length of a boat is 15.540m