# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:24:54 2020

@author: JULSP

TODO: move all the model validation before train/test evlauation
TODo: get seperate results for seperate vessels types
"""

import pandas as pd
from os.path import dirname
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

import mglearn
from sklearn.metrics import mean_absolute_error
#%%
"""
Loading Data
& Pre-Processing
"""

path = dirname(__file__)
data = pd.read_csv(path + "/DYNAMIC_DAYS_test.csv") #column 3 to 12 is dynamic data

#%%
"""
dropping nan
"""

data_clean = data.drop(['trips'], axis=1)
data_clean = data_clean[data_clean.iwrap_type_from_dataset != 'Other ship']
data_clean = data_clean.dropna()

#%%

data_processed = data_clean[data_clean.length_from_data_set < 400] #vessels that are over 400m do not exist
data_processed = data_processed[data_processed.length_from_data_set > 3] #do not want to look at tiny boats
data_processed = data_processed[data_processed.Speed_mean != 0] #boats that stad still
data_processed = data_processed[data_processed.Speed_max <65] #boats that go faster than that are helicopters
data_processed = data_processed[data_processed.Speed_max >2] #boats that stand still

data_processed = data_processed.reset_index(drop =True) #reset index

#%%
"""
Picking the x and y variables
"""

# X = data_processed.iloc[:,[21,23,24,25,26]]
# y = data_processed.iloc[:,9]

X = data_processed[['Speed_mean', 'Speed_median', 'Speed_min', 'Speed_max', 'Speed_std', 'ROT_mean', 'ROT_median', 'ROT_min', 'ROT_max', 'ROT_std']]
y = data_processed[['length_from_data_set']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#%%
"""
Normalizing and Scaling
"""
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler() #accounts for outliers

scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#%%
"""
Cross validation test on LR
"""
lr_test = LinearRegression()


from sklearn.model_selection import ShuffleSplit
shuffle_split = ShuffleSplit(test_size=.2, train_size=.8, n_splits=10)
scores = cross_val_score(lr_test, X, y, cv=shuffle_split)
print("Cross-validation scores:\n{}".format(scores))
print("Cross-validation mean scores:\n{:.3f}".format(np.mean(scores)))

#RESULTS:
#Cross-validation mean scores:
#0.456 - pretty sad
#%%
"""
Linear Regression
"""
#linear regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr.predict(X_test_scaled)
print("Training set score: {:.3f}".format(lr.score(X_train_scaled, y_train)))
print("Test set score: {:.3f}\n".format(lr.score(X_test_scaled, y_test)))
#cross validation on scaled data
cross_val_scores_lr = cross_val_score(lr, X =X_train_scaled, y =y_train, cv = 5)
print("Cross-validation scores:\n {}\n".format(cross_val_scores_lr))
#mean absolute error - abismal 
lr.predict(X_test_scaled)
y_pred = lr.predict(X_test_scaled)
MAE = mean_absolute_error(y_test, y_pred)
print("absolute mean error for LR is {:.3f}m\n".format(MAE))

#linear regression on not scaled data - should not make a difference
lr2 = LinearRegression().fit(X_train, y_train)
print("Unscaled Training set score: {:.3f}".format(lr2.score(X_train, y_train)))
print("Unscaled Test set score: {:.3f}\n".format(lr2.score(X_test, y_test)))

#linear regression with regularization of w megnitude
ridge01 = Ridge(alpha=0.001).fit(X_train, y_train)
print("Ridge Training set score: {:.3f}".format(ridge01.score(X_train, y_train)))
print("Ridge Test set score: {:.3f}\n".format(ridge01.score(X_test, y_test)))

#linear regression with regularization og w amount
lasso = Lasso(alpha = 0.8).fit(X_train, y_train)
print("Lasso Training set score: {:.3f}".format(lasso.score(X_train, y_train)))
print("Lasoo Test set score: {:.3f}\n".format(lasso.score(X_test, y_test)))

print("The mean of corss validation score is {:.3f}".format(np.mean(cross_val_scores_lr)))

#%%
"""
Ensamble models
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

"""
Grid Search - for random forest
!Takes a long time
"""
forest_regr = RandomForestRegressor() #ccp_alpha - number of nodes pruned

param_grid = {'n_estimators': [ 5, 10, 35, 60, 80],
              'max_depth': [ 5, 8, 10, 20, 30],}
grid_search = GridSearchCV(forest_regr, param_grid, cv=5, n_jobs =-1)
grid_search.fit(X_train_scaled, y_train)
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# RESULTS:
# Best parameters: {'max_depth': 10, 'n_estimators': 60}
# Best cross-validation score: 0.81

#%%
"""
Grid Search continued - visualising results
"""

#convert to DF
results = pd.DataFrame(grid_search.cv_results_)
scores = np.array(results.mean_test_score).reshape(5, 5)

# plot the mean cross-validation scores
mglearn.tools.heatmap(scores, xlabel='n_estimators', xticklabels=param_grid['n_estimators'],
                      ylabel='max_depth', yticklabels=param_grid['max_depth'], cmap="viridis")

#CV_scores = cross_val_score(grid_search)
#print("Cross-validation scores: ", CV_scores)
#print("Mean cross-validation score: ", CV_scores.mean())

#“RF can achieve _% mean cross-validation accuracy on the Dynamic AIS dataset”—nothing more and nothing less.

#%%
"""
Encorporate the best hyper parameters and preform ensamble tree models & cross validation
"""
regr = RandomForestRegressor(n_estimators=60, max_depth=10, max_features =3, random_state=0, ccp_alpha =5)
                            #max features log2(number of features)
regr.fit(X_train_scaled, y_train)
print("Random Forest Training set score: {:.2f}".format(regr.score(X_train_scaled, y_train)))
print("Random Forest Test set score: {:.2f}\n".format(regr.score(X_test_scaled, y_test)))
#cross validation
cross_val_scores_regr = cross_val_score(regr, X =X_train_scaled, y =y_train, cv = 5)
print("Cross-validation scores:\n {}\n".format(cross_val_scores_regr))
print("The mean of corss validation score is {:.3f}\n".format(np.mean(cross_val_scores_regr)))

#mean absolute error
y_pred = regr.predict(X_test_scaled)
MAE_RF = mean_absolute_error(y_test, y_pred)
print("absolute mean error for LR is {:.3f}m\n".format(MAE_RF))
#TODO seperate for each class

# RESULTS:
# Random Forest Training set score: 0.83
# Random Forest Test set score: 0.80

# Cross-validation scores:
#  [0.81183482 0.81723434 0.809962   0.77736357 0.79178391]

# The mean of corss validation score is 0.802

#%%

#ignores certain features, should outpreform Random Forest
regr2 = GradientBoostingRegressor()
regr2.fit(X_train_scaled, y_train)
print("Gradient Boosting Training set score: {:.2f}".format(regr2.score(X_train_scaled, y_train)))
print("Gradient Boosting Test set score: {:.2f}\n".format(regr2.score(X_test_scaled, y_test)))
#cross_validation
cross_val_scores_regr2 = cross_val_score(regr2, X =X_train_scaled, y =y_train, cv = 5)
print("Cross-validation scores:\n {}\n".format(cross_val_scores_regr2))
print("The mean of corss validation score is {:.3f}\n".format(np.mean(cross_val_scores_regr2)))

#%%
"""
Neural Network
"""
from sklearn.neural_network import MLPRegressor

"""
Grid Search - for Neural Network
!Takes a long time
"""
nn = MLPRegressor(max_iter = 10000, random_state=0)

param_grid = {'hidden_layer_sizes': [10, 50, 100],
              'alpha': [0.0001, 0.001, 0.01],}
grid_search = GridSearchCV(nn, param_grid, cv=5, n_jobs =-1) #incormporates Cross validation, , n_jobs =-1 uses all cores
grid_search.fit(X_train_scaled, y_train)
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

#%%

"""
Grid Search continued - visualising results
"""

#convert to DF
results = pd.DataFrame(grid_search.cv_results_)
scores = np.array(results.mean_test_score).reshape(3, 3)

# plot the mean cross-validation scores
mglearn.tools.heatmap(scores, xlabel='hidden_layer_sizes', xticklabels=param_grid['hidden_layer_sizes'],
                      ylabel='alpha', yticklabels=param_grid['alpha'], cmap="viridis")

#%%
nn = MLPRegressor(max_iter = 10000, random_state=0, hidden_layer_sizes=50, alpha = 0.0001)
nn.fit(X_train, y_train)
print("Neural Network Training set score: {:.2f}".format(nn.score(X_train, y_train)))
print("Neural Network Test set score: {:.2f}\n".format(nn.score(X_test, y_test)))

cross_val_scores_nn = cross_val_score(nn, X =X_train_scaled, y =y_train, cv = 5, n_jobs =-1)
print("Cross-validation scores:\n {}\n".format(cross_val_scores_nn))
print("The mean of corss validation score is {:.3f}\n".format(np.mean(cross_val_scores_nn)))

#%%

"""
How well are the Regression models working for each ship type?
"""
# lr
# regr
# regr2
# nn

print(data_processed.iwrap_type_from_dataset.unique())

#tanker = data_processed.iloc[:,9]
tanker = data_processed[data_processed['iwrap_type_from_dataset'] == 'Oil products tanker']
cargo = data_processed[data_processed['iwrap_type_from_dataset'] == 'General cargo ship']
passenger = data_processed[data_processed['iwrap_type_from_dataset'] == 'Passenger ship']
support = data_processed[data_processed['iwrap_type_from_dataset'] == 'Support ship']
fast_ferry = data_processed[data_processed['iwrap_type_from_dataset'] == 'Fast ferry']
fishing = data_processed[data_processed['iwrap_type_from_dataset'] == 'Fishing ship']
boat = data_processed[data_processed['iwrap_type_from_dataset'] == 'Pleasure boat']


tanker_y = tanker.iloc[:,9]
tanker_X =  tanker.iloc[:,[21,23,24,25,26]]
tanker_X_train, tanker_X_test, tanker_y_train, tanker_y_test = train_test_split(tanker_X, tanker_y, test_size=0.2, random_state=0)

cargo_y = cargo.iloc[:,9]
cargo_X =  cargo.iloc[:,[21,23,24,25,26]]

passenger_y = passenger.iloc[:,9]
passenger_X =  passenger.iloc[:,[21,23,24,25,26]]

support_y = support.iloc[:,9]
support_X =  support.iloc[:,[21,23,24,25,26]]

fast_ferry_y = fast_ferry.iloc[:,9]
fast_ferry_X =  fast_ferry.iloc[:,[21,23,24,25,26]]

fishing_y = fishing.iloc[:,9]
fishing_X =  fishing.iloc[:,[21,23,24,25,26]]

boat_y = boat.iloc[:,9]
boat_X =  boat.iloc[:,[21,23,24,25,26]]


lr_predicted_tanker_y = lr.predict(tanker_X_test)
lr_MAE_tanker = mean_absolute_error(tanker_y_test, lr_predicted_tanker_y)
print("absolute mean error for Tanker LR is {:.3f}m\n".format(lr_MAE_tanker))
#%%
"""
Neural network Model evaluation for different ship types
Maybe change to test set
"""

nn_predicted_tanker_y = nn.predict(tanker_X_test)
nn_MAE_tanker = mean_absolute_error(tanker_y_test, nn_predicted_tanker_y)
print("absolute mean error for Tanker NN is {:.3f}m".format(nn_MAE_tanker))
print("mean length of a Tanker is {:.3f}m\n".format(np.mean(tanker_y)))

nn_predicted_cargo_y = nn.predict(cargo_X)
nn_MAE_cargo = mean_absolute_error(cargo_y, nn_predicted_cargo_y)
print("absolute mean error for Cargo NN is {:.3f}m".format(nn_MAE_cargo))
print("mean length of a Cargo is {:.3f}m\n".format(np.mean(cargo_y)))

nn_predicted_passenger_y = nn.predict(passenger_X)
nn_MAE_passenger = mean_absolute_error(passenger_y, nn_predicted_passenger_y)
print("absolute mean error for Passenger NN is {:.3f}m".format(nn_MAE_passenger))
print("mean length of a Passenger is {:.3f}m\n".format(np.mean(passenger_y)))

nn_predicted_support_y = nn.predict(support_X)
nn_MAE_support = mean_absolute_error(support_y, nn_predicted_support_y)
print("absolute mean error for Support NN is {:.3f}m".format(nn_MAE_support))
print("mean length of a Support is {:.3f}m\n".format(np.mean(support_y)))

nn_predicted_fast_ferry_y = nn.predict(fast_ferry_X)
nn_MAE_fast_ferry = mean_absolute_error(fast_ferry_y, nn_predicted_fast_ferry_y)
print("absolute mean error for Fast Ferry NN is {:.3f}m".format(nn_MAE_fast_ferry))
print("mean length of a Fast Ferry is {:.3f}m\n".format(np.mean(fast_ferry_y)))

nn_predicted_fishing_y = nn.predict(fishing_X)
nn_MAE_fishing = mean_absolute_error(fishing_y, nn_predicted_fishing_y)
print("absolute mean error for Fishing NN is {:.3f}m".format(nn_MAE_fishing))
print("mean length of a Fishing is {:.3f}m\n".format(np.mean(fishing_y)))

nn_predicted_boat_y = nn.predict(boat_X)
nn_MAE_boat = mean_absolute_error(boat_y, nn_predicted_boat_y)
print("absolute mean error for Pleasure Boat NN is {:.3f}m".format(nn_MAE_boat))
print("mean length of a Pleasure Boat is {:.3f}m\n".format(np.mean(boat_y)))