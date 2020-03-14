# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 17:14:51 2020

@author: JULSP
"""

import pandas as pd
from os.path import dirname
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import mglearn
from sklearn.metrics import mean_absolute_error

#%%
"""
Loading Data
& Pre-Processing
"""

path = dirname(__file__)
data = pd.read_csv(path + "/DYNAMIC_DAYS_test.csv") #column 3 to 12 is dynamic data

data_clean = data.drop(['trips'], axis=1)
#data_clean = data_clean[data_clean.iwrap_type_from_dataset != 'Other ship']
data_clean = data_clean.dropna() #drop nan values, our models can't handle this
data_processed = data_clean[data_clean.length_from_data_set < 400] #vessels that are over 400m do not exist
data_processed = data_processed[data_processed.length_from_data_set > 3] #do not want to look at tiny boats
data_processed = data_processed[data_processed.Speed_mean != 0] #boats that stad still
data_processed = data_processed[data_processed.Speed_max <65] #boats that go faster than that are helicopters
data_processed = data_processed[data_processed.Speed_max >2] #boats that stand still

data_processed = data_processed.reset_index(drop =True) #reset index

#%%
"""
Limiting each amount of vessels to 80? for each vessel type
"""

types_of_vessels = data_processed.iwrap_type_from_dataset.value_counts()
print(data_processed.iwrap_type_from_dataset.unique())

amount_limit = 80

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

#%%
"""
Ecoding the labels
Picking X and y variables
Splitting trai/test
Scaling
"""

#Ecoding the labels
lb = LabelEncoder()
data_processed['iwrap_cat'] = lb.fit_transform(data_processed['iwrap_type_from_dataset'])
#picking X and y and splitting
X = data_processed[['Speed_mean', 'Speed_median', 'Speed_min', 'Speed_max', 'Speed_std', 'ROT_mean', 'ROT_median', 'ROT_min', 'ROT_max', 'ROT_std']]
y = data_processed[['iwrap_cat']]
y = y.values.ravel() #somethe model wants this to be an array


X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, random_state=0)
# split train+validation set into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(
    X_trainval, y_trainval, random_state=1)

print("Size of training set: {}   size of validation set: {}   size of test set:" " {}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))

#Scaling the data
scaler = RobustScaler() #accounts for outliers
scaler.fit(X_trainval)
X_trainval_scaled = scaler.transform(X_trainval)
X_valid_scaled = scaler.transform(X_valid)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
#%%
"""
Logistic Regression 
"""
logreg = LogisticRegression(max_iter=10000)
#Validation
kfold = StratifiedKFold(n_splits=7, shuffle=True, random_state=0)
print("Cross-validation scores:\n{}".format(
    cross_val_score(logreg, X_valid_scaled, y_valid, cv=kfold)))
#Training and test
logreg.fit(X_trainval_scaled, y_trainval)
print("Training set score: {:.3f}".format(logreg.score(X_trainval_scaled, y_trainval)))
print("Test set score: {:.3f}".format(logreg.score(X_test_scaled, y_test)))

#%%
prob_test_scores_logreg = logreg.predict_proba(X_test)

#print("Predicted probabilities:\n{:.2f}%".format, prob_test_scores)
print("Sums: {}".format(logreg.predict_proba(X_test)[:6].sum(axis=1)))

#%%
"""
SVM -remake like log reg
"""

from sklearn.svm import SVC

# X_trainval, X_test, y_trainval, y_test = train_test_split(
#     X, y, random_state=0)
# # split train+validation set into training and validation sets
# X_train, X_valid, y_train, y_valid = train_test_split(
#     X_trainval, y_trainval, random_state=1)

# print("Size of training set: {}   size of validation set: {}   size of test set:"
#       " {}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))

best_score = 0

for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        # for each combination of parameters, train an SVC
        svm = SVC(gamma=gamma, C=C)
        svm.fit(X_train_scaled, y_train)
        # evaluate the SVC on the test set
        score = svm.score(X_valid, y_valid)
        # if we got a better score, store the score and parameters
        if score > best_score:
            best_score = score
            best_parameters = {'C': C, 'gamma': gamma}

# rebuild a model on the combined training and validation set,
# and evaluate it on the test set
svm = SVC(**best_parameters)
svm.fit(X_trainval_scaled, y_trainval)
test_score = svm.score(X_test_scaled, y_test)
print("Best score on validation set: {:.2f}".format(best_score))
print("Best parameters: ", best_parameters)
print("Test set score with best parameters: {:.2f}".format(test_score))

"""
TRees
NN
"""
