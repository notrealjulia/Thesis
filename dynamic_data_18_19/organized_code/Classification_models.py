# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 17:14:51 2020

@author: JULSP
"""

import pandas as pd
import sklearn 
import numpy as np

from os.path import dirname

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC


from sklearn.metrics import classification_report
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import mglearn

#%%
"""
0.
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
0.
Limiting each amount of vessels to 80? for each vessel type to balance the data 
DON't do this for SVM
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
y = data_processed[['iwrap_cat']]
y = y.values.ravel() #somethe model wants this to be an array


#split into test and train+val 
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2,  random_state=0)
# split train+validation set into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=1)

print("\nSize of training set: {}   size of validation set: {}   size of test set:" " {}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))

#Scaling the data
scaler = RobustScaler() #accounts for outliers
scaler.fit(X_trainval)
X_trainval_scaled = scaler.transform(X_trainval)
X_valid_scaled = scaler.transform(X_valid)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

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
Logistic Regression 
"""
logreg = LogisticRegression(max_iter=10000)
#Validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
corss_val_log = cross_val_score(logreg, X_valid_scaled, y_valid, cv=kfold)
print("Cross-validation scores for Logistic reg:\n{}".format(corss_val_log))
print("Mean cross validation for Logistic reg {:.3f}".format(np.mean(corss_val_log)))

#Training and test
logreg.fit(X_train_scaled, y_train)
print("Training set score: {:.3f}".format(logreg.score(X_train_scaled, y_train)))
print("Test set score: {:.3f}\n".format(logreg.score(X_test_scaled, y_test)))

#Classification report
print(classification_report(y_test, logreg.predict(X_test_scaled), target_names=target_names))
print('Precision is positive predictive value \nRecall is true positive rate or sensitivity \n')

# RESULTS:

# Cross-validation scores for Logistic reg:
# [0.29166667 0.33333333 0.375      0.375      0.52173913]
# Mean cross validation for Logistic reg 0.379
# Training set score: 0.377
# Test set score: 0.286

#                      precision    recall  f1-score   support

#          Fast ferry       0.50      0.40      0.44         5
#        Fishing ship       0.25      0.38      0.30        16
#  General cargo ship       0.21      0.18      0.19        17
# Oil products tanker       0.18      0.43      0.26        14
#          Other ship       0.38      0.23      0.29        13
#      Passenger ship       0.56      0.26      0.36        19
#       Pleasure boat       0.55      0.33      0.41        18
#        Support ship       0.19      0.18      0.18        17

#            accuracy                           0.29       119
#           macro avg       0.35      0.30      0.30       119
#        weighted avg       0.35      0.29      0.29       119

# Precision is positive predictive value 
# Recall is true positive rate or sensitivity 

#%%
"""
1.
Probabilistic results for test dataset
"""
prob_test_scores_logreg = logreg.predict_proba(X_test)

print("Predicted probabilities:\n{:.2f}%".format, prob_test_scores_logreg[:6])
print("Sums: {}".format(logreg.predict_proba(X_test)[:6].sum(axis=1)))
#%%
"""
2.
SVM 
TODO look up kernel trick
"""

best_score = 0
#validation
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        # for each combination of parameters, train an SVC
        svm = SVC(gamma=gamma, C=C, random_state = 0)
        svm.fit(X_valid_scaled, y_valid)
        # evaluate the SVC on the test set
        score = svm.score(X_valid, y_valid)
        # if we got a better score, store the score and parameters
        if score > best_score:
            best_score = score
            best_parameters = {'C': C, 'gamma': gamma}
            
print("Mean cross-validated score of the best_estimator: {:.2f}".format(best_score))
print("Best parameters: ", best_parameters)

#%%
"""
2.
SVM
"""
# rebuild a model on best parameters
svm_best = SVC(**best_parameters, random_state = 0)

#Training and test - is it useless?
svm_best.fit(X_train_scaled, y_train)
print("Training set score: {:.3f}".format(svm_best.score(X_train_scaled, y_train)))
print("Test set score: {:.3f}\n".format(svm_best.score(X_test_scaled, y_test)))

#Classification report
print(classification_report(y_test, svm_best.predict(X_test_scaled), target_names=target_names))
print('Precision is positive predictive value \nRecall is true positive rate or sensitivity \n')

#%%
"""
3.
Random Forest Classifier
Validation
"""
from sklearn.ensemble import RandomForestClassifier

forest_class = RandomForestClassifier(random_state = 42)

#parameteres for RF
param_grid = {'n_estimators': [ 70, 80, 90],
              'max_depth': [15, 20, 35],
              'max_features': [2, 3, 5],
              'ccp_alpha': [2, 3, 5]}

grid_search = GridSearchCV(forest_class, param_grid, cv=5, n_jobs =-1)
grid_search.fit(X_valid_scaled, y_valid)
print("Best parameters: {}".format(grid_search.best_params_))
best_parameters = grid_search.best_params_
print("Mean cross-validated score of the best_estimator: {:.2f}".format(grid_search.best_score_))

#%%
"""
3. 
Random Forest
Classification report
"""

forest_class_best = RandomForestClassifier(**best_parameters, random_state = 42)

#Train/test
forest_class_best.fit(X_train_scaled, y_train)
print("Random Forest Training set score: {:.2f}".format(forest_class_best.score(X_train_scaled, y_train)))
print("Random Forest Test set score: {:.2f}\n".format(forest_class_best.score(X_test_scaled, y_test)))

#Classification report
print(classification_report(y_test, forest_class_best.predict(X_test_scaled), target_names=target_names))
print('Precision is positive predictive value \nRecall is true positive rate or sensitivity \n')

#%%
"""
4.
NN
"""
from sklearn.neural_network import MLPClassifier
nn_class = MLPClassifier(random_state =0, max_iter = 20000)

#%%
#Training and test 
#NEEDS more data overfits
nn_class.fit(X_train_scaled, y_train)
print("Training set score: {:.3f}".format(nn_class.score(X_train_scaled, y_train)))
print("Test set score: {:.3f}\n".format(nn_class.score(X_test_scaled, y_test)))

#Classification report
print(classification_report(y_test, nn_class.predict(X_test_scaled), target_names=target_names))
print('Precision is positive predictive value \nRecall is true positive rate or sensitivity \n')

# RESULTS: with limiting each class
# Training set score: 0.972
# Test set score: 0.345

#                      precision    recall  f1-score   support

#          Fast ferry       0.44      0.67      0.53         6
#        Fishing ship       0.30      0.19      0.23        16
#  General cargo ship       0.29      0.17      0.22        23
# Oil products tanker       0.40      0.59      0.48        17
#          Other ship       0.30      0.38      0.33        16
#      Passenger ship       0.42      0.36      0.38        14
#       Pleasure boat       0.38      0.60      0.46        10
#        Support ship       0.23      0.18      0.20        17

#            accuracy                           0.34       119
#           macro avg       0.34      0.39      0.35       119
#        weighted avg       0.33      0.34      0.33       119

# Precision is positive predictive value 
# Recall is true positive rate or sensitivity 
