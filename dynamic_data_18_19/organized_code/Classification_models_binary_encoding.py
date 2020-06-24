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
from sklearn.neural_network import MLPClassifier

import mglearn

#%%
"""
0.
Loading Data
& Pre-Processing
"""

path = dirname(__file__)

data_all = pd.read_csv(path + "/data_jun_jul_aug_sep_oct_withdate.csv")

data_clean = data_all.drop(['trips'], axis=1)
#data_clean = data_clean[data_clean.iwrap_type_from_dataset != 'Other ship']
data_clean = data_clean.dropna() #drop nan values, our models can't handle this
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
NEURAL NETWORK  and KNN doesn't need this step
"""

amount_limit = 1000

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

#Ecoding the labels with one-hot-encoder
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
labels = lb.fit_transform(data_processed['iwrap_type_from_dataset'])

#Ecoding the labels
# lb = LabelEncoder()
# data_processed['iwrap_cat'] = lb.fit_transform(data_processed['iwrap_type_from_dataset'])
#picking X and y and splitting
X = data_processed[['Speed_mean', 'Speed_median', 'Speed_min', 'Speed_max', 'Speed_std', 'ROT_mean', 'ROT_median', 'ROT_min', 'ROT_max', 'ROT_std']]
# y = data_processed[['iwrap_cat']]
# y = y.values.ravel() #somethe model wants this to be an array

y = lb.inverse_transform(labels)

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

print(lb.classes_)


#%%
"""
1.
Logistic Regression 
"""
logreg = LogisticRegression(max_iter=10000,C=0.5)
# Validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
corss_val_log = cross_val_score(logreg, X_valid_scaled, y_valid, cv=kfold)
print("Cross-validation scores for Logistic reg:\n{}".format(corss_val_log))
print("Mean cross validation for Logistic reg {:.3f}".format(np.mean(corss_val_log)))

#Training and test
logreg.fit(X_train_scaled, y_train)
print("Training set score: {:.3f}".format(logreg.score(X_train_scaled, y_train)))
print("Test set score: {:.3f}\n".format(logreg.score(X_test_scaled, y_test)))
y_pred = logreg.predict(X_test_scaled)

#Classification report
print(classification_report(y_test, y_pred, target_names=lb.classes_))
print('Precision is positive predictive value \nRecall is true positive rate or sensitivity \n')

# RESULTS:
## jun+ july + aug +sept + oct, with limit of 1000 vessels per category 
# Cross-validation scores for Logistic reg:
# [0.37410072 0.35018051 0.36462094 0.4368231  0.39350181]
# Mean cross validation for Logistic reg 0.384
# Training set score: 0.386
# Test set score: 0.403

#                      precision    recall  f1-score   support

#          Fast ferry       0.55      0.28      0.37        40
#        Fishing ship       0.45      0.36      0.40       147
#  General cargo ship       0.36      0.63      0.46       200
# Oil products tanker       0.39      0.34      0.36       200
#          Other ship       0.36      0.29      0.32       200
#      Passenger ship       0.41      0.31      0.35       200
#       Pleasure boat       0.51      0.67      0.58       200
#        Support ship       0.33      0.24      0.28       200

#            accuracy                           0.40      1387
#           macro avg       0.42      0.39      0.39      1387
#        weighted avg       0.40      0.40      0.39      1387

# Precision is positive predictive value 
# Recall is true positive rate or sensitivity 


#WITHOUT THE 1000 vessel limit
# Cross-validation scores for Logistic reg:
# [0.5850939  0.58568075 0.58661186 0.59953024 0.57956547]
# Mean cross validation for Logistic reg 0.587
# Training set score: 0.578
# Test set score: 0.584

#                      precision    recall  f1-score   support

#          Fast ferry       0.17      0.03      0.05        36
#        Fishing ship       0.00      0.00      0.00       133
#  General cargo ship       0.58      0.90      0.71      2679
# Oil products tanker       0.00      0.00      0.00       788
#          Other ship       0.46      0.26      0.33       925
#      Passenger ship       0.42      0.22      0.29      1256
#       Pleasure boat       0.64      0.91      0.75      2237
#        Support ship       0.50      0.00      0.01       463

#            accuracy                           0.58      8517
#           macro avg       0.35      0.29      0.27      8517
#        weighted avg       0.49      0.58      0.50      8517

#%%
# """
# 1.
# Probabilistic results for test dataset
# TODO use this later for predicting Unidentified vessels
# """
# prob_test_scores_logreg = logreg.predict_proba(X_test)

# print("Predicted probabilities:\n{:.2f}%".format, prob_test_scores_logreg[:6])
# print("Sums: {}".format(logreg.predict_proba(X_test)[:6].sum(axis=1)))
#%%
"""
2.
SVM 
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

#RESULTS
# Mean cross-validated score of the best_estimator: 0.18
# Best parameters:  {'C': 0.001, 'gamma': 0.001}

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
print(classification_report(y_test, svm_best.predict(X_test_scaled), target_names=lb.classes_))
print('Precision is positive predictive value \nRecall is true positive rate or sensitivity \n')

#RESULTS WITH 1000 limit
# Training set score: 0.162
# Test set score: 0.163

#                      precision    recall  f1-score   support

#          Fast ferry       0.00      0.00      0.00        40
#        Fishing ship       0.00      0.00      0.00       147
#  General cargo ship       0.15      1.00      0.27       200
# Oil products tanker       0.00      0.00      0.00       200
#          Other ship       0.22      0.02      0.04       200
#      Passenger ship       0.00      0.00      0.00       200
#       Pleasure boat       0.28      0.11      0.16       200
#        Support ship       0.00      0.00      0.00       200

#            accuracy                           0.16      1387
#           macro avg       0.08      0.14      0.06      1387
#        weighted avg       0.10      0.16      0.07      1387

#%%
"""
3.
Random Forest Classifier
Hyper-paremeter tuning
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

#RESULTS: with limit of 1000
# Best parameters: {'ccp_alpha': 2, 'max_depth': 15, 'max_features': 2, 'n_estimators': 70}
# Mean cross-validated score of the best_estimator: 0.14



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
print(classification_report(y_test, forest_class_best.predict(X_test_scaled), target_names=lb.classes_))
print('Precision is positive predictive value \nRecall is true positive rate or sensitivity \n')

#RESULTS with 1000 limit
# Random Forest Training set score: 0.15
# Random Forest Test set score: 0.15

#                      precision    recall  f1-score   support

#          Fast ferry       0.00      0.00      0.00        40
#        Fishing ship       0.00      0.00      0.00       147
#  General cargo ship       0.00      0.00      0.00       200
# Oil products tanker       0.14      1.00      0.25       200
#          Other ship       0.00      0.00      0.00       200
#      Passenger ship       0.00      0.00      0.00       200
#       Pleasure boat       0.00      0.00      0.00       200
#        Support ship       0.00      0.00      0.00       200

#            accuracy                           0.14      1387
#           macro avg       0.02      0.12      0.03      1387
#        weighted avg       0.02      0.14      0.04      1387

# Precision is positive predictive value 
# Recall is true positive rate or sensitivity 


#%%
"""
4.
NN
"""

nn_class = MLPClassifier(random_state =0, max_iter = 20000)

"""
Neural Network
Grid Search - for Neural Network
!Takes a long time
"""


param_grid = {'hidden_layer_sizes': [(100,50,10), (100,100), (100,50)],
              'alpha': [0.1, 0.05, 0.01],}

grid_search = GridSearchCV(nn_class, param_grid, cv=5, n_jobs =-1) #incorporates Cross validation, , n_jobs =-1 uses all PC cores
grid_search.fit(X_valid_scaled, y_valid)
print("Best parameters: {}".format(grid_search.best_params_))
print("Mean cross-validated score of the best_estimator: {:.2f}".format(grid_search.best_score_))
best_parameters3 = grid_search.best_params_

# Best parameters: {'alpha': 0.1, 'hidden_layer_sizes': (100, 50, 10)}
# Mean cross-validated score of the best_estimator: 0.64
#%%

nn_class = MLPClassifier(random_state =0, max_iter = 20000, hidden_layer_sizes = (100,100), alpha = 0.1)


#Training and test 
#NEEDS more data overfits
nn_class.fit(X_train_scaled, y_train)
print("Training set score: {:.3f}".format(nn_class.score(X_train_scaled, y_train)))
print("Test set score: {:.3f}\n".format(nn_class.score(X_test_scaled, y_test)))

#Classification report
print(classification_report(y_test, nn_class.predict(X_test_scaled), target_names=lb.classes_))
print('Precision is positive predictive value \nRecall is true positive rate or sensitivity \n')

# RESULTS without the limit wit architecture 100,100
# Training set score: 0.735
# Test set score: 0.713

#                      precision    recall  f1-score   support

#          Fast ferry       0.67      0.31      0.42        39
#        Fishing ship       0.56      0.13      0.21       147
#  General cargo ship       0.67      0.88      0.76      2959
# Oil products tanker       0.58      0.22      0.32       874
#          Other ship       0.60      0.45      0.52      1002
#      Passenger ship       0.80      0.67      0.73      1409
#       Pleasure boat       0.78      0.94      0.85      2659
#        Support ship       0.71      0.29      0.41       510

#            accuracy                           0.71      9599
#           macro avg       0.67      0.48      0.53      9599
#        weighted avg       0.70      0.71      0.69      9599

# Precision is positive predictive value 
# Recall is true positive rate or sensitivity  


#%%

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

#Validation
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


#validation and grid search
        
param_grid = {'n_neighbors': [4, 5, 6, 7],
              'leaf_size': [30, 35, 10]}

grid_search = GridSearchCV(knn, param_grid, cv=5, n_jobs =-1) #incorporates Cross validation, , n_jobs =-1 uses all PC cores
grid_search.fit(X_valid_scaled, y_valid)
print("Best parameters: {}".format(grid_search.best_params_))
print("Mean cross-validated score of the best_estimator: {:.2f}".format(grid_search.best_score_))
best_parameters_knn = grid_search.best_params_
            

knn_model = KNeighborsClassifier(**best_parameters_knn, weights = 'distance')

# corss_val_knn = cross_val_score(knn, X_valid_scaled, y_valid, cv=kfold)
# print("Cross-validation scores for Logistic reg:\n{}".format(corss_val_knn))
# print("Mean cross validation for Logistic reg {:.3f}".format(np.mean(corss_val_knn)))

#Training and test
knn_model.fit(X_train_scaled, y_train)
print("Training set score: {:.3f}".format(knn_model.score(X_train_scaled, y_train)))
print("Test set score: {:.3f}\n".format(knn_model.score(X_test_scaled, y_test)))
y_pred = knn_model.predict(X_test_scaled)

#Classification report
print(classification_report(y_test, y_pred, target_names=lb.classes_))
print('Precision is positive predictive value \nRecall is true positive rate or sensitivity \n')

# Best parameters: {'leaf_size': 30, 'n_neighbors': 7}
# Mean cross-validated score of the best_estimator: 0.67
# Training set score: 1.000
# Test set score: 0.683

#                      precision    recall  f1-score   support

#          Fast ferry       0.68      0.33      0.45        39
#        Fishing ship       0.33      0.12      0.17       147
#  General cargo ship       0.68      0.80      0.74      2959
# Oil products tanker       0.45      0.30      0.36       874
#          Other ship       0.57      0.39      0.46      1002
#      Passenger ship       0.72      0.65      0.68      1409
#       Pleasure boat       0.75      0.91      0.82      2659
#        Support ship       0.59      0.36      0.45       510

#            accuracy                           0.68      9599
#           macro avg       0.60      0.48      0.52      9599
#        weighted avg       0.66      0.68      0.66      9599

# Precision is positive predictive value 
# Recall is true positive rate or sensitivity  