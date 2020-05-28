# -*- coding: utf-8 -*-
"""
Created on Tue May 26 20:48:49 2020

@author: JULSP

Classification: predicting vessel types from movement features

INSTRUCTIONS: run all cells sequentially, wait until each cell has completed before running the next one
"""

import pandas as pd
from os.path import dirname
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

#%%
"""
0.
Loading Data
& Pre-Processing
"""
path = dirname(__file__)

data_all = pd.read_csv(path + "/data_jun_jul_aug_sep_oct.csv")
data_clean = data_all.drop(['trips'], axis=1)
#data_clean = data_clean[data_clean.iwrap_type_from_dataset != 'Other ship']
data_clean = data_clean.dropna() #drop nan values, our models can't handle this
data_processed = data_clean[data_clean.length_from_data_set < 400] #vessels that are over 400m do not exist
data_processed = data_processed[data_processed.Speed_max <65] #boats that go faster than that are helicopters
data_processed = data_processed[data_processed.Speed_max >0] #boats that stand still
data_processed = shuffle(data_processed)
data_processed = data_processed.reset_index(drop =True) #reset index
types_of_vessels = data_processed.iwrap_type_from_dataset.value_counts()
print("These are the ship type classes with the amount in each class\n\n", types_of_vessels)

#%%
"""
1. 
Establishing a baseline - Dummy model
"""
#Ecoding the labels
lb = preprocessing.LabelBinarizer() #one-hot encoding for labels, tp avoid sequential relationship
labels = lb.fit_transform(data_processed['iwrap_type_from_dataset'])
X = data_processed[['Speed_mean', 'Speed_median', 'Speed_min', 'Speed_max', 'Speed_std', 'ROT_mean', 'ROT_median', 'ROT_min', 'ROT_max', 'ROT_std']]
y = labels

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

#Dummy classifier
dummy_clf = DummyClassifier(strategy="stratified")
dummy_clf.fit(X_train_scaled, y_train)
y_pred = dummy_clf.predict(X_test_scaled)
print(classification_report(y_test, y_pred, target_names=lb.classes_))

"""
RESULTS:
                     precision    recall  f1-score   support

         Fast ferry       0.00      0.00      0.00        39
       Fishing ship       0.03      0.03      0.03       147
 General cargo ship       0.32      0.32      0.32      2959
Oil products tanker       0.09      0.09      0.09       874
         Other ship       0.11      0.11      0.11      1002
     Passenger ship       0.15      0.15      0.15      1409
      Pleasure boat       0.27      0.26      0.26      2659
       Support ship       0.05      0.05      0.05       510

          micro avg       0.22      0.22      0.22      9599
          macro avg       0.13      0.13      0.13      9599
       weighted avg       0.22      0.22      0.22      9599
        samples avg       0.15      0.22      0.17      9599
"""
#%%
"""
2. 
Neural Network - Takes some minutes to run and will 
"""

#Tuning hyper parameters - takes a while
# nn_class = MLPClassifier(random_state =0, max_iter = 20000)

# param_grid = {'hidden_layer_sizes': [(100,50,10), (100,100), (100,50)],
#               'alpha': [0.1, 0.05, 0.01],}

# grid_search = GridSearchCV(nn_class, param_grid, cv=5, n_jobs =-1) #incorporates Cross validation, , n_jobs =-1 uses all PC cores
# grid_search.fit(X_valid_scaled, y_valid)
# print("Best parameters: {}".format(grid_search.best_params_))
# print("Mean cross-validated score of the best_estimator: {:.2f}".format(grid_search.best_score_))
# best_parameters = grid_search.best_params_

# #Multi-layer preceptron with best parameters found through grid search
# nn_class_model = MLPClassifier(random_state =0, max_iter = 20000, **best_parameters)

nn_class_model = MLPClassifier(random_state =0, max_iter = 20000, hidden_layer_sizes = (100,100), alpha = 0.1)

#Training and test 
nn_class_model.fit(X_train_scaled, y_train)
print("Training set score: {:.3f}".format(nn_class_model.score(X_train_scaled, y_train)))
print("Test set score: {:.3f}\n".format(nn_class_model.score(X_test_scaled, y_test)))

#Classification report
print(classification_report(y_test, nn_class_model.predict(X_test_scaled), target_names=lb.classes_))
print('Precision is positive predictive value \nRecall is true positive rate or sensitivity \n')

"""
RESULTS:

Training set score: 0.731
Test set score: 0.719

                     precision    recall  f1-score   support

         Fast ferry       0.88      0.38      0.54        39
       Fishing ship       0.37      0.07      0.11       147
 General cargo ship       0.70      0.81      0.75      2959
Oil products tanker       0.48      0.37      0.42       874
         Other ship       0.61      0.49      0.54      1002
     Passenger ship       0.78      0.72      0.75      1409
      Pleasure boat       0.80      0.93      0.86      2659
       Support ship       0.61      0.35      0.45       510

           accuracy                           0.72      9599
          macro avg       0.65      0.52      0.55      9599
       weighted avg       0.70      0.72      0.70      9599
"""
#%%
"""
3. 
K-nearest neighbor
"""

#validation and grid search
knn = KNeighborsClassifier()       
param_grid = {'n_neighbors': [4, 5, 6, 7],
              'leaf_size': [30, 35, 10]}

grid_search = GridSearchCV(knn, param_grid, cv=5, n_jobs =-1) #incorporates Cross validation, , n_jobs =-1 uses all PC cores
grid_search.fit(X_valid_scaled, y_valid)
print("Best parameters: {}".format(grid_search.best_params_))
print("Mean cross-validated score of the best_estimator: {:.2f}".format(grid_search.best_score_))
best_parameters_knn = grid_search.best_params_
 
#knn with best parameters found through grid search           
knn_model = KNeighborsClassifier(**best_parameters_knn, weights = 'distance')

#Training and test
knn_model.fit(X_train_scaled, y_train)
print("Training set score: {:.3f}".format(knn_model.score(X_train_scaled, y_train)))
print("Test set score: {:.3f}\n".format(knn_model.score(X_test_scaled, y_test)))
y_pred = knn_model.predict(X_test_scaled)

#Classification report
print(classification_report(y_test, y_pred, target_names=lb.classes_))
print('Precision is positive predictive value \nRecall is true positive rate or sensitivity \n')

"""
RESULTS:

Training set score: 1.000
Test set score: 0.692

                     precision    recall  f1-score   support

         Fast ferry       0.79      0.38      0.52        39
       Fishing ship       0.39      0.11      0.17       147
 General cargo ship       0.69      0.80      0.74      2959
Oil products tanker       0.46      0.34      0.39       874
         Other ship       0.61      0.43      0.50      1002
     Passenger ship       0.73      0.66      0.70      1409
      Pleasure boat       0.76      0.90      0.83      2659
       Support ship       0.60      0.38      0.46       510

           accuracy                           0.69      9599
          macro avg       0.63      0.50      0.54      9599
       weighted avg       0.68      0.69      0.68      9599
"""
#%%
"""
4.
TesnorFlow a more complicated neural network
"""

import pandas as pd
from os.path import dirname
import numpy as np
import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
from sklearn.utils import class_weight

#%%

"""
Different encoding with weights for imbalanced classes
"""

from sklearn.preprocessing import LabelEncoder
#Ecoding the labels
lb = LabelEncoder()
data_processed['iwrap_cat'] = lb.fit_transform(data_processed['iwrap_type_from_dataset'])
#picking X and y and splitting
X = data_processed[['Speed_mean', 'Speed_median', 'Speed_min', 'Speed_max', 'Speed_std', 'ROT_mean', 'ROT_median', 'ROT_min', 'ROT_max', 'ROT_std']]
y = data_processed[['iwrap_cat']]
y = y.values.ravel() #somethe model wants this to be an array
unique_class = np.unique(y)

#calculating class weights
class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = unique_class, y=y)

#need to use categorical_crossentropy as the loss function with categorical classes
y = keras.utils.to_categorical(y)

#split into test and train+val 
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2,  random_state=0, stratify = y)
# split train+validation set into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=1, stratify = y_trainval)

print("\nSize of training set: {}   size of validation set: {}   size of test set:" " {}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

#%%

#create model

model = Sequential()
#get number of columns in training data
n_cols = X_train.shape[1]
#add model layers
model.add(keras.layers.Flatten(input_shape = [10]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(32, activation="relu"))
model.add(keras.layers.Dense(24, activation="relu"))
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(8, activation="softmax"))

#compile model using mse as a measure of model performance
model.compile(loss="categorical_crossentropy", #don't change the loss function
              optimizer="adam",
              metrics=["accuracy"])

#set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=10)


#%%

#train model

history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=30, class_weight=class_weights, callbacks=[early_stopping_monitor])

#Ends on
#Epoch 21/30
#28797/28797 [==============================] - 5s 165us/sample - loss: 0.7868 - accuracy: 0.7209 - val_loss: 0.8270 - val_accuracy: 0.7114

#%%

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.xlabel("Epochs")
plt.title("Learning rate of a Neural Network built on TensorFlow")
plt.show()

#%%

print('\n# Evaluate on test data')
results = model.evaluate(X_test, y_test)
print('test loss, test acc:', results)

#%%

y_pred = model.predict(X_test)

#Classification report
print(classification_report(y_test, y_pred, target_names=lb.classes_))
print('Precision is positive predictive value \nRecall is true positive rate or sensitivity \n')