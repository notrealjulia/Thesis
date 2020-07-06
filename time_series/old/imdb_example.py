# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 15:27:29 2020

@author: JULSP
"""

import numpy
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Embedding


# (batch_size, time_steps, seq_len) ()

#%%
from tensorflow.keras.preprocessing import sequence

# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# truncate and pad input sequences
max_review_length = 500
X_train_ = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)


#%%
#compare my data

from os.path import dirname
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tensorflow import keras

path = dirname(__file__)
data = pd.read_csv(path + "/test_speed_sequence.csv")

#Ecoding the labels
lb = LabelEncoder()
data['iwrap_cat'] = lb.fit_transform(data['iwrap_type_from_dataset'])

#picking X and y and splitting
X = data[['speed_sequence']]

# X = X.T

y = data[['iwrap_cat']]
y = y.values.ravel() #somethe model wants this to be an array
unique_class = np.unique(y)
#finishing encoding
y = keras.utils.to_categorical(y)


#%%
# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#%%
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))