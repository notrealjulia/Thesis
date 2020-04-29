# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:19:12 2020

@author: JULSP
"""

import tensorflow as tf
from tensorflow import keras

#%%
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

#%%
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

#%%
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

#%%
history = model.fit(X_train, y_train, epochs=30,
                     validation_data=(X_valid, y_valid))

#%%
model.evaluate(X_test, y_test)

#%%
X_new = X_test[:3]
y_proba = model.predict(X_new)

y_new = y_test[:3]