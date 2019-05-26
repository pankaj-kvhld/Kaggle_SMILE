# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from sklearn.model_selection import train_test_split

batch_size = 128
epochs = 12

DATA_DIR = Path(__file__).parent.parent / "03_Processed"

pos_X = np.load(DATA_DIR / "all_positive_examples.npy")
neg_X = np.load(DATA_DIR / "all_negative_examples.npy")

# Append 1's to positive cases and 0's to negative
pos = np.concatenate((pos_X, np.ones((pos_X.shape[0], 1))), axis=1)
neg = np.concatenate((neg_X, np.zeros((neg_X.shape[0], 1))), axis=1)

X = np.concatenate((pos, neg), axis=0)[:, :-1]
y = np.concatenate((pos, neg), axis=0)[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# AUC function


def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

# Baseline model
model = Sequential()
model.add(Dense(128, input_shape=(256,), activation="relu"))
model.add(Dense(64, input_shape=(256,), activation="relu"))
model.add(Dense(32, input_shape=(256,), activation="relu"))
model.add(Dense(1, activation="sigmoid"))


model.compile(loss="binary_crossentropy", optimizer="adam", metrics=[auc])


model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(X_test, y_test),
)
