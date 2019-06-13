# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from pathlib import Path
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

batch_size = 2048
epochs = 10

DATA_DIR = Path(__file__).parent.parent / "03_Processed"
TEST_DIR = Path(__file__).parent.parent / "01_Data" / "test"
FACE_REC_MODEL = DATA_DIR.parent / "01_Data" / "dlib_models" / "dlib_face_recognition_resnet_model_v1.dat"
PREDICTOR = DATA_DIR.parent / "01_Data" / "dlib_models" / "shape_predictor_5_face_landmarks.dat"

pos_X = np.load(DATA_DIR / "all_positive_examples.npy")
neg_X = np.load(DATA_DIR / "all_negative_examples.npy")

# Append 1's to positive cases and 0's to negative
pos = np.concatenate((pos_X, np.ones((pos_X.shape[0], 1))), axis=1)
neg = np.concatenate((neg_X, np.zeros((neg_X.shape[0], 1))), axis=1)

X = np.concatenate((pos, neg), axis=0)[:, :-1]
y = np.concatenate((pos, neg), axis=0)[:, -1]


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.05, random_state=42)

X_test = np.load(DATA_DIR / "X_test.npy")

# Normalize
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# AUC function
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


model = Sequential()
model.add(Dense(128, input_shape=(256,), activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(64, input_shape=(256,), activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(32, input_shape=(256,), activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=[auc])

history = model.fit(
    X,
    y,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(X_valid, y_valid),
)

# Plot learning rates
auc = history.history['auc']
val_auc = history.history['val_auc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(auc) + 1)
plt.plot(epochs, auc, 'bo', label='Training auc')
plt.plot(epochs, val_auc, 'b', label='Validation auc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# Score the model with test data
# Load test data
df_test = pd.read_csv(DATA_DIR.parent / "01_Data" / "sample_submission.csv")
df_test.is_related = model.predict(X_test)
df_test.to_csv( str (DATA_DIR.parent / "04_Results" / "10_Norm_Submission.csv"), index = False)