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
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import dlib

batch_size = 128
epochs = 12

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


## Preprocessing images
# Function to extract 128 vector for a given image
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(str(PREDICTOR))
facerec = dlib.face_recognition_model_v1(str(FACE_REC_MODEL))

def img_2_128vec(img):
    """ Takes an image and retuns the 128 dimension vector
    """
    dets = detector(img, 1)
    if len(dets) == 1:
        shape = sp(img, dets[0])
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        return face_descriptor
    else:
        #print("None or more than one faces in the image")
        return -1

# Score the model with test data
# Load test data
df_test = pd.read_csv(DATA_DIR.parent / "01_Data" / "sample_submission.csv")
preds = []
for ind, row in df_test.iterrows():
    if ind % 100 == 0:
        print(ind)
    
    img_1 = row.img_pair.split("-")[0]
    img_2 = row.img_pair.split("-")[1]
    
    img_1 = dlib.load_rgb_image(str(TEST_DIR / img_1))
    img_1_128_vec = img_2_128vec(img_1)
    
    img_2 = dlib.load_rgb_image(str(TEST_DIR / img_2))
    img_2_128_vec = img_2_128vec(img_2)
    
    if (img_1_128_vec == -1) or (img_2_128_vec == -1):
        pred = 0
    else:
        x = np.concatenate((img_1_128_vec, img_2_128_vec))    
        y = model.predict(x.reshape(-1, 256))[0][0]
    
    preds.append(y)

df_test.is_related = preds
df_test.to_csv( str (DATA_DIR.parent / "04_Results" / "01_Submission.csv"), index = False)