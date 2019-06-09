#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 20:57:35 2019

@author: pankaj
"""

import dlib
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tpot import TPOTClassifier


DATA_DIR = Path(__file__).parent.parent / "03_Processed"
TEST_DIR = Path(__file__).parent.parent / "01_Data" / "test"
FACE_REC_MODEL = (
    DATA_DIR.parent
    / "01_Data"
    / "dlib_models"
    / "dlib_face_recognition_resnet_model_v1.dat"
)
PREDICTOR = (
    DATA_DIR.parent / "01_Data" / "dlib_models" / "shape_predictor_5_face_landmarks.dat"
)

# Load test data
df_test = pd.read_csv(DATA_DIR.parent / "01_Data" / "sample_submission.csv")


def load_train_data():
    """
    Loads the positive and negative preprocessed samples
    """

    pos_X = np.load(DATA_DIR / "all_positive_examples.npy")
    neg_X = np.load(DATA_DIR / "all_negative_examples.npy")

    # Append 1's to positive cases and 0's to negative
    pos = np.concatenate((pos_X, np.ones((pos_X.shape[0], 1))), axis=1)
    neg = np.concatenate((neg_X, np.zeros((neg_X.shape[0], 1))), axis=1)

    X = np.concatenate((pos, neg), axis=0)[:, :-1]
    y = np.concatenate((pos, neg), axis=0)[:, -1]

    return X, y

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
        # print("None or more than one faces in the image")
        return -1


X, y = load_train_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

print('starting tpot...')
tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, n_jobs=-1)
tpot.fit(X_train, y_train)

# Score the submission test set
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
        y = tpot.predict_proba(x.reshape(-1, 256))[0][0]

    preds.append(y)

df_test.is_related = preds
df_test.to_csv(
    str(DATA_DIR.parent / "04_Results" / "07_Tpot_Submission.csv"), index=False
)
