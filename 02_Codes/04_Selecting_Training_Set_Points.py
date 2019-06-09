#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Selecting only those images which have similarity with the test set images. 

Created on Thu Jun  6 20:00:45 2019

@author: pankaj
"""

import dlib
import numpy as np
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "03_Processed"
TEST_DIR = Path(__file__).parent.parent / "01_Data" / "test"
FACE_REC_MODEL = DATA_DIR.parent / "01_Data" / "dlib_models" / "dlib_face_recognition_resnet_model_v1.dat"
PREDICTOR = DATA_DIR.parent / "01_Data" / "dlib_models" / "shape_predictor_5_face_landmarks.dat"

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


X, y = load_train_data()

img_2 = df_test.iloc[0, :].img_pair.split('-')[1]
img_2 = dlib.load_rgb_image(str(TEST_DIR / img_2))
img_2_128_vec = img_2_128vec(img_2)
print(img_2_128_vec)