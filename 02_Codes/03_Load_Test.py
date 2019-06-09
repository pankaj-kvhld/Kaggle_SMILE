#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 21:36:11 2019

@author: pankaj
"""
import pandas as pd
import numpy as np
import dlib
from pathlib import Path

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
        # print("None or more than one faces in the image")
        return -1


df_test = pd.read_csv(DATA_DIR.parent / "01_Data" / "sample_submission.csv")
X_test = np.empty((df_test.shape[0], 256))
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
        continue
    else:
        x = np.concatenate((img_1_128_vec, img_2_128_vec))
        X_test[ind, :] = x

np.save(DATA_DIR.parent / "03_Processed" / "X_test.npy", X_test)
