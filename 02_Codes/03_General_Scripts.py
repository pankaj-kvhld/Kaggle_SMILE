#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The scripts achieve the following:
    1. Identifies number of unique images in the submission file
    
Created on Thu Jun  6 19:17:48 2019

@author: pankaj
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "03_Processed"
TEST_DIR = Path(__file__).parent.parent / "01_Data" / "test"

df_test = pd.read_csv(DATA_DIR.parent / "01_Data" / "sample_submission.csv")

all_pairs = df_test.img_pair.to_list()
all_imgs = [x.split("-")[0] for x in all_pairs] + [x.split("-")[1] for x in all_pairs]
num_unique_imgs = len(set(all_imgs))
print(f"Number of unique images in the submission file : {num_unique_imgs}")