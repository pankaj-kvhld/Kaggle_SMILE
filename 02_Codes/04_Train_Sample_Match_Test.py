#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 21:38:18 2019

@author: pankaj
"""

import numpy as np
from pathlib import Path
import pandas as pd
import time

start = time.time()

DATA_DIR = Path(__file__).parent.parent / "03_Processed"

pos_X = np.load(DATA_DIR / "all_positive_examples.npy")
neg_X = np.load(DATA_DIR / "all_negative_examples.npy")

# Append 1's to positive cases and 0's to negative
pos = np.concatenate((pos_X, np.ones((pos_X.shape[0], 1))), axis=1)
neg = np.concatenate((neg_X, np.zeros((neg_X.shape[0], 1))), axis=1)

X = np.concatenate((pos, neg), axis=0)[:, :-1]
y = np.concatenate((pos, neg), axis=0)[:, -1]

# Removing very small values
X_filtered = X[np.abs(X[:, 0]) > 1e-4, :]
y = y[np.abs(X[:, 0]) > 1e-4]

X_test = np.load(DATA_DIR / "X_test.npy")
X_test_sensible = X_test[abs(X_test[:, 0]) > 1e-4]

# Find distance of each train set sample from test set 
dis = np.empty((0, X_test_sensible.shape[0]))
for i in range(X.shape[0]):
        d = np.linalg.norm(X_test_sensible - X[i, :], axis=1)
        dis = np.append(dis, d.reshape((1, -1)), axis=0)
        
        if i%1000 == 0:
            print(i)
            
np.save(DATA_DIR / "distance.npy")

end = time.time()
print(f'Take taken : {end - start}')        