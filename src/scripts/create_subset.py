# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 15:55:36 2022

@author: tobia
"""



import shutil
import numpy as np
import os
import torch
import shutil

SORT_BY_CLASS = False
SORT_BY_GRAPH_LABEL = True
LS_THRESHOLD = 0.1
path_from_processed='processed/'
path_subset = 'subset/'
N_class0 = 4
N_BELOW_THRESHOLD = 850
count = 0


for file in os.listdir(path_from_processed):
    if file.startswith('data'):
        data = torch.load(path_from_processed+file)
        if SORT_BY_CLASS and data.y_class == 0: 
            count += 1
            if count < N_class0:    shutil.copy2(path_from_processed+file, path_subset+file)
            else:                   continue
        elif SORT_BY_GRAPH_LABEL and data.y < LS_THRESHOLD and count < N_BELOW_THRESHOLD:
            count += 1
            shutil.copy2(path_from_processed+file, path_subset+file)
    if count >= N_BELOW_THRESHOLD:  break
        

        



