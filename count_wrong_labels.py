# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 10:53:56 2023

@author: tobia
"""

import numpy as np
import os

raw_path='./raw/'
count=0
for file in os.listdir(raw_path):
    y=np.load(raw_path+file)['y']
    node_labels= np.load(raw_path+file)['node_labels']
    if abs(node_labels.sum()-y)>0.01:
        print(node_labels.sum()-y)
        print(file)
        count+=1
print(f'Number of wrong labels: {count}')