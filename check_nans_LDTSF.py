# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 15:55:36 2022

@author: tobia
"""

"needs the ordered labels as input"

import shutil
import numpy as np
import os
import torch
import shutil


path_from_processed='processed/'
path_dump = 'dump/'
N_files_NaNs = 0



for file in os.listdir(path_from_processed):
    if file.startswith('data'):
        data = torch.load(path_from_processed+file)
        if torch.isnan(data.y):
            shutil.move(path_from_processed+file, path_dump+file)
            N_files_NaNs += 1

        
        
print(f'Files removed because of NaNs: {N_files_NaNs}')


