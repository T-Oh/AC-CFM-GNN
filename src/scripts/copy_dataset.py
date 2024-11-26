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
path_dump = 'test/'



for file in os.listdir(path_from_processed):
    if file.startswith('data'):
        data = torch.load(path_from_processed+file)
        

        shutil.copy2(path_from_processed+file, path_dump+file)



