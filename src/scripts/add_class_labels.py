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
from torch_geometric.data import Data


path_from_processed='processed/'
path_dump = 'dump/'
N_files_NaNs = 0
N_files_outliers = 0


for file in os.listdir(path_from_processed):
    if file.startswith('data'):
        data = torch.load(path_from_processed+file)
        y = data.y
        if y < 0.18:    y_class = 0
        elif y < 0.65:  y_class = 1
        elif y < 0.88:  y_class = 2
        else:           y_class = 3

        data = Data(x=data.x, y=y, y_class=y_class)
        torch.save(data, path_from_processed+file)

    
