# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 15:55:36 2022

@author: tobia
"""

"needs the ordered labels as input"

import shutil
import numpy as np
import os
from get_scenario_statistics import get_data_list
import torch
import shutil


path_from_processed='/p/tmp/tobiasoh/machine_learning/Ike_ajusted_nonans/processed/'
path_dump = '/p/tmp/tobiasoh/machine_learning/Ike_ajusted_nonans/dump/'
N_files_removed=0


for file in os.listdir(path_from_processed):
    if file.startswith('data'):
        data = torch.load(path_from_processed+file)
        if any(torch.isnan(data.x[:,0])) or any(torch.isnan(data.x[:,1])):
            print(file+' in x')
            shutil.move(path_from_processed+file, path_dump+file)
            N_files_removed += 1
        elif any(torch.isnan(data.edge_attr[:,0])) or any(torch.isnan(data.edge_attr[:,1])) or any(torch.isnan(data.edge_attr[:,2])) or any(torch.isnan(data.edge_attr[:,3])) or any(torch.isnan(data.edge_attr[:,4])) or any(torch.isnan(data.edge_attr[:,5])):
            print(file+ ' in y')
            shutil.move(path_from_processed+file, path_dump+file)
            N_files_removed += 1
        elif any(torch.isnan(data.node_labels)):
            print(file + ' in node labels')
            shutil.move(path_from_processed+file, path_dump+file)
            N_files_removed += 1

print(N_files_removed)
print('files removed')

