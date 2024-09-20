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

# Constants for number of features
N_NODE_FEATURES = 19       # Number of node features (x)
N_EDGE_FEATURES = 6        # Number of edge features (edge_attr)
N_NODE_LABELS = 1          # Number of node labels (node_labels)

# Paths for data
path_from_processed = 'processed/'
path_dump = 'dump/'
N_files_NaNs = 0
N_files_outliers = 0

for file in os.listdir(path_from_processed):
    move = False
    if file.startswith('data'):
        data = torch.load(path_from_processed + file)

        # Check for and remove data with NaNs in node features and edge features
        for i in range(N_NODE_FEATURES):
            # Check NaNs in node features (data.x)
            if any(torch.isnan(data.x[:, i])):
                print(f'{file} has NaNs in node feature {i}')
                shutil.move(path_from_processed + file, path_dump + file)
                N_files_NaNs += 1
                move = True
                break
            # Check NaNs in edge features (data.edge_attr)
            elif i < N_EDGE_FEATURES:
                if any(torch.isnan(data.edge_attr[:, i])):
                    print(f'{file} has NaNs in edge feature {i}')
                    shutil.move(path_from_processed + file, path_dump + file)
                    N_files_NaNs += 1
                    move = True
                    break
            # Check NaNs in node labels (data.node_labels)
            elif i == 0 and any(torch.isnan(data.node_labels)):
                print(f'{file} has NaNs in node labels')
                shutil.move(path_from_processed + file, path_dump + file)
                N_files_NaNs += 1
                move = True
                break

        # Check for and remove data with outliers  
        if not move and (any(data.edge_attr[:, 1] > 5000) or any(data.edge_attr[:, 2] > 4000)):
            print(f'{file} has outliers in edge features P or Q')
            shutil.move(path_from_processed + file, path_dump + file)
            N_files_outliers += 1
            continue
        if not move and any(data.x[:, 11] > 1500):  # Check for outliers in Gen Q
            print(f'{file} has outliers in Gen Q')
            shutil.move(path_from_processed + file, path_dump + file)
            N_files_outliers += 1
            continue
        if not move and any(data.x[:, 2] > 2.5):  # Check for outliers in Vm
            print(f'{file} has outliers in Vm')
            shutil.move(path_from_processed + file, path_dump + file)
            N_files_outliers += 1

print(f'Files removed because of NaNs: {N_files_NaNs}')
print(f'Files removed because of Outliers: {N_files_outliers}')
