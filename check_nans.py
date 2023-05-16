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


path_from_processed='/p/tmp/tobiasoh/machine_learning/complete_datasets/large_set/processed/'
path_dump = '/p/tmp/tobiasoh/machine_learning/complete_datasets/large_set/dump/'
N_files_NaNs = 0
N_files_outliers = 0


for file in os.listdir(path_from_processed):
    move = False
    if file.startswith('data'):
        data = torch.load(path_from_processed+file)
        
        #Check for and remove data with NaNs
        if any(torch.isnan(data.x[:,0])) or any(torch.isnan(data.x[:,1])):
            print(file+' in x')
            shutil.move(path_from_processed+file, path_dump+file)
            N_files_NaNs += 1
            continue
        elif any(torch.isnan(data.edge_attr[:,0])) or any(torch.isnan(data.edge_attr[:,1])) or any(torch.isnan(data.edge_attr[:,2])) or any(torch.isnan(data.edge_attr[:,3])) or any(torch.isnan(data.edge_attr[:,4])) or any(torch.isnan(data.edge_attr[:,5])):
            print(file+ ' in y')
            shutil.move(path_from_processed+file, path_dump+file)
            N_files_NaNs += 1
            continue
        elif any(torch.isnan(data.node_labels)):
            print(file + ' in node labels')
            shutil.move(path_from_processed+file, path_dump+file)
            N_files_NaNs += 1
            continue
            
        #Check for and remove data with outliers    
        if (any(data.x[:,1]>1.12) or any(data.x[:,1]<0.88)):
            for i in range(len(data.x[:,1])):
                if (data.x[i,1]>1.12 or data.x[i,1]<0.88) and data.x[i,0] != 0: 
                    print(file+' in x')
                    print(f'{i}, VM={data.x[i,1]}, PD={data.x[i,0]}')
                    move = True
            if move:
                shutil.move(path_from_processed+file, path_dump+file)
                N_files_outliers += 1
                continue
        if any(torch.sqrt(data.edge_attr[:,1]**2+data.edge_attr[:,2]**2) > data.edge_attr[:,1]*1.01):
            print(file+ ' in edges')
            for i in range(len(data.edge_attr[:,1])):
                if torch.sqrt(data.edge_attr[i,1]**2+data.edge_attr[i,2]**2) > data.edge_attr[i,1]*1.01:
                    print(f'{i}, PF={data.edge_attr[i,1]}')
                    move = True
            if move:
                shutil.move(path_from_processed+file, path_dump+file)
                N_files_outliers += 1
                continue
                      
        elif any(torch.isnan(data.node_labels)):
            print(file + ' in node labels')
            shutil.move(path_from_processed+file, path_dump+file)
            N_files_outliers += 1

print(f'Files removed because of NaNs: {N_files_NaNs}')
print(f'Files removed because of Outliers: {N_files_outliers}')

