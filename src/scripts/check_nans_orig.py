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
N_files_outliers = 0


for file in os.listdir(path_from_processed):
    move = False
    if file.startswith('data'):
        data = torch.load(path_from_processed+file)
        
        #Check for and remove data with NaNs
        for i in range(19):
            if any(torch.isnan(data.x[:,i])): 
                print(file+' in x{i}')
                shutil.move(path_from_processed+file, path_dump+file)
                N_files_NaNs += 1
                move = True
                break
            elif i<6:
                if any(torch.isnan(data.edge_attr[:,i])):
                    print(file+ ' in edge{i}')
                    shutil.move(path_from_processed+file, path_dump+file)
                    N_files_NaNs += 1
                    move = True
                    break
            elif i == 0 and any(torch.isnan(data.node_labels)):
                move =True
                print(file + ' in node labels')
                shutil.move(path_from_processed+file, path_dump+file)
                N_files_NaNs += 1
                break
            
        #Check for and remove data with outliers  
        if not move and (any(data.edge_attr[:,1]>5000) or any(data.edge_attr[:,2]>4000)):    #largest realistic value is around 4000/3000
            print('Outlier in Edges P Q')
            shutil.move(path_from_processed+file, path_dump+file)
            N_files_outliers += 1
            continue
        if not move and any(data.x[:,11]>1500):     #largest realistic value is around 1000
            print('Outlier in Gen Q')
            shutil.move(path_from_processed+file, path_dump+file)
            N_files_outliers += 1
            continue
        if not move and any(data.x[:,2]>2.5):     #largest realistic value is around 2
            print('Outlier Vm')
            shutil.move(path_from_processed+file, path_dump+file)
            N_files_outliers += 1
        
print(f'Files removed because of NaNs: {N_files_NaNs}')
print(f'Files removed because of Outliers: {N_files_outliers}')

