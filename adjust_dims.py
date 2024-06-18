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
N_files_dim_mismatch = 0


for file in os.listdir(path_from_processed):
    move = False
    if file.startswith('data'):
        data = torch.load(path_from_processed+file)        
        if len(data.edge_attr) != data.edge_index.shape[1]:
            #shutil.move(path_from_processed+file, path_dump+file)
            #print(file)
            #N_files_dim_mismatch += 1
            if len(data.edge_attr)+2000 == data.edge_index.shape[1]:
                if all(data.edge_index[0,-2000:-1] == data.edge_index[0,-4000:-2001]) and all(data.edge_index[1,-2000:-1] == data.edge_index[1,-4000:-2001]):
                    data.edge_index = data.edge_index[:,:-2000]
                    #ajusted_data = Data(x=data.x, edge_index=data.edge_index[:,:-2000], edge_attr=data.edge_attr, node_labels=data.node_labels, y=data.y) 
                    torch.save(data,path_from_processed+file)
            else:
                print(file)
                print('Different in Array length:',len(data.edge_attr)-data.edge_index.shape[1])
                N_files_dim_mismatch += 1
                shutil.move(path_from_processed+file, path_dump+file)

 




'''print('Removed')
print(N_files_dim_mismatch)
print('files because of dimensionality mismatch in adjacency and edge_weight')'''

        


