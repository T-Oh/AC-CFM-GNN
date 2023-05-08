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


path_from_processed='processed/'#'/p/tmp/tobiasoh/machine_learning/Ike_ajusted_nonans/processed/'
path_dump = 'dump/'#'/p/tmp/tobiasoh/machine_learning/Ike_ajusted_nonans/dump/'
N_files_removed=0



for file in os.listdir(path_from_processed):
    move=False
    if file.startswith('data'):
        data = torch.load(path_from_processed+file)
        if any(data.x[:,1]>500):
            print(file+' in x')
            for i in range(len(data.x[:,1])):
                if data.x[i,1]>500: print(f'{i}, VM={data.x[i,1]}')
            shutil.move(path_from_processed+file, path_dump+file)
            move=True
        if any(data.edge_attr[:,1]>685484.75) or any(data.edge_attr[:,1]<-563764.75) or any(data.edge_attr[:,2]>231259.625) or any(data.edge_attr[:,2]<-152719.78125):
            print(file+ ' in edges')
            for i in range(len(data.edge_attr[:,1])):
                if data.edge_attr[i,1]>685484.75 or data.edge_attr[i,1]<-563764.75:
                    print(f'{i}, PF={data.edge_attr[i,1]}')
                if data.edge_attr[i,2]>231259.625 or data.edge_attr[i,2]<-152719:
                    print(f'{i}, QF={data.edge_attr[i,2]}')      
            move=True
        elif any(torch.isnan(data.node_labels)):
            print(file + ' in node labels')
            shutil.move(path_from_processed+file, path_dump+file)
            move=True
        if move: shutil.move(path_from_processed+file, path_dump+file)
        

print(N_files_removed)
print('files removed')

