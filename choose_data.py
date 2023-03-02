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
from get_scenario_statistics import get_data_list


path_from='C:/Users/tobia/OneDrive/Dokumente/Master/Semester4/Masterarbeit/line_regression_nauck_cluster/DC-CFM-GNN/processed/'
path_to='C:/Users/tobia/OneDrive/Dokumente/Master/Semester4/Masterarbeit/line_regression_nauck_cluster/DC-CFM-GNN/test/'     #in path_to the files that wont be used are stored
path_from_processed='C:/Users/tobia/OneDrive/Dokumente/Master/Semester4/Masterarbeit/line_regression_nauck_cluster/DC-CFM-GNN/processed/'
path_to_processed='C:/Users/tobia/OneDrive/Dokumente/Master/Semester4/Masterarbeit/line_regression_nauck_cluster/DC-CFM-GNN/processed_rest/'
count=0
threshold = 0
N_files_removed=0
#data_list=get_data_list(100,path_from)

for file in os.listdir(path_from):
    if file.startswith('data'):
        total_outage = torch.load(path_from+file).node_labels.sum()
        if total_outage == 0:
            if count < threshold:
                count +=1
            else: 
                shutil.move(path_from+file,path_to+file)
                N_files_removed+=1
        
        


"""
for i in range(len(data_list)-1):
    if data_list[i,0]!=data_list[i+1,0]:
        file=f'scenario_{data_list[i+1,0]}_{data_list[i+1,1]}.npz'
        shutil.move(path_from+file,path_to+file)
        N_files_removed+=1        
        
for i in range(len(data_list)-1):
    file=f'scenario_{data_list[i,0]}_{data_list[i,1]}.npz'
    y=np.load(path_from+file)['y']
    node_labels=np.load(path_from+file)['node_labels']
    if abs(y-node_labels.sum())>0.1:
        print(abs(y-node_labels.sum()))
        N_files_removed+=1
        shutil.move(path_from+file,path_to+file)
        shutil.move(path_from_processed+file,path_to_processed+file)
"""        
    
""" 
for file in os.listdir(path_from):
    label=np.load(path_from+file)['y']
    if label<1: 
        shutil.move(path_from+file,path_to+file)
        N_files_removed+=1
"""


print(N_files_removed)
print('files removed')

