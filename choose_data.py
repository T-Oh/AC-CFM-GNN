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


        

            

path_from_processed='/p/tmp/tobiasoh/machine_learning/complete_datasets/Claudette_ajusted/processed/'
path_to_processed='/p/tmp/tobiasoh/machine_learning/complete_datasets/large_set/processed/'
N_files_removed=0
zero_count = 0
lowest_bar_count = 0
zero_threshold = 15
lowest_bar_threshold = 500 #23200 
total_outage_belowzero = 0
#data_list=get_data_list(100,path_from)
files_copied = 0
N_files_NaNs = 0
N_files_outliers = 0

for file in os.listdir(path_from_processed):
    if file.startswith('data'):
        total_outage = torch.load(path_from_processed+file).node_labels.sum()
        data = torch.load(path_from_processed+file)
        move = False

        if total_outage < 6975:
            total_outage_belowzero += 1
            if lowest_bar_count < lowest_bar_threshold:
                move = True
                lowest_bar_count += 1
            else:
                N_files_removed += 1 
        else:
            move = True

        if move:
            #Check for and remove data with NaNs
            if any(torch.isnan(data.x[:,0])) or any(torch.isnan(data.x[:,1])):
                print(file+' in x')
                N_files_NaNs += 1
                continue
            elif any(torch.isnan(data.edge_attr[:,0])) or any(torch.isnan(data.edge_attr[:,1])) or any(torch.isnan(data.edge_attr[:,2])) or any(torch.isnan(data.edge_attr[:,3])) or any(torch.isnan(data.edge_attr[:,4])) or any(torch.isnan(data.edge_attr[:,5])):
                print(file+ ' in y')
                N_files_NaNs += 1
                continue
            elif any(torch.isnan(data.node_labels)):
                print(file + ' in node labels')
                N_files_NaNs += 1
                continue
                
            #Check for and remove data with outliers    
            elif any(data.x[:,1]>1.2):
                print(file+' in x')
                for i in range(len(data.x[:,1])):
                    if data.x[i,1]>1.2: print(f'{i}, VM={data.x[i,1]}')
                N_files_outliers += 1
                continue
            elif any(data.edge_attr[:,1]>6300) or any(data.edge_attr[:,1]<-6300) or any(data.edge_attr[:,2]>6300) or any(data.edge_attr[:,2]<-6300):
                print(file+ ' in edges')
                for i in range(len(data.edge_attr[:,1])):
                    if data.edge_attr[i,1]>6300 or data.edge_attr[i,1]<-6300:
                        print(f'{i}, PF={data.edge_attr[i,1]}')
                        N_files_outliers += 1
                        continue
                    if data.edge_attr[i,2]>6300 or data.edge_attr[i,2]<-6300:
                        print(f'{i}, QF={data.edge_attr[i,2]}')  
                        N_files_outliers += 1
                        continue
                          
            elif any(torch.isnan(data.node_labels)):
                print(file + ' in node labels')
                N_files_outliers += 1
                continue
            else:
                files_copied += 1
                shutil.copyfile(path_from_processed+file, path_to_processed+file[0:5]+'1'+file[5:])




print(f'Files removed because of NaNs: {N_files_NaNs}')
print(f'Files removed because of Outliers: {N_files_outliers}')
print(N_files_removed)
print('files removed')
print(f'Files Copied: {files_copied}')
print(total_outage_belowzero)
print('Total outages smaller than 0')
"""
for i in range(len(data_list)-1):
    if data_list[i,0]!=data_list[i+1,0]:
        file=f'scenario_{data_list[i+1,0]}_{data_list[i+1,1]}.npz'
        shutil.move(path_from+file,path_to+file)
        N_files_removed+=1        
"""
"""     
for i in range(len(data_list)-1):
    file=f'scenario_{int(data_list[i,0])}_{int(data_list[i,1])}.npz'
    y=np.load(path_from+file)['y']
    node_labels=np.load(path_from+file)['node_labels']
    if abs(y-node_labels.sum())>0.1:
        print(abs(y-node_labels.sum()))
        torchfile=f'data_{int(data_list[i,0])}_{int(data_list[i,1])}.pt'
        N_files_removed+=1
        shutil.move(path_from+file,path_to+file)
        shutil.move(path_from_processed+torchfile,path_to_processed+torchfile)
"""        
    
"""
for file in os.listdir(path_from):
    label=np.load(path_from+file)['y']
    if label<1: 
        shutil.move(path_from+file,path_to+file)
        N_files_removed+=1
"""



