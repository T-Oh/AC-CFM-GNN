# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 16:38:12 2023

@author: tobia
"""
import os
import torch
import numpy as np
from torch_geometric.data import Data

def get_min_max_features(processed_dir):
    
    
    graph_labels_min = np.Inf
    graph_labels_max = np.NINF
    graph_labels_mean = 0
    graph_count = 0
    
    
    for file in os.listdir(processed_dir):
        if file.startswith('data'):
            graph_count = graph_count+1
            data = torch.load(processed_dir +'/' + file)
           
            graph_label = data['y']
            if graph_label > graph_labels_max: graph_labels_max = graph_label
            if graph_label < graph_labels_min: graph_labels_min = graph_label
            graph_labels_mean += graph_label
            
        

    return graph_labels_min, graph_labels_max, graph_labels_mean/graph_count



processed_dir = 'processed/'
normalized_dir = 'processed/'
data_stats_file = 'unnormalized_data_stats.npy'

if os.path.isfile(processed_dir + data_stats_file):
    print(f'Using presaved data stats of file: {data_stats_file} for normalization')
    data_stats = np.load(processed_dir + data_stats_file, allow_pickle=True).item()
   
    graph_label_min = data_stats['graph_label_min']
    graph_label_max = data_stats['graph_label_max']
    graph_label_mean = data_stats['graph_label_mean']
else:
    print('No presaved data stats found - Calculating data stats')
    graph_label_min, graph_label_max, graph_label_mean = get_min_max_features(processed_dir)

    
    data_stats = {
                'graph_label_min'     : graph_label_min,
                'graph_label_max'     : graph_label_max,
                'graph_label_mean'    : graph_label_mean}

    np.save(processed_dir+'unnormalized_data_stats.npy', data_stats)
for file in os.listdir(processed_dir):
    if file.startswith('data'):
        data = torch.load(processed_dir + '/' + file)
        #Node features
        
        graph_label = torch.log(data['y']+1)/torch.log(graph_label_max+1)
        data = Data(x=data.x, y=graph_label) 
        torch.save(data, os.path.join(normalized_dir, file))
