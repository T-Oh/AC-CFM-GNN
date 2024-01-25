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
    x_max=torch.zeros(4)
    x_min=torch.zeros(4)
    x_means = torch.zeros(4)
    edge_attr_max=torch.zeros(5)
    edge_attr_min=torch.zeros(5)
    edge_attr_means = torch.zeros(5)
    node_count = 0
    edge_count = 0
    

    node_count = 0
    edge_count = 0
    for i in range(5):
        edge_attr_max[i] =  np.NINF
        edge_attr_min[i] = np.Inf
        if i <3:
            x_max[i] = np.NINF
            x_min[i] = np.Inf
    node_labels_max=0
    node_labels_min=1e6
    node_labels_mean = 0
    graph_labels_min = np.Inf
    graph_labels_max = np.NINF
    graph_labels_mean = 0
    graph_count = 0
    
    
    for file in os.listdir(processed_dir):
        if file.startswith('data'):
            graph_count = graph_count+1
            data = torch.load(processed_dir +'/' + file)
            x = data['x']
            for i in range(x.shape[0]):
                if x[i,0]>x_max[0]: x_max[0]=x[i,0]
                if x[i,0]<x_min[0]: x_min[0]=x[i,0]
                if x[i,1]>x_max[1]: x_max[1]=x[i,1]
                if x[i,1]<x_min[1]: x_min[1]=x[i,1]
                if x[i,2]>x_max[2]: x_max[2]=x[i,2]
                if x[i,2]<x_min[2]: x_min[2]=x[i,2]
                if x[i,3]>x_max[3]: x_max[3]=x[i,3]
                if x[i,3]<x_min[3]: x_min[3]=x[i,3]
                x_means[0] += x[i,0]
                x_means[1] += x[i,1]
                x_means[2] += x[i,2]
                x_means[3] += x[i,3]
                node_count += 1
                
                
                #if x[i,2]>x_max[2]: x_max[2]=x[i,2]    can be used for a third node feature
                #if x[i,2]<x_min[2]: x_min[2]=x[i,2]
            
            edge_attr = data['edge_attr']
            for i in range(len(edge_attr)):
                if edge_attr[i,0]>edge_attr_max[0]: edge_attr_max[0]=edge_attr[i,0]
                if edge_attr[i,0]<edge_attr_min[0]: edge_attr_min[0]=edge_attr[i,0]
                if edge_attr[i,1]>edge_attr_max[1]: edge_attr_max[1]=edge_attr[i,1]
                if edge_attr[i,1]<edge_attr_min[1]: edge_attr_min[1]=edge_attr[i,1]
                if edge_attr[i,2]>edge_attr_max[2]: edge_attr_max[2]=edge_attr[i,2]
                if edge_attr[i,2]<edge_attr_min[2]: edge_attr_min[2]=edge_attr[i,2]
                if edge_attr[i,4]>edge_attr_max[3]: edge_attr_max[3]=edge_attr[i,4]
                if edge_attr[i,4]<edge_attr_min[3]: edge_attr_min[3]=edge_attr[i,4]
                if edge_attr[i,5]>edge_attr_max[4]: edge_attr_max[4]=edge_attr[i,5]
                if edge_attr[i,5]<edge_attr_min[4]: edge_attr_min[4]=edge_attr[i,5]
                edge_attr_means[0] += edge_attr[i,0]
                edge_attr_means[1] += edge_attr[i,1]
                edge_attr_means[2] += edge_attr[i,2]
                edge_attr_means[3] += edge_attr[i,4]
                edge_attr_means[4] += edge_attr[i,5]
                edge_count += 1
                
    
            node_labels = data['node_labels']
            for i in range(len(node_labels)):
                if node_labels[i] > node_labels_max: node_labels_max = node_labels[i]
                if node_labels[i] < node_labels_min: node_labels_min = node_labels[i]
                node_labels_mean += node_labels[i]
                
            graph_label = data['y']
            if graph_label > graph_labels_max: graph_labels_max = graph_label
            if graph_label < graph_labels_min: graph_labels_min = graph_label
            graph_labels_mean += graph_label
            
        

    return x_min, x_max, x_means/node_count, edge_attr_min, edge_attr_max, edge_attr_means/edge_count, node_labels_min, node_labels_max, node_labels_mean/node_count, graph_labels_min, graph_labels_max, graph_labels_mean/graph_count


def get_feature_stds(processed_dir, x_means, edge_means, graph_label_mean):
    x_stds = torch.zeros(4)
    edge_stds = torch.zeros(5)
    graph_label_std =0
    node_count = 0
    edge_count = 0
    graph_count = 0
    for file in os.listdir(processed_dir):
        if file.startswith('data'):
            graph_count += 1
            data = torch.load(processed_dir +'/' + file)
            x = data['x']
            for i in range(x.shape[0]):
                x_stds[0] += (x[i,0]-x_means[0])**2
                x_stds[1] += (x[i,1]-x_means[1])**2
                x_stds[2] += (x[i,2]-x_means[2])**2
                x_stds[3] += (x[i,3]-x_means[3])**2
                node_count += 1
            edge_attr = data['edge_attr']
            for i in range(len(edge_attr)):
                edge_stds[0] += (edge_attr[i,0] - edge_means[0])**2
                edge_stds[1] += (edge_attr[i,1] - edge_means[1])**2
                edge_stds[2] += (edge_attr[i,2] - edge_means[2])**2
                edge_stds[3] += (edge_attr[i,4] - edge_means[3])**2
                edge_stds[4] += (edge_attr[i,5] - edge_means[4])**2
                edge_count += 1
            graph_label = data['y']
            graph_label_std += (graph_label - graph_label_mean)**2
        return np.sqrt(x_stds/node_count), np.sqrt(edge_stds/edge_count), np.sqrt(graph_label_std/graph_count)
    
    
processed_dir = 'processed/'
normalized_dir = 'processed/'
data_stats_file = 'unnormalized_data_stats.npy'

if os.path.isfile(processed_dir + data_stats_file):
    print(f'Using presaved data stats of file: {data_stats_file} for normalization')
    data_stats = np.load(processed_dir + data_stats_file, allow_pickle=True).item()
    x_min   = data_stats['x_min']
    x_max   = data_stats['x_max']
    x_means = data_stats['x_means']
    x_stds  = data_stats['x_stds']
    
    edge_attr_min   = data_stats['edge_attr_min']
    edge_attr_max   = data_stats['edge_attr_max']
    edge_attr_means = data_stats['edge_attr_means']
    edge_stds       = data_stats['edge_attr_stds']
    
    node_labels_min = data_stats['node_labels_min']
    node_labels_max = data_stats['node_labels_max']
    node_labels_means = data_stats['node_labels_means']
    
    graph_label_min = data_stats['graph_label_min']
    graph_label_max = data_stats['graph_label_max']
    graph_label_mean = data_stats['graph_label_mean']
    graph_label_std = data_stats['graph_label_std']
else:
    print('No presaved data stats found - Calculating data stats')
    x_min, x_max, x_means, edge_attr_min, edge_attr_max, edge_attr_means, node_labels_min, node_labels_max, node_labels_means, graph_label_min, graph_label_max, graph_label_mean = get_min_max_features(processed_dir)

    x_stds, edge_stds, graph_label_std= get_feature_stds(processed_dir, x_means, edge_attr_means, graph_label_mean)
    data_stats = {'x_min'   : x_min,
                  'x_max'   : x_max,
                  'x_means' : x_means,
                  'x_stds'  : x_stds,
                  
                  'edge_attr_min'   : edge_attr_min,
                  'edge_attr_max'   : edge_attr_max,
                  'edge_attr_means' : edge_attr_means,
                  'edge_attr_stds'  : edge_stds,
                  
                  'node_labels_min' : node_labels_min,
                  'node_labels_max' : node_labels_max,
                  'node_labels_means'   : node_labels_means,
                  
                  'graph_label_min'     : graph_label_min,
                  'graph_label_max'     : graph_label_max,
                  'graph_label_mean'    : graph_label_mean,
                  'graph_label_std'     : graph_label_std}

    np.save(processed_dir+'unnormalized_data_stats.npy', data_stats)
for file in os.listdir(processed_dir):
    if file.startswith('data'):
        data = torch.load(processed_dir + '/' + file)
        #Node features
        x = data['x']
        #node power
        if any(torch.isnan(x[:,0])) or any(torch.isnan(x[:,1])) or any(torch.isnan(x[:,2])):
            print('NaN Before Normalization x:')
            print(file)
            for i in range(len(x[:,1])):
                if torch.isnan(x[i,0]): print(f'Before, x0 {i}')
                if torch.isnan(x[i,1]): print(f'Before, x1 {i}')
                if torch.isnan(x[i,2]): print(f'Before, x2 {i}')
                if torch.isnan(x[i,3]): print(f'Before, x3 {i}')
        x[:,0] = torch.log(x[:,0]+1)/torch.log(x_max[0]+1)
        x[:,1] = torch.log(x[:,1]+1)/torch.log(x_max[1]+1)
        #node voltage magnitude
        x[:,2] = torch.log(x[:,2]+1)/torch.log(x_max[2]+1)  #((x[:,1]-x_means[1])/x_stds[1])/((x_max[1]-x_means[1])/x_stds[1])
        #Voltage angle
        x[:,3] = (x[:,3]-x_means[3])/x_stds[3]/((x_max[3]-x_means[3])/x_stds[3])
        if any(torch.isnan(x[:,0])) or any(torch.isnan(x[:,1])) or any(torch.isnan(x[:,2])):
            print('NaN After Normalization x:')
            print(file)
            for i in range(len(x[:,1])):
                if torch.isnan(x[i,0]): print(f'After, x0 {i}')
                if torch.isnan(x[i,1]): print(f'After, x1 {i}')
                if torch.isnan(x[i,2]): print(f'After, x2 {i}')
                if torch.isnan(x[i,3]): print(f'After, x3 {i}')
        #Edge Features
        edge_attr = data['edge_attr']
        adj = data['edge_index']
        #capacity
        edge_attr[:,0] = torch.log(data['edge_attr'][:,0]+1)/torch.log(edge_attr_max[0]+1)
        #Pf, QF and resistance
        if any(torch.isnan(edge_attr[:,0])) or any(torch.isnan(edge_attr[:,1])) or any(torch.isnan(edge_attr[:,2])) or any(torch.isnan(edge_attr[:,3])) or any(torch.isnan(edge_attr[:,4])) or any(torch.isnan(edge_attr[:,5])):
            print('NaN in edges Before Normalization:')
            print(file)
            for i in range(len(edge_attr[:,1])):
                if torch.isnan(edge_attr[i,0]) and edge_attr[i,3]==1: print(f'Before, edge0 {i}')
                if torch.isnan(edge_attr[i,1]) and edge_attr[i,3]==1: print(f'Before, edge1 {i}')
                if torch.isnan(edge_attr[i,3]): print(f'Before, edge3 {i}')
                if torch.isnan(edge_attr[i,4]) and edge_attr[i,3]==1: print(f'Before, edge4 {i}')
                if torch.isnan(edge_attr[i,5]) and edge_attr[i,3]==1: print(f'Before, edge5 {i}')
                if torch.isnan(edge_attr[i,6]): print(f'Before, edge6 {i}')
                if torch.isnan(edge_attr[i,2]) and edge_attr[i,3]==1: print(f'Before, edge2 {i}')
        edge_attr[:,1] = (data['edge_attr'][:,1]-edge_attr_means[1])/edge_stds[1]/((edge_attr_max[1]-edge_attr_means[1])/edge_stds[1])
        edge_attr[:,2] = (data['edge_attr'][:,2]-edge_attr_means[2])/edge_stds[2]/((edge_attr_max[2]-edge_attr_means[2])/edge_stds[2])
        edge_attr[:,4] = torch.log(data['edge_attr'][:,4]+1)/torch.log(edge_attr_max[3]+1)# -edge_attr_means[3])/edge_stds[3]/((edge_attr_max[3]-edge_attr_means[3])/edge_stds[3])
        #reactance
        edge_attr[:,5] = torch.log(data['edge_attr'][:,5]+1)/torch.log(edge_attr_max[4]+1)
        if any(torch.isnan(edge_attr[:,0])) or any(torch.isnan(edge_attr[:,1])) or any(torch.isnan(edge_attr[:,2])) or any(torch.isnan(edge_attr[:,3])) or any(torch.isnan(edge_attr[:,4])) or any(torch.isnan(edge_attr[:,5])):
            print('NaN in edges after Normalization:')
            print(file)
            for i in range(len(edge_attr[:,1])):
                if torch.isnan(edge_attr[i,0]) and edge_attr[i,3]==1: print(f'after, edge0 {i}')
                if torch.isnan(edge_attr[i,1]) and edge_attr[i,3]==1: print(f'after, edge1 {i}')
                if torch.isnan(edge_attr[i,3]): print(f'after, edge3 {i}')
                if torch.isnan(edge_attr[i,4]) and edge_attr[i,3]==1: print(f'after, edge4 {i}')
                if torch.isnan(edge_attr[i,5]) and edge_attr[i,3]==1: print(f'after, edge5 {i}')
                if torch.isnan(edge_attr[i,6]): print(f'after, edge6 {i}')
                if torch.isnan(edge_attr[i,2]) and edge_attr[i,3]==1: print(f'after, edge2 {i}')
        
        #Node Labels
        if any(torch.isnan(data['node_labels'])):
            print('NaN in node labels before norm:')
            print(file)
            for i in range(len(data['node_labels'])):
                if torch.isnan(data['node_labels'][i]): print(f'Before, node_labels {i}')
        node_labels = torch.log(data['node_labels']+1)/torch.log(node_labels_max+1)
        if any(torch.isnan(node_labels)):
            print('NaN in node labels after norm:')
            print(file)
            for i in range(len(node_labels)):
                if torch.isnan(node_labels[i]): print(f'Before, after {i}')
                
        graph_label = torch.log(data['y']+1)/torch.log(graph_label_max+1)
        data = Data(x=x, edge_index=adj, edge_attr=edge_attr, node_labels=node_labels, y=graph_label) 
        torch.save(data, os.path.join(normalized_dir, file))
