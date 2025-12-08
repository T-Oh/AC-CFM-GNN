# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 16:38:12 2023

@author: tobia
"""
import os
import torch
import numpy as np
from torch_geometric.data import Data


def main():
    processed_dir = 'processed/'
    normalized_dir = 'normalized/'
    data_stats_file = 'unnormalized_data_stats_Zhu_nobustype.npy'
    N_NODE_FEATURES = 4    #if NodeIDs are added as features substract 2000 from N_Features
    N_EDGE_FEATURES = 2
    GEN_FEATURE_BIAS = -1   #Used to mark where the generator features start in the node features (used to skip the 4 one hot encoded bus type features) set to -1 if no generator features are used
    N_TARGETS = 2

    #If data stats (min/max/mean/std) file already exists load the file instead of recomputing
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
        node_label_stds = data_stats['node_label_stds']
        
        graph_label_min = data_stats['graph_label_min']
        graph_label_max = data_stats['graph_label_max']
        graph_label_mean = data_stats['graph_label_mean']
        graph_label_std = data_stats['graph_label_std']
    #If file is not found calculate the stats first
    else:
        print('No presaved data stats found - Calculating data stats')
        x_min, x_max, x_means, edge_attr_min, edge_attr_max, edge_attr_means, node_labels_min, node_labels_max, node_labels_means, graph_label_min, graph_label_max, graph_label_mean = get_min_max_features(processed_dir, N_NODE_FEATURES, N_EDGE_FEATURES, N_TARGETS)

        x_stds, edge_stds, node_label_stds, graph_label_std= get_feature_stds(processed_dir, x_means, edge_attr_means, node_labels_means, graph_label_mean, N_NODE_FEATURES, N_EDGE_FEATURES, N_TARGETS)
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
                    'node_label_stds'   : node_label_stds,
                    
                    'graph_label_min'     : graph_label_min,
                    'graph_label_max'     : graph_label_max,
                    'graph_label_mean'    : graph_label_mean,
                    'graph_label_std'     : graph_label_std}

        np.save(processed_dir+'unnormalized_data_stats.npy', data_stats)

    #Go through files to normalize
    for file in os.listdir(processed_dir):
        if file.startswith('data'): #Only process data files
            data = torch.load(processed_dir + '/' + file)
            #Node features
            x = data['x']
            #node power
            for j in range(len(x_max)):
                if any(torch.isnan(x[:,j])):
                    print('NaN Before Normalization x:')
                    print(file)
                    for i in range(len(x[:,1])):
                        if torch.isnan(x[i,j]): print(f'Before, x{j} at bus {i}')


            #x[:,0] = torch.log(x[:,0]+1)/torch.log(x_max[0]+1)
            #x[:,1] = torch.log(x[:,1]+1)/torch.log(x_max[1]+1)
            x[:,0] = (x[:,0]-x_means[0])/x_stds[0]/((x_max[0]-x_means[0])/x_stds[0])
            x[:,1] = (x[:,1]-x_means[1])/x_stds[1]/((x_max[1]-x_means[1])/x_stds[1])
            #node voltage magnitude
            x[:,2] = torch.sign(x[:,2])*torch.log(torch.abs(x[:,2])+1)/torch.log(x_max[2]+1)  #((x[:,1]-x_means[1])/x_stds[1])/((x_max[1]-x_means[1])/x_stds[1])
            #Voltage angle
            #x[:,3] = (x[:,3]-x_means[3])/x_stds[3]/((x_max[3]-x_means[3])/x_stds[3])
            #Shunt susceptance
            x[:,3] = torch.sign(x[:,3])*torch.log(torch.abs(x[:,3])+1)/6  #/torch.log(x_max[3]+1)
            #baseKV
            #x[:,5] = x[:,5]/500 #baseKV max baseKV in ACTIVSg2000 is 500 (min is 13.8)
            #Generator Features
            if GEN_FEATURE_BIAS >= 0:
                for j in range(8):
                    if j == 3 or j == 1:
                        x[:,j+GEN_FEATURE_BIAS] = torch.log(x[:,j+GEN_FEATURE_BIAS]+1-x_min[j+GEN_FEATURE_BIAS])/torch.log(x_max[j+GEN_FEATURE_BIAS]+1-x_min[j+GEN_FEATURE_BIAS]) #10 is the first gen feature in node_features after 6 node features + 4 features for one hot encoded bus type    
                    else:
                        x[:,j+GEN_FEATURE_BIAS] = torch.log(x[:,j+GEN_FEATURE_BIAS]+1)/torch.log(x_max[j+GEN_FEATURE_BIAS]+1) #10 is the first gen feature in node_features after 6 node features + 4 features for one hot encoded bus type

            for i in range(len(x_max)):
                if any(torch.isnan(x[:,i])):
                    print('NaN After Normalization x:')
                    print(file)
                    for j in range(len(x[:,i])):
                        if torch.isnan(x[j,i]): print(f'After, x{i} {j}')


            #Edge Features
            edge_attr = data['edge_attr']
            adj = data['edge_index']

            #when Y is used as edge feature there are the old version (with Y=sqrt(Y.real**2+Y.imag**2)) i.e. 1 feature 
            # and the newer version where the real and imag part are saved seperately i.e. 2 features
            if N_EDGE_FEATURES <= 2 : 

                #edge_attr = torch.log(data['edge_attr']-edge_attr_min+1)/torch.log(edge_attr_max+1)
                edge_attr[:,0] = (data['edge_attr'][:,0]-edge_attr_means[0])/edge_stds[0]/((edge_attr_max[0]-edge_attr_means[0])/edge_stds[0])
                edge_attr[:,1] = (data['edge_attr'][:,1]-edge_attr_means[1])/edge_stds[1]/((edge_attr_max[1]-edge_attr_means[1])/edge_stds[1])
                
            else:   #Multiple edge features
            
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
            node_labels = data['node_labels']
            
            
            if N_TARGETS == 1:
                node_labels = (data['node_labels']-node_labels_means)/node_label_stds
                if any(torch.isnan(node_labels)):
                    print('NaN in node labels after norm:')
                    print(file)
                    for i in range(len(node_labels)):
                        if torch.isnan(node_labels[i]): print(f'After {i}')
            else:
                node_labels[:,0] = torch.sign(data['node_labels'][:,0])*torch.log(torch.abs(data['node_labels'][:,0])+1)/torch.log(node_labels_max[0]+1)
                node_labels[:,1] = torch.sign(data['node_labels'][:,1])*torch.log(torch.abs(data['node_labels'][:,1])+1)/6    #/torch.log(node_labels_max[1]+1.1)
                if any(torch.isnan(node_labels[:,0])) or any(torch.isnan(node_labels[:,1])):
                    print('NaN in node labels after norm:')
                    print(file)
                    for i in range(len(node_labels)):
                        if torch.isnan(node_labels[i,0]): print(f'After {i} label0') 
                        if torch.isnan(node_labels[i,1]): print(f'After {i} label1') 
            #Graph Labels
            if 'y' in data.keys:    
                graph_label = torch.log(data['y']+1)/torch.log(graph_label_max+1)
                data = Data(x=x, edge_index=adj, edge_attr=edge_attr, node_labels=node_labels, y=graph_label) 
            else:
                data = Data(x=x, edge_index=adj, edge_attr=edge_attr, node_labels=node_labels) 

            #Save normalized Data
            torch.save(data, os.path.join(normalized_dir, file))



def get_min_max_features(processed_dir, n_node_features, n_edge_features, n_targets):
    #identifies and saves the min and max values as well as the mean values of all features and labels of the data
    
    #Variables to save the min/max/means
    x_max=torch.zeros(n_node_features)
    x_min=torch.zeros(n_node_features)
    x_means = torch.zeros(n_node_features)
    edge_attr_max=torch.zeros(n_edge_features)
    edge_attr_min=torch.zeros(n_edge_features)
    edge_attr_means = torch.zeros(n_edge_features)
    node_labels_max = torch.zeros(n_targets)
    node_labels_min = torch.zeros(n_targets)
    node_labels_mean = torch.zeros(n_targets)

    #Counts of nodes, edges and instances for mean calculation
    node_count = 0
    edge_count = 0
    graph_count = 0

    #Initialize mins and max values
    for i in range(len(x_max)):
        #Nodefeatures
        x_max[i] = -np.Inf
        x_min[i] = np.Inf
        #EdgeFeatures
        if i <len(edge_attr_max):
            edge_attr_max[i] = -np.Inf
            edge_attr_min[i] = np.Inf
        #NodeLabels
        if i < len(node_labels_min):
            node_labels_max[i] = -np.Inf
            node_labels_min[i] = np.Inf

    #Same for Graph Label
    graph_labels_min = np.Inf
    graph_labels_max = np.NINF
    graph_labels_mean = 0

    
    #Loop through files
    for file in os.listdir(processed_dir):
        if file.startswith('data'): #only process data files
            graph_count = graph_count+1 
            data = torch.load(processed_dir +'/' + file)
            #Nodes
            x = data['x']
            for i in range(x.shape[0]): #node_loop
                for j in range(len(x_max)): #feature_loop
                    if x[i,j]>x_max[j]: x_max[j]=x[i,j]
                    if x[i,j]<x_min[j]: x_min[j]=x[i,j]
                    x_means[j] += x[i,j]
                node_count += 1
            #Edges
            edge_attr = data['edge_attr']
            if edge_attr.dim() == 1: edge_attr = edge_attr.unsqueeze(1)
            for i in range(len(edge_attr)):
                for j in range(len(edge_attr_max)):
                    if edge_attr[i,j]>edge_attr_max[j]: edge_attr_max[j]=edge_attr[i,j]
                    if edge_attr[i,j]<edge_attr_min[j]: edge_attr_min[j]=edge_attr[i,j]
                    edge_attr_means[j] += edge_attr[i,j]
                edge_count += 1
                
            #Node Labels
            node_labels = data['node_labels']
            if node_labels.dim() == 1:  node_labels = node_labels.unsqueeze(1)
            for i in range(len(node_labels)):
                for j in range(len(node_labels_min)):
                    if node_labels[i,j] > node_labels_max[j]: node_labels_max[j] = node_labels[i,j]
                    if node_labels[i,j] < node_labels_min[j]: node_labels_min[j] = node_labels[i,j]
                    node_labels_mean[j] += node_labels[i,j]
            #Graph Labels
            if 'y' in data.keys:   
                graph_label = data['y']
                if graph_label > graph_labels_max: graph_labels_max = graph_label
                if graph_label < graph_labels_min: graph_labels_min = graph_label
                graph_labels_mean += graph_label
            
        

    return x_min, x_max, x_means/node_count, edge_attr_min, edge_attr_max, edge_attr_means/edge_count, node_labels_min, node_labels_max, node_labels_mean/node_count, graph_labels_min, graph_labels_max, graph_labels_mean/graph_count



def get_feature_stds(processed_dir, x_means, edge_means, node_label_means, graph_label_mean, n_node_features, n_edge_features, n_targets):
    x_stds = torch.zeros(n_node_features)
    edge_stds = torch.zeros(n_edge_features)
    node_label_stds = torch.zeros(n_targets)
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
                for j in range(len(x_stds)):
                    x_stds[j] += (x[i,j]-x_means[j])**2

                node_count += 1
            edge_attr = data['edge_attr']
            
            if edge_attr.dim() == 1:    edge_attr = edge_attr.unsqueeze(1)
            for i in range(len(edge_attr)):
                for j in range(len(edge_stds)):
                    edge_stds[j] += (edge_attr[i,j] - edge_means[j])**2
                edge_count += 1

            node_labels = data['node_labels']
            if node_labels.dim() == 1:  node_labels = node_labels.unsqueeze(1)
            for i in range(node_labels.shape[0]):
                for j in range(len(node_label_stds)):
                    node_label_stds[j] += (node_labels[i,j]-node_label_means[j])**2


            #Graph Label
            if 'y' in data.keys:
                graph_label = data['y']
                graph_label_std += (graph_label - graph_label_mean)**2

    return np.sqrt(x_stds/node_count), np.sqrt(edge_stds/edge_count), np.sqrt(node_label_stds/node_count), np.sqrt(graph_label_std/graph_count)
    
if __name__ == "__main__":
    main()

