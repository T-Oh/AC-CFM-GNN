import os
import torch
import numpy as np

from torch_geometric.data import Data

from utils.normalization import get_min_max_features_from_dataset, get_feature_stds_from_dataset

def normalize_DP(cfg, trainset, testset):
    #If data stats (min/max/mean/std) file already exists load the file instead of recomputing
    #get N_Node_Features, N_Edge_Features and N_Targets from one instance of trainset
    if isinstance(trainset[0], list):
        data_sample = trainset[0][0]  # Assuming the first element of the sequence is representative
    else:
        data_sample = trainset[0]
    N_Node_Features = data_sample.x.shape[1]
    N_Edge_Features = data_sample.edge_attr.shape[1] if data_sample.edge_attr.size(0) > 0 else 0
    N_Targets = data_sample.node_labels.shape[1] if data_sample.node_labels.dim() > 1 else 1

    if os.path.isfile(cfg.processed_dir + cfg.data_stats_filename):
        print(f'Using presaved data stats of file: {cfg.data_stats_filename} for normalization')
        data_stats = np.load(cfg.processed_dir + cfg.data_stats_filename, allow_pickle=True).item()
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
        x_min, x_max, x_means, edge_attr_min, edge_attr_max, edge_attr_means, node_labels_min, node_labels_max, node_labels_means, graph_label_min, graph_label_max, graph_label_mean = get_min_max_features_from_dataset(trainset, N_Node_Features, N_Edge_Features, N_Targets)

        x_stds, edge_stds, node_label_stds, graph_label_std = get_feature_stds_from_dataset(trainset, x_means, edge_attr_means, node_labels_means, graph_label_mean, N_Node_Features, N_Edge_Features, N_Targets)
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
        print(f'Saving data stats to file: {cfg.processed_dir + cfg.data_stats_filename}')
        np.save(os.path.join(cfg.processed_dir, cfg.data_stats_filename), data_stats)

    all_data = list(trainset) + list(testset)
    for data in all_data:
    #Go through files to normalize
        #Node features
        #Save normalized Data
        save_path = data.path.replace(cfg.processed_dir, cfg.normalized_dir)
        x = data['x']
        #node power
        for j in range(len(x_max)):
            if any(torch.isnan(x[:,j])):
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
        if cfg.gen_feature_index >= 0:
            for j in range(8):
                if j == 3 or j == 1:
                    x[:,j+cfg.gen_feature_index] = torch.log(x[:,j+cfg.gen_feature_index]+1-x_min[j+cfg.gen_feature_index])/torch.log(cfg.x_max[j+cfg.gen_feature_index]+1-cfg.x_min[j+cfg.gen_feature_index]) #10 is the first gen feature in node_features after 6 node features + 4 features for one hot encoded bus type    
                else:
                    x[:,j+cfg.gen_feature_index] = torch.log(x[:,j+cfg.gen_feature_index]+1)/torch.log(x_max[j+cfg.gen_feature_index]+1) #10 is the first gen feature in node_features after 6 node features + 4 features for one hot encoded bus type

        for i in range(len(x_max)):
            if any(torch.isnan(x[:,i])):
                print('NaN After Normalization x:')
                for j in range(len(x[:,i])):
                    if torch.isnan(x[j,i]): print(f'After, x{i} {j}')


        #Edge Features
        edge_attr = data['edge_attr']
        adj = data['edge_index']

        #when Y is used as edge feature there are the old version (with Y=sqrt(Y.real**2+Y.imag**2)) i.e. 1 feature 
        # and the newer version where the real and imag part are saved seperately i.e. 2 features
        if N_Edge_Features <= 2 : 

            #edge_attr = torch.log(data['edge_attr']-edge_attr_min+1)/torch.log(edge_attr_max+1)
            edge_attr[:,0] = (data['edge_attr'][:,0]-edge_attr_means[0])/edge_stds[0]/((edge_attr_max[0]-edge_attr_means[0])/edge_stds[0])
            edge_attr[:,1] = (data['edge_attr'][:,1]-edge_attr_means[1])/edge_stds[1]/((edge_attr_max[1]-edge_attr_means[1])/edge_stds[1])
            
        else:   #Multiple edge features
        
            #capacity
            edge_attr[:,0] = torch.log(data['edge_attr'][:,0]+1)/torch.log(edge_attr_max[0]+1)
            #Pf, QF and resistance
            if any(torch.isnan(edge_attr[:,0])) or any(torch.isnan(edge_attr[:,1])) or any(torch.isnan(edge_attr[:,2])) or any(torch.isnan(edge_attr[:,3])) or any(torch.isnan(edge_attr[:,4])) or any(torch.isnan(edge_attr[:,5])):
                print('NaN in edges Before Normalization:')
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
        
        
        if N_Targets == 1:
            node_labels = (data['node_labels']-node_labels_means)/node_label_stds
            if any(torch.isnan(node_labels)):
                print('NaN in node labels after norm:')
                for i in range(len(node_labels)):
                    if torch.isnan(node_labels[i]): print(f'After {i}')
        else:
            node_labels[:,0] = torch.sign(data['node_labels'][:,0])*torch.log(torch.abs(data['node_labels'][:,0])+1)/torch.log(node_labels_max[0]+1)
            node_labels[:,1] = torch.sign(data['node_labels'][:,1])*torch.log(torch.abs(data['node_labels'][:,1])+1)/6    #/torch.log(node_labels_max[1]+1.1)
            if any(torch.isnan(node_labels[:,0])) or any(torch.isnan(node_labels[:,1])):
                for i in range(len(node_labels)):
                    if torch.isnan(node_labels[i,0]): print(f'After {i} label0') 
                    if torch.isnan(node_labels[i,1]): print(f'After {i} label1') 
        #Graph Labels
        if 'y' in data.keys():    
            graph_label = torch.log(data['y']+1)/torch.log(graph_label_max+1)
            data = Data(x=x, edge_index=adj, edge_attr=edge_attr, node_labels=node_labels, y=graph_label) 
        else:
            data = Data(x=x, edge_index=adj, edge_attr=edge_attr, node_labels=node_labels) 
        torch.save(data, save_path)


