
import numpy as np
import torch
import os




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
            if 'y' in data.keys():   
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
            if 'y' in data.keys():
                graph_label = data['y']
                graph_label_std += (graph_label - graph_label_mean)**2

    return np.sqrt(x_stds/node_count), np.sqrt(edge_stds/edge_count), np.sqrt(node_label_stds/node_count), np.sqrt(graph_label_std/graph_count)
    

def get_min_max_features_from_dataset(dataset, n_node_features, n_edge_features, n_targets):
    # Computes min/max/means from a dataset (e.g., trainset) instead of directory
    x_max = torch.zeros(n_node_features)
    x_min = torch.zeros(n_node_features)
    x_means = torch.zeros(n_node_features)
    edge_attr_max = torch.zeros(n_edge_features)
    edge_attr_min = torch.zeros(n_edge_features)
    edge_attr_means = torch.zeros(n_edge_features)
    node_labels_max = torch.zeros(n_targets)
    node_labels_min = torch.zeros(n_targets)
    node_labels_mean = torch.zeros(n_targets)

    node_count = 0
    edge_count = 0
    graph_count = 0

    # Initialize
    for i in range(len(x_max)):
        x_max[i] = -np.Inf
        x_min[i] = np.Inf
        if i < len(edge_attr_max):
            edge_attr_max[i] = -np.Inf
            edge_attr_min[i] = np.Inf
        if i < len(node_labels_min):
            node_labels_max[i] = -np.Inf
            node_labels_min[i] = np.Inf

    graph_labels_min = np.Inf
    graph_labels_max = -np.Inf
    graph_labels_mean = 0

    # Loop through dataset
    for data in dataset:
        graph_count += 1
        x = data.x
        for i in range(x.shape[0]):
            for j in range(len(x_max)):
                if x[i, j] > x_max[j]: x_max[j] = x[i, j]
                if x[i, j] < x_min[j]: x_min[j] = x[i, j]
                x_means[j] += x[i, j]
            node_count += 1

        edge_attr = data.edge_attr
        if edge_attr.dim() == 1: edge_attr = edge_attr.unsqueeze(1)
        for i in range(len(edge_attr)):
            for j in range(len(edge_attr_max)):
                if edge_attr[i, j] > edge_attr_max[j]: edge_attr_max[j] = edge_attr[i, j]
                if edge_attr[i, j] < edge_attr_min[j]: edge_attr_min[j] = edge_attr[i, j]
                edge_attr_means[j] += edge_attr[i, j]
            edge_count += 1

        node_labels = data.node_labels
        if node_labels.dim() == 1: node_labels = node_labels.unsqueeze(1)
        for i in range(len(node_labels)):
            for j in range(len(node_labels_min)):
                if node_labels[i, j] > node_labels_max[j]: node_labels_max[j] = node_labels[i, j]
                if node_labels[i, j] < node_labels_min[j]: node_labels_min[j] = node_labels[i, j]
                node_labels_mean[j] += node_labels[i, j]

        if 'y' in data.keys():
            graph_label = data.y
            if graph_label > graph_labels_max: graph_labels_max = graph_label
            if graph_label < graph_labels_min: graph_labels_min = graph_label
            graph_labels_mean += graph_label

    return (x_min, x_max, x_means / node_count, edge_attr_min, edge_attr_max, edge_attr_means / edge_count,
            node_labels_min, node_labels_max, node_labels_mean / node_count, graph_labels_min, graph_labels_max, graph_labels_mean / graph_count)

def get_feature_stds_from_dataset(dataset, x_means, edge_means, node_label_means, graph_label_mean, n_node_features, n_edge_features, n_targets):
    x_stds = torch.zeros(n_node_features)
    edge_stds = torch.zeros(n_edge_features)
    node_label_stds = torch.zeros(n_targets)
    graph_label_std = 0
    node_count = 0
    edge_count = 0
    graph_count = 0

    for data in dataset:
        graph_count += 1
        x = data.x
        for i in range(x.shape[0]):
            for j in range(len(x_stds)):
                x_stds[j] += (x[i, j] - x_means[j]) ** 2
            node_count += 1

        edge_attr = data.edge_attr
        if edge_attr.dim() == 1: edge_attr = edge_attr.unsqueeze(1)
        for i in range(len(edge_attr)):
            for j in range(len(edge_stds)):
                edge_stds[j] += (edge_attr[i, j] - edge_means[j]) ** 2
            edge_count += 1

        node_labels = data.node_labels
        if node_labels.dim() == 1: node_labels = node_labels.unsqueeze(1)
        for i in range(node_labels.shape[0]):
            for j in range(len(node_label_stds)):
                node_label_stds[j] += (node_labels[i, j] - node_label_means[j]) ** 2

        if 'y' in data.keys():
            graph_label = data.y
            graph_label_std += (graph_label - graph_label_mean) ** 2

    return np.sqrt(x_stds / node_count), np.sqrt(edge_stds / edge_count), np.sqrt(node_label_stds / node_count), np.sqrt(graph_label_std / graph_count)
