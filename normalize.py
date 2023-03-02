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
    x_max=torch.zeros(2)
    x_min=torch.zeros(2)
    x_means = torch.zeros(2)
    edge_attr_max=torch.zeros(5)
    edge_attr_min=torch.zeros(5)
    edge_attr_means = torch.zeros(5)
    node_count = 0
    edge_count = 0
    for i in range(5):
        edge_attr_max[i] =  np.NINF
        edge_attr_min[i] = np.Inf
        if i <2:
            x_max[i] = np.NINF
            x_min[i] = np.Inf
    node_labels_max=0
    node_labels_min=1e6
    node_labels_mean = 0
    
    
    for file in os.listdir(processed_dir):
        if file.startswith('data'):
            data = torch.load(processed_dir +'/' + file)
            x = data['x']
            for i in range(x.shape[0]):
                if x[i,0]>x_max[0]: x_max[0]=x[i,0]
                if x[i,0]<x_min[0]: x_min[0]=x[i,0]
                if x[i,1]>x_max[1]: x_max[1]=x[i,1]
                if x[i,1]<x_min[1]: x_min[1]=x[i,1]
                x_means[0] += x[i,0]
                x_means[1] += x[i,1]
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
                if node_labels[i]>node_labels_max: node_labels_max=node_labels[i]
                if node_labels[i]<node_labels_min: node_labels_min=node_labels[i]
                node_labels_mean += node_labels[i]
        

    return x_min, x_max, x_means/node_count, edge_attr_min, edge_attr_max, edge_attr_means/edge_count, node_labels_min, node_labels_max, node_labels_mean/node_count


def get_feature_stds(processed_dir, x_means, edge_means):
    x_stds = torch.zeros(2)
    edge_stds = torch.zeros(5)
    node_count = 0
    edge_count = 0
    for file in os.listdir(processed_dir):
        if file.startswith('data'):
            data = torch.load(processed_dir +'/' + file)
            x = data['x']
            for i in range(x.shape[0]):
                x_stds[0] += (x[i,0]-x_means[0])**2
                x_stds[1] += (x[i,1]-x_means[1])**2
                node_count += 1
            edge_attr = data['edge_attr']
            for i in range(len(edge_attr)):
                edge_stds[0] += (edge_attr[i,0] - edge_means[0])**2
                edge_stds[1] += (edge_attr[i,1] - edge_means[1])**2
                edge_stds[2] += (edge_attr[i,2] - edge_means[2])**2
                edge_stds[3] += (edge_attr[i,4] - edge_means[3])**2
                edge_stds[4] += (edge_attr[i,5] - edge_means[4])**2
                edge_count += 1
        return np.sqrt(x_stds/node_count), np.sqrt(edge_stds/edge_count)
    
    
processed_dir = 'processed'

x_min, x_max, x_means, edge_attr_min, edge_attr_max, edge_attr_means, node_labels_min, node_labels_max, node_labels_means = get_min_max_features(processed_dir)
x_stds, edge_stds = get_feature_stds(processed_dir, x_means, edge_attr_means)

for file in os.listdir(processed_dir):
    if file.startswith('data'):
        data = torch.load(processed_dir + '/' + file)
        #Node features
        x = data['x']
        #node power
        x[:,0] = torch.log(x[:,0]+1)/torch.log(x_max[0]+1)
        #node voltage magnitude
        x[:,1] = torch.log(x[:,0]+1)/torch.log(x_max[0]+1)  #((x[:,1]-x_means[1])/x_stds[1])/((x_max[1]-x_means[1])/x_stds[1])
        
        #Edge Features
        edge_attr = data['edge_attr']
        adj = data['edge_index']
        #capacity
        edge_attr[:,0] = torch.log(data['edge_attr'][:,0]+1)/torch.log(edge_attr_max[0]+1)
        #Pf, QF and resistance
        edge_attr[:,1] = (data['edge_attr'][:,1]-edge_attr_means[1])/edge_stds[1]/((edge_attr_max[1]-edge_attr_means[1])/edge_stds[1])
        edge_attr[:,2] = (data['edge_attr'][:,2]-edge_attr_means[2])/edge_stds[2]/((edge_attr_max[2]-edge_attr_means[2])/edge_stds[2])
        edge_attr[:,4] = (data['edge_attr'][:,4]-edge_attr_means[3])/edge_stds[3]/((edge_attr_max[3]-edge_attr_means[3])/edge_stds[3])
        #reactance
        edge_attr[:,5] = torch.log(data['edge_attr'][:,5]+1)/torch.log(edge_attr_max[4]+1)
        
        #Node Labels
        node_labels = torch.log(data['node_labels']+1)/torch.log(node_labels_max+1)
        data = Data(x=x, edge_index=adj, edge_attr=edge_attr, node_labels=node_labels) 
        torch.save(data, os.path.join(processed_dir, file))