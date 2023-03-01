# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 16:33:19 2022

@author: tobia
"""

import numpy as np
import torch
from torch_geometric.utils import to_undirected
from torch_geometric.data import Dataset, Data


#CONFIG

N_nodes = 5
N_node_features = 3
N_edge_features = 1
limit=2.0
#x=torch.randn(2000,2)
x=torch.zeros(N_nodes,3)
mask=torch.randn(N_nodes,3)

count=0
for i in range(N_nodes):
    if mask[i,0]>limit: 
        x[i,0]=torch.randn(1)
        count+=1
    if mask[i,1] > limit: x[i,1]=torch.randn(1)
    if mask[i,2] > limit: x[i,2]=torch.randn(1)
print(count)
#x=(x-torch.mean(x))/torch.std(x)
#adj=torch.from_numpy(np.load("raw/scenario_9_56.npz")["adj"])
adj = torch.tensor([[0,0,1,1,4], [1,1,0,4,0]])
#edge_attr=torch.randn(adj.shape[1],2)
edge_attr=torch.zeros(adj.shape[1],2)
edge_attr[:,1]+=1
edge_mask=torch.randn(adj.shape[1])
for i in range(adj.shape[1]):
    if edge_mask[i]>limit:
        edge_attr[i,0]=torch.randn(1)
        edge_attr[i,1]=0
        
node_labels = torch.tensor([1,1,1,1,1])
#edge_attr=torch.from_numpy(np.load("raw/scenario_36_20.npz")["edge_weights"])
#print(edge_attr)
#y=torch.randn(1)
y=1.0
adj, edge_attr = to_undirected(adj, edge_attr)
data = Data(x=x.float(), node_labels =node_labels, edge_index=adj, edge_attr=edge_attr.float())
torch.save(data,  'processed/data_9_56.pt')



"""
#x=torch.randn(2000,2)
x=torch.zeros(2000,2)
#x=(x-torch.mean(x))/torch.std(x)
adj=torch.from_numpy(np.load("raw/scenario_37_15.npz")["adj"])
#edge_attr=torch.randn(adj.shape[1],2)
edge_attr=torch.zeros(adj.shape[1],2)
#y=torch.randn(1)
y=0.0
adj, edge_attr = to_undirected(adj, edge_attr)
data = Data(x=x.float(), y=y, edge_index=adj, edge_attr=edge_attr.float())
torch.save(data,  'processed/data_37_15.pt')"""