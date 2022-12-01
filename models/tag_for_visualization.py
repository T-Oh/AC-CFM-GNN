# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 11:21:50 2022

@author: tobia
"""
from torch_geometric.nn import Sequential, TAGConv, global_add_pool, global_mean_pool
from torch.nn import Module, ReLU, Dropout, Sigmoid, Linear, Sequential
import torch
import numpy as np
import hiddenlayer as hl

class TAGNet01(Module):
    def __init__(self, num_node_features=2, num_targets=1, hidden_size=64, num_layers=1, dropout=.15):
        super(TAGNet01, self).__init__()
        self.conv1 = TAGConv(num_node_features, hidden_size).to(float)
        self.conv2 = TAGConv(hidden_size, hidden_size).to(float)
        self.endconv = TAGConv(hidden_size, num_targets,bias=False).to(float)
        self.endLinear = Linear(hidden_size,num_targets,bias=False).to(float)
        self.endSigmoid = Sigmoid()
        self.pool = global_mean_pool    #global add pool does not work for it produces too large negative numbers
        self.dropout = Dropout(p=dropout)
        self.relu = ReLU()
        self.num_layers = num_layers

    def forward(self, data):
        x=torch.tensor(data[0:2000,:],requires_grad=False)
        edge_index=torch.tensor(data[2001:7988,:],requires_grad=False).type(torch.int64)

        x = self.conv1(x=x,edge_index=edge_index)
        x.requires_grad=False


        for _ in range(self.num_layers - 2):
            x = self.relu(x)
            #x = self.endSigmoid(x)
            print(x)
            x = self.dropout(x)
            print(x)
            x = self.conv2(x=x)
            print(x)

        x = self.relu(x)
        x.requires_grad=False
        #x = self.endSigmoid(x)
        #print(x)
        x = self.dropout(x)
        x.requires_grad=False
        #print(x)

        x = self.endconv(x=x,edge_index=edge_index.type(torch.int64))   #produces negative values for some reason
        x.requires_grad=False

        x = self.pool(x)
        x.requires_grad=False
        
        #print("Pool")
        #print(x)
        x = self.endSigmoid(x)
        x.requires_grad=False
        #print(x)
        #print("END")
        return x
    
model=Sequential(
    TAGConv(2, 64).to(float),
    ReLU(),
    Dropout(p=0.5),
    TAGConv(64, 1,bias=False).to(float),
    global_mean_pool,
    Sigmoid()
    )
data=torch.zeros([7989,2],requires_grad=True)
graph=hl.build_graph(model,data)
graph.save("gnn_hiddenlayer",format="png")
