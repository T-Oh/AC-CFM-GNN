# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 16:33:19 2022

@author: tobia
"""

import numpy as np
import torch
from torch_geometric.utils import to_undirected
from torch_geometric.data import Dataset, Data

x=torch.randn(2000,2)
x=(x-torch.mean(x))/torch.std(x)
adj=torch.from_numpy(np.load("raw/scenario_9_56.npz")["adj"])
edge_attr=torch.randn(adj.shape[1])
edge_attr=(edge_attr-torch.mean(edge_attr)/torch.std(edge_attr))
y=torch.randn(1)
adj, edge_attr = to_undirected(adj, edge_attr)
data = Data(x=x, y=y, edge_index=adj, edge_attr=edge_attr)
torch.save(data,  'data_1.pt')