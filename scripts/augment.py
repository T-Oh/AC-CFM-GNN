# -*- coding: utf-8 -*-
"""
24.06.2024
@author: tobia
"""

"Used to perform data augmentation as described in Zhu 2023"

import shutil
import numpy as np
import os
import torch
import shutil
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data


path_from_processed='processed/'
path_augmented = 'processed/'
N_AUGMENTED = 3     #the number of augmented instances created per original instance i.e. if set to 1 the final dataset is double the size of the original dataset

def get_scenario_step_of_file(name):  
    """
    

    Parameters
    ----------
    name : string
            name of the processed data file

    Returns
    -------
    scenario : int
        Scenario of which the file stems
    step : int
        Step in that scenario

    """
    name=name[5:]
    i=0
    while name[i].isnumeric():
        i+=1
    scenario=int(name[0:i])
    j=i+1
    while name[j].isnumeric():
        j+=1
    step=int(name[i+1:j])
    return scenario,step

def permute(data):

    perm = torch.eye(2000)[torch.randperm(2000)]

    x_ = perm@data.x

    Y = torch.zeros((2000,2000))
    for idx, edge in enumerate(data.edge_index.t().tolist()):                
        source, target = edge
        Y[source, target] = data.edge_attr[idx]
    Y_ = perm@Y@perm
    edge_index_, edge_attr_ = dense_to_sparse(Y_)
    data_ = Data(x=x_, edge_index=edge_index_, edge_attr=edge_attr_, y=data.y, node_labels=data.node_labels)
    return data_


for file in os.listdir(path_from_processed):
    move = False
    if file.startswith('data'):
        data = torch.load(path_from_processed+file)
        if data.y > 0:
            scenario, step = get_scenario_step_of_file(file)
            for i in range(N_AUGMENTED):
                data_ = permute(data) 
                torch.save(data_, os.path.join(path_augmented, f'data_{scenario}_{step}{i+1000}.pt'))  









        


    