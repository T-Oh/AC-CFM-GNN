# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 14:10:09 2023

@author: tobia
"""

import numpy as np
import os
from get_scenario_statistics import get_data_list

path='./raw/'

datalist=get_data_list(100,path)

for i in range(len(datalist)):
    file=f'scenario_{int(datalist[i,0])}_{int(datalist[i,1])}.npz'
    print(file)
    data=dict(np.load(path+file))
    edge_weights=data['edge_weights']
    new_edge_weights=np.zeros((2,len(edge_weights)))
    for i in range(len(edge_weights)):
        new_edge_weights[0,i]=edge_weights[i]
        if edge_weights[i]>0: new_edge_weights[1,i]=1

    data['edge_weights']=new_edge_weights
    print(data['edge_weights'])
    np.savez(path+file,**data)
    

    