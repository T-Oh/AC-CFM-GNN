# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 14:46:05 2023

@author: tobia
"""

import numpy as np
from get_scenario_statistics import get_data_list

last_scenario=100  #index of the scenario with the highest index appearing in the dataset

raw_path='./raw/'

data_list=get_data_list(last_scenario,raw_path)

for i in range(len(data_list)-1):
    sc_pre=int(data_list[i,0])
    sc_post=int(data_list[i+1,0])
    st_pre=int(data_list[i,1])
    st_post=int(data_list[i+1,1])
    if data_list[i+1,0]!=data_list[i,0]:
        continue
    else:
        if data_list[i+1,1]!=data_list[i,1]+1:
            print(f'Steps without succession in Scenario {data_list[i,0]} after step {data_list[i,1]}')
        else:
            with np.load(f'./raw/scenario_{sc_pre}_{st_pre}.npz') as data_pre:  #load data of scenarios
                with np.load(f'./raw/scenario_{sc_post}_{st_post}.npz') as data_post:
                    node_labels=data_pre['x'][:,0]-data_post['x'][:,0]
                    data_pre=dict(data_pre)
                    data_pre['node_labels']=node_labels
                    np.savez(f'./raw/scenario_{sc_pre}_{st_pre}.npz',**data_pre)
            
        
            

