# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 13:41:24 2022

@author: tobia
"""
import os
import numpy as np


def get_scenario_step_of_file(name):       
    name=name[9:]
    i=1
    while name[i].isnumeric():
        i+=1
    scenario=int(name[0:i])
    j=i+1
    while name[j].isnumeric():
        j+=1
    step=int(name[i+1:j])
    return scenario,step


def get_data_list(N_scenarios,raw_path='./raw/'):
    """Returns data_list a 2D array which has in the first column the scenario number and in the second column the step number of all
    files in  the dataset
    N_scenario must be last Scenario that appears in raw (if scenario 1,2 and 100 are used N_scenarios must be 100)"""
    raw_paths=os.listdir(raw_path)
    data_list=np.zeros((len(raw_paths),2))
    idx=0                           
    for i in range(N_scenarios):
        first=True
        for file in raw_paths:
            if file.startswith(f'scenario_{i+1}_'):
                
                _,step=get_scenario_step_of_file(file)
                if first:
                    first=False                        
                    scenario_list=[step]
                else:
                    scenario_list.append(step)
        if not first:   #just so scenarios that dont appear in the dataset are not called in the next part

            scenario_list=np.sort(np.array(scenario_list))
            data_list[idx:idx+len(scenario_list),1]=scenario_list
            data_list[idx:idx+len(scenario_list),0]=i+1
            idx+=len(scenario_list)
    return data_list

"""
N_scenarios=100
data_list=get_data_list(N_scenarios)
scenario_stats=np.zeros((100,2))
idx=0
raw_path='./subset_raw/'
for i in range(N_scenarios):
    first=True
    while data_list[idx,0]==i+1:
        scenario=i+1
        step=data_list[idx,1]

        idx+=1
        print(os.path.join(raw_path, f'data_{scenario}'
                                       f'_{step}.pt'))
        if first:
            labels = [np.load(os.path.join(raw_path, f'data_{scenario}'
                                           f'_{step}.pt'))['y']]
            first=False
        else:
            labels.append(np.load(os.path.join(raw_path, f'data_{scenario}'
                                           f'_{step}.pt'))['y'])
    if not first:
        mean=np.mean(labels)
        std=np.std(labels)
        scenario_stats[i,0]=mean
        scenario_stats[i,1]=std
"""


