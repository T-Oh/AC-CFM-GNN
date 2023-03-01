# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 16:47:59 2022

@author: tobia
"""

from os.path import exists
import numpy as np

raw_path="C:/Users/tobia/OneDrive/Dokumente/Master/Semester4/Masterarbeit/line_regression_nauck_cluster/DC-CFM-GNN/subset_raw"
N_scenarios=100
labels=np.zeros(10000)
label_index=0
scenario_stats=np.zeros((N_scenarios,2))
for i in range(N_scenarios):
    step=1
    start_index=label_index
    file=raw_path+f'/scenario_{i}'+f'_{step}.npz'
    while exists(file):
        print(f'scenario {i}'+f' step {step}')
        labels[label_index]=np.load(file)['y']
        print(f'label {labels[label_index]}')
        step+=1
        label_index+=1
        file=raw_path+f'/scenario_{i}'+f'_{step}.npz'
    print(labels[start_index:label_index-1])
    scenario_stats[i,0]=np.mean(labels[start_index:label_index])
    scenario_stats[i,0]=np.std(labels[start_index:label_index])
np.savez('labels_ordered',labels=labels,stats=scenario_stats)