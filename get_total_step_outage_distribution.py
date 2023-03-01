# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 14:34:34 2023

@author: tobia
"""

#-*- coding: utf-8 -*-
"""
Created on Mon Nov 28 09:25:03 2022

@author: tobia
"""
import torch

from datasets.dataset import create_datasets
import matplotlib.pyplot as plt
import numpy as np
import json
import os




def get_hist(data, bins):
    hist = np.zeros(len(bins)-1)
    for j in range(len(bins)-1):
        if data <= bins[j+1]:
            hist[j]+=1
            break
    return hist

def get_min_max_features(path):

    node_labels_max = 0
    node_labels_min = 1e6
    #loop through files
    for file in os.listdir(path):
        if not file.startswith('data'):
            continue
        print(path + file)
        
        node_labels=torch.load(path+file)['node_labels']
        
        if node_labels.sum() > node_labels_max: node_labels_max = node_labels.sum()
        if node_labels.sum() < node_labels_min: node_labels_min = node_labels.sum()
                
    return node_labels_min, node_labels_max



path='./processed/'
node_labels_min, node_labels_max = get_min_max_features(path)

#labelbins=np.arange(label_min,label_max,(label_max-label_min)/10)
node_label_bins = np.arange(node_labels_min,node_labels_max+node_labels_max/10,(node_labels_max-node_labels_min)/10)


first = True
for file in os.listdir(path):
    if file.startswith('data'):
        data=torch.load(path+file)
        if first:
            node_label_hist=get_hist(data['node_labels'].sum(), node_label_bins)
            first = False
        else:

            node_label_hist_temp = get_hist(data['node_labels'].sum(),node_label_bins)
            node_label_hist += node_label_hist_temp
np.savez("total_step_outage_distribution",node_label_hist=node_label_hist, node_label_bins=node_label_bins)

#Plotting
fig5,ax5=plt.subplots()
ax5.bar(node_label_bins[0:10],node_label_hist,width=node_label_bins[1]-node_label_bins[0],align='edge')
ax5.set_title("Power Outage at Nodes (NodeLabel)")
ax5.set_xlabel("")
fig5.savefig("ac_node_label_distr.png")



                



