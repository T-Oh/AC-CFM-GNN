# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 09:01:42 2023

@author: tobia
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import os

#Data
#hists=np.load('correct_node_labels_histograms.npz')
path='processed/'
for file in os.listdir(path):
    data=torch.load(path+file)['x']
    x=np.arange(len(data[:,0]))
    fig1,ax1=plt.subplots()
    ax1.plot(x,data[:,0],x,data[:,1])

    
"""bins=hists['node_label_bins']


#Plotting
fig1,ax1=plt.subplots()
#ax1.set_ylim(0,8000)
print(hists['node_label_hist'])
print(hists['node_label_bins'])
ax1.bar(bins[0:9],hists['node_label_hist'],width=bins[8]-bins[7],align='edge')
ax1.set_title("Label (Power Outage)")
ax1.set_xlabel("Power Outage")
#fig1.savefig("local_subset_node_feature_distr_power.png")
"""

