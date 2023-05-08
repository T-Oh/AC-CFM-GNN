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
"""path='processed/'
for file in os.listdir(path):
    data=torch.load(path+file)['x']
    x=np.arange(len(data[:,0]))
    fig1,ax1=plt.subplots()
    ax1.plot(x,data[:,0],x,data[:,1])"""

    
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


def plot_step_outage(file=None, name=None):
    if name == None:
        name = 'Ike_ajusted'
    if file == None:
        data = np.load('C:/Users/tobia/OneDrive/Dokumente/Master/Semester4/Masterarbeit/results/cluster/AC_Feature_Distr/Ike_ajusted/total_step_outage_distribution_Ike_ajusted.npz')
    else:
        data = np.load(file)
    node_label_bins=data['node_label_bins']
    node_label_hist=data['node_label_hist']
    fig5,ax5=plt.subplots()
    ax5.bar(node_label_bins[0:10],node_label_hist,width=node_label_bins[1]-node_label_bins[0],align='edge')
    ax5.set_title(name)
    ax5.set_xlabel("Total Step Outage")
    ax5.set_ylabel("Count")
    ax5.set_ylim(0,4000)
    #fig5.savefig("Total_Step_Outage_Distribution_"+name+'png')