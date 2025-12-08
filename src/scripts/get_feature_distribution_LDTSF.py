#-*- coding: utf-8 -*-
"""
Created on Mon Nov 28 09:25:03 2022

@author: tobia
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_ONLY = False
path='subset/'
NAME = 'LDTSF_subset'



if PLOT_ONLY:
    data = np.load(NAME + '.npz')
    y_hist = data['y_hist']
    y_bins = data['y_bins']
    y_class_hist = data['y_class_hist']
    y_class_bins = data['y_class_bins']

    
        
else:
    y_min = 0
    y_max = 1
    y_class_bins = [0,1,2,3,4]
    
    first = True
    
    #y_hist = np.zeros(len(y_bins))
    #class_hist = np.zeros(len(class_bins))
    for file in os.listdir(path):
        if file.startswith('data'):
            data=torch.load(path+file)
            if first:            
                y_hist, y_bins = np.histogram(data.y, bins=10, range=[y_min, y_max])
                y_class_hist, y_class_bins = np.histogram(data.y_class, bins=y_class_bins)
                first = False
            else:               
                y_hist_temp, _ = np.histogram(data.y, bins=y_bins)
                y_class_hist_temp, _ = np.histogram(data.y_class, bins=y_class_bins)
                y_hist += y_hist_temp
                y_class_hist += y_class_hist_temp

    np.savez(NAME,  
                    y_hist = y_hist,
                    y_bins = y_bins,
                    y_class_hist = y_class_hist,
                    y_class_bins = y_class_bins)
    
    


#Plotting
fig0,ax0=plt.subplots()
ax0.bar(y_bins[0:10], y_hist, width=(y_bins[1]-y_bins[0]),align='edge', color='blue')
ax0.set_xlabel("Total Power Outage [GW]")
ax0.set_ylabel('Number of Instances')
fig0.savefig(path+"ac_graph_label_distr_"+NAME+".png", bbox_inches='tight')

fig1,ax1=plt.subplots()
ax1.bar(y_class_bins[0:4], y_class_hist, width=(y_class_bins[1]-y_class_bins[0]),align='center', color='blue')
ax1.set_xlabel("Total Power Outage [GW]")
ax1.set_ylabel('Number of Instances')
ax1.set_xticks([0,1,2,3])
ax1.set_xlim([-0.5, 3.5])
fig1.savefig(path+"ac_graph_class_label_distr_"+NAME+".png", bbox_inches='tight')




