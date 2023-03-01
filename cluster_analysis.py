# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 11:59:56 2023

@author: tobia
"""

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

raw_path='./raw/'
plot_basic_analysis=True
power=[]
angle=[]
label=[]
capacity=[]
for file in os.listdir(raw_path):
    data=np.load(raw_path+file)
    power.append(data['x'][:,0].mean())
    angle.append(data['x'][:,1].mean())
    capacity.append(data['edge_weights'][0].mean())
    label.append(data['y'])

#Kmeans
features=np.array([power,angle,capacity])
kmeans=KMeans(n_clusters=2).fit(features.T)
print(kmeans.labels_)
print(kmeans.cluster_centers_)





#Plotting
fig3 = plt.figure()
ax3 = fig3.add_subplot(projection='3d')
ax3.scatter(features[0],features[1],features[2],c=kmeans.labels_)
ax3.set_xlabel('Node Power')
ax3.set_ylabel('Node Voltage Angle')
ax3.set_zlabel('Edge Capacity')
ax3.set_title('Kmeans 2 clusters')
fig3.savefig('kmeans_plot_2clusters.png')

if plot_basic_analysis:    
    fig = plt.figure()
    ax1 = fig.add_subplot(projection='3d')
    ax1.scatter(power,angle,capacity,c=label)
    ax1.set_xlabel('Node Power')
    ax1.set_ylabel('Node Voltage Angle')
    ax1.set_zlabel('Edge Capacity')
    fig.savefig('feature_plot_3d.png')
    
    fig2, ax2 = plt.subplots()
    ax2.scatter(power,angle,c=label)
    ax2.set_xlabel('Node Power')
    ax2.set_ylabel('Node Voltage Angle')
    fig.savefig('feature_plot_2d.png')

        
