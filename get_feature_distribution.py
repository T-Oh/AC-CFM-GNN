# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 09:25:03 2022

@author: tobia
"""
import torch

from datasets.dataset import create_datasets
import matplotlib.pyplot as plt
import numpy as np
import json

def get_hist(data,bins):
    hist=np.zeros(len(bins)-1)
    for i in range(len(data)):
        for j in range(len(bins)-1):
            if data[i]<bins[j+1]:
                hist[j]+=1
                break
    return hist


with open("configurations/configuration.json", "r") as io:
    cfg = json.load(io)

#dataset,_=create_datasets(root="",div=1,cfg=cfg)
x1bins=np.arange(-3,17,2)
x2bins=np.arange(-100,37.5,12.5)
edgebins=np.arange(0,71.5,6.5)

dataset=torch.load(f"processed/data_{0}.pt")
x1hist = get_hist(dataset.x[:,0],x1bins)
x2hist = get_hist(dataset.x[:,1],x2bins)
edgehist = get_hist(dataset.edge_attr,edgebins)
#print(x2hist.shape)
for i in range(4):
    dataset=torch.load(f"processed/data_{i+1}.pt")
    x1hist_temp=get_hist(dataset.x[:,0],x1bins)
    x1hist+=x1hist_temp
    x2hist_temp = get_hist(dataset.x[:,1],x2bins)
    x2hist+=x2hist_temp
    edgehist_temp = get_hist(dataset.edge_attr,edgebins)
    edgehist+=edgehist_temp
    print(edgehist)
#np.savez("data_histograms",x1hist=x1hist,x2hist=x2hist,edgehist=edgehist,x1bins=x1bins,x2bins=x2bins,edgebins=edgebins)


#Plotting
fig1,ax1=plt.subplots()
ax1.bar(x1bins[0:9],x1hist,width=2,align='edge')
ax1.set_title("Node Feature Power")
ax1.set_xlabel("Power [100MW]")
fig1.savefig("node_feature_distr_power.png")
"""
fig2,ax2=plt.subplots()
ax2.bar(x2bins[0:10],x2hist,width=12.5,align='edge')
ax2.set_title("Node Feature Voltage Angle")
ax2.set_xlabel("Voltage Angle [degree]")
fig2.savefig("node_feature_distr_va.png")

fig3,ax3=plt.subplots()
ax3.bar(edgebins[0:10],edgehist,width=6.5,align='edge')
ax3.set_title("Edge Feature Capacity")
ax3.set_xlabel("Capacity [MVA]")
fig3.savefig("edge_feature_distr.png")
"""


                



