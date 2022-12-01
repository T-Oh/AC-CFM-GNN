# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 16:18:08 2022

@author: tobia
"""
import matplotlib.pyplot as plt
from torch import load
import torch
import numpy as np

def plot_loss(file=None):
    if file == None:
        losses=load("C:/Users/tobia/OneDrive/Dokumente/Master/Semester4/Masterarbeit/results/cluster/test2/losses.pt")
    else:
        losses=load(file)
    plt.plot(losses)
    #plt.xlim(1,100)
    #plt.ylim(1.705e6,1.710e6)
    plt.title("TAG LR=0.015, feature_scaling=min_max, label_scaling=max")
    plt.ylabel("MSE")
    plt.xlabel("Epoch")
    plt.show()
    
def plot_hist(file=None):
    if file == None:
        data=load("C:/Users/tobia/OneDrive/Dokumente/Master/Semester4/Masterarbeit/results/TAGNet01_supernode/labels.pt")
    else:
        data=load(file)
    plt.hist(data,bins=10)
    plt.ylim((0,110))
    plt.title("Labels")
    plt.xlabel("Outage")
    plt.ylabel("N labels")
    plt.show()
    
def find_outage_scenarios():
    labels=load("C:/Users/tobia/OneDrive/Dokumente/Master/Semester4/Masterarbeit/results/TAGNet01/100/labels.pt",map_location=torch.device('cpu'))
    scenarios=np.zeros((5,2))
    count=0
    for i in range(len(labels)):
        #print(i)
        #print(labels[i])
        #print(count)
        if labels[i]>0:
            scenarios[count,0]=i
            scenarios[count,1]=labels[i]
            count=count+1
        if count>=5:
            break
    print(scenarios)
    
def check_output():
    output=torch.load("output.pt")
    labels=torch.load("labels.pt")
    print("Labels")
    print(labels)
    print("Output")
    print(output)
    
def plot_bar():
    data=np.load("C:/Users/tobia/OneDrive/Dokumente/Master/Semester4/Masterarbeit/line_regression_nauck_cluster/data_histograms.npz")
    plt.bar(data["edgebins"][0:10],data["edgehist"],width=6.5,align='edge')
    plt.ylim(0,1e+07)
    #plt.xlim(-3,15)
    plt.title("Node Feature Voltage Angle")
    plt.xlabel("Voltage Angle [degree]")
    