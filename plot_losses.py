# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 16:18:08 2022

@author: tobia
"""
import matplotlib.pyplot as plt
from torch import load
import torch
import numpy as np

def analysis(folder):
    plot_loss(folder+'losses.pt', 'inputmask')
    
    labels=torch.load(folder+'labels.pt',map_location=torch.device('cpu'))
    output=torch.load(folder+'output.pt',map_location=torch.device('cpu'))
    
    for i in range(len(output)-1):
        fig1,ax1=plt.subplots()
        x_ticks = np.array(range(2000))
        ax1.bar(x_ticks, labels[len(output)-1][i*2000:(i+1)*2000])
        ax1.bar(x_ticks, output[len(output)-1][i*2000:(i+1)*2000].detach().numpy())
        ax1.set_ylim(-0.1,1)
        ax1.set_title("Load Shed at Node")
        ax1.set_xlabel("Node ID")
    
        ax1.set_ylabel('Load Shed in p.U.')


    
def plot_train_test_loss(train_loss=None, test_loss = None):
    if train_loss == None:
        train_loss=load("C:/Users/tobia/OneDrive/Dokumente/Master/Semester4/Masterarbeit/results/cluster/GINE/4000/New_R2/MaskingTests/lossmask/train_losses_5L_21HF_0.000lr_0.07GC_TrueSC.pt")
        test_loss = load("C:/Users/tobia/OneDrive/Dokumente/Master/Semester4/Masterarbeit/results/cluster/GINE/4000/New_R2/mask_conv_test/test_losses_5L_21HF_0.000lr_0.07GC_TrueSC.pt")
    else:
        train_loss=load(train_loss)
        test_loss = load(test_loss)

    fig = plt.figure()
    ax = fig.add_subplot(111,label='train_loss')
    ax2 = fig.add_subplot(111,label='test_loss', frame_on=False)
    ax.plot(train_loss, color='orange', label='train_loss')
    ax2.plot([0,100,200,300,400,500,600],test_loss, label='test_loss')
    ax2.set_ylim(0.047,0.06)
    ax.set_ylim(0.047,0.06)
    ax2.set_xlim(-50,750)
    ax.set_xlim(-50,750)
    plt.title("4000 subset")
    plt.ylabel("MSE")
    plt.xlabel("Epoch")
    ax.legend()
    ax2.legend(loc='center right')
    plt.show()


def plot_loss(file=None,title=None):
    if file == None:
        losses=load("C:/Users/tobia/OneDrive/Dokumente/Master/Semester4/Masterarbeit/results/cluster/GAT/4000_subset/output_through_time_5000epochs/losses.pt")
    else:
        losses=load(file)
    plt.plot(losses)
    #plt.xlim(1,100)
    #plt.ylim(1.705e6,1.710e6)
    plt.title(title)
    plt.ylabel("MSE")
    plt.xlabel("Epoch")
    plt.show()
    
def plot_hist(file=None):
    if file == None:
        data=load("C:/Users/tobia/OneDrive/Dokumente/Master/Semester4/Masterarbeit/results/cluster/subset/logtrans/labels.pt")
    else:
        data=load(file)
    plt.hist(data,bins=10)
    #plt.ylim((0,110))
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
    