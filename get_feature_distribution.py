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
    for i in range(len(data)):
        for j in range(len(bins)-1):
            if data[i] <= bins[j+1]:
                hist[j]+=1
                break
    return hist

def get_min_max_features(path):
    x_max = torch.zeros(3)
    x_min = torch.zeros(3)
    edge_attr_max = torch.zeros(5)
    edge_attr_min = torch.zeros(5)
    for i in range(5):
        edge_attr_max[i] =  np.NINF
        edge_attr_min[i] = np.Inf
        if i <3:
            x_max[i] = np.NINF
            x_min[i] = np.Inf
    node_labels_max = 0
    node_labels_min = 1e6
    #loop through files
    for file in os.listdir(path):
        if not file.startswith('data'):
            continue
        print(path + file)
        x = torch.load(path+file)['x']
        #x=data['x']
        for i in range(x.shape[0]):
            if x[i,0]>x_max[0]: x_max[0]=x[i,0]
            if x[i,0]<x_min[0]: x_min[0]=x[i,0]
            if x[i,1]>x_max[1]: x_max[1]=x[i,1]
            if x[i,1]<x_min[1]: x_min[1]=x[i,1]
            #if x[i,2]>x_max[2]: x_max[2]=x[i,2]
            #if x[i,2]<x_min[2]: x_min[2]=x[i,2]
            
        edge_attr=torch.load(path+file)['edge_attr']
        for i in range(len(edge_attr[:,0])):
            if edge_attr[i,0] > edge_attr_max[0]: edge_attr_max[0] = edge_attr[i,0]
            if edge_attr[i,0] < edge_attr_min[0]: edge_attr_min[0] = edge_attr[i,0]
            if edge_attr[i,1] > edge_attr_max[1]: edge_attr_max[1] = edge_attr[i,1]
            if edge_attr[i,1] < edge_attr_min[1]: edge_attr_min[1] = edge_attr[i,1]
            if edge_attr[i,2] > edge_attr_max[2]: edge_attr_max[2] = edge_attr[i,2]
            if edge_attr[i,2] < edge_attr_min[2]: edge_attr_min[2] = edge_attr[i,2]
            if edge_attr[i,4] > edge_attr_max[3]: edge_attr_max[3] = edge_attr[i,4]
            if edge_attr[i,4] < edge_attr_min[3]: edge_attr_min[3] = edge_attr[i,4]
            if edge_attr[i,5] > edge_attr_max[4]: edge_attr_max[4] = edge_attr[i,5]
            if edge_attr[i,5] < edge_attr_min[4]: edge_attr_min[4] = edge_attr[i,5]
            
        if torch.is_tensor(torch.load(path+file)['node_labels']):
            print('TEST')
            node_labels=torch.load(path+file)['node_labels']
            for i in range(len(node_labels)):
                if node_labels[i] > node_labels_max: node_labels_max = node_labels[i]
                if node_labels[i] < node_labels_min: node_labels_min = node_labels[i]
                
    return x_min,x_max,edge_attr_min,edge_attr_max, node_labels_min, node_labels_max



path='./processed/'
x_min, x_max, edge_attr_min, edge_attr_max, node_labels_min, node_labels_max = get_min_max_features(path)

x1bins=np.arange(x_min[0],x_max[0]+x_max[0]/10,(x_max[0]-x_min[0])/10)
x2bins=np.arange(x_min[1],x_max[1]+x_max[1]/10,(x_max[1]-x_min[1])/10)
#x3bins=np.arange(x_min[2],x_max[2],(x_max[2]-x_min[2])/10)
edgebins1 = np.arange(edge_attr_min[0],edge_attr_max[0]+edge_attr_max[0]/10,(edge_attr_max[0]-edge_attr_min[0])/10)
edgebins2 = np.arange(edge_attr_min[1],edge_attr_max[1]+edge_attr_max[1]/10,(edge_attr_max[1]-edge_attr_min[1])/10)
edgebins3 = np.arange(edge_attr_min[2],edge_attr_max[2]+edge_attr_max[2]/10,(edge_attr_max[2]-edge_attr_min[2])/10)
edgebins4 = np.arange(0,1.1,1/10)
edgebins5 = np.arange(edge_attr_min[3],edge_attr_max[3]+edge_attr_max[3]/10,(edge_attr_max[3]-edge_attr_min[3])/10)
edgebins6 = np.arange(edge_attr_min[4],edge_attr_max[4]+edge_attr_max[4]/10,(edge_attr_max[4]-edge_attr_min[4])/10)
edgebins7 = np.arange(0,1.1,1/10)


#labelbins=np.arange(label_min,label_max,(label_max-label_min)/10)
node_label_bins = np.arange(node_labels_min,node_labels_max+node_labels_max/10,(node_labels_max-node_labels_min)/10)


first = True
for file in os.listdir(path):
    if file.startswith('data'):
        data=torch.load(path+file)
        if first:
            x1hist=get_hist(data['x'][:,0],x1bins)
            x2hist = get_hist(data['x'][:,1],x2bins)
            #x3hist = get_hist(data['x'][:,2],x3bins)
            edgehist1 = get_hist(data['edge_attr'][:,0],edgebins1)
            edgehist2 = get_hist(data['edge_attr'][:,1],edgebins2)
            edgehist3 = get_hist(data['edge_attr'][:,2],edgebins3)
            edgehist4 = get_hist(data['edge_attr'][:,3],edgebins4)
            edgehist5 = get_hist(data['edge_attr'][:,4],edgebins5)
            edgehist6 = get_hist(data['edge_attr'][:,5],edgebins6)
            edgehist7 = get_hist(data['edge_attr'][:,6],edgebins7)
            #labelhist = get_hist(data['y'],labelbins)
            node_label_hist=get_hist(data['node_labels'], node_label_bins)
            first = False
        else:
            x1hist_temp=get_hist(data['x'][:,0],x1bins)
            x2hist_temp = get_hist(data['x'][:,1],x2bins)
            #x3hist_temp = get_hist(data['x'][:,2],x3bins)
            edgehist1_temp = get_hist(data['edge_attr'][:,0],edgebins1)
            edgehist2_temp = get_hist(data['edge_attr'][:,1],edgebins2)
            edgehist3_temp = get_hist(data['edge_attr'][:,2],edgebins3)
            edgehist4_temp = get_hist(data['edge_attr'][:,3],edgebins4)
            edgehist5_temp = get_hist(data['edge_attr'][:,4],edgebins5)
            edgehist6_temp = get_hist(data['edge_attr'][:,5],edgebins6)
            edgehist7_temp = get_hist(data['edge_attr'][:,6],edgebins7)
            #labelhist_temp = get_hist(data['y'],labelbins)
            node_label_hist_temp = get_hist(data['node_labels'],node_label_bins)
            x1hist+=x1hist_temp
            x2hist+=x2hist_temp
            #x3hist+=x3hist_temp
            edgehist1 += edgehist1_temp
            edgehist2 += edgehist2_temp
            edgehist3 += edgehist3_temp
            edgehist4 += edgehist4_temp
            edgehist5 += edgehist5_temp
            edgehist6 += edgehist6_temp
            edgehist7 += edgehist7_temp
            #labelhist += labelhist_temp
            node_label_hist += node_label_hist_temp
np.savez("4000_subset_data_histograms",x1hist=x1hist, x2hist=x2hist, 
                                 edgehist1=edgehist1, edgehist2=edgehist2, edgehist3=edgehist3, edgehist4=edgehist4, edgehist5=edgehist5, edgehist6=edgehist6, edgehist7=edgehist7,
                                 node_label_hist=node_label_hist,
                                 x1bins=x1bins, x2bins=x2bins, 
                                 edgebins1=edgebins1, edgebins2=edgebins2, edgebins3=edgebins3, edgebins4=edgebins4, edgebins5=edgebins5, edgebins6=edgebins6, edgebins7=edgebins7,
                                 node_label_bins=node_label_bins)


"""
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
"""

#Plotting
fig1,ax1=plt.subplots()
ax1.bar(x1bins[0:10],x1hist,width=x1bins[1]-x1bins[0],align='edge')
ax1.set_title("Node Feature Apparent Power")
#ax1.set_xlabel("Power [100MW]")
fig1.savefig("ac_node_feature_distr_active_power_4000_subset.png")

fig2,ax2=plt.subplots()
ax2.bar(x2bins[0:10],x2hist,width=x2bins[1]-x2bins[0],align='edge')
ax2.set_title("Node Feature Voltage magnitude")
#ax2.set_xlabel("Voltage Angle [degree]")
fig2.savefig("ac_node_feature_distr_vm_power_4000_subset.png")

fig3,ax3=plt.subplots()
ax3.bar(edgebins1[0:11],edgehist1,width=edgebins1[1]-edgebins1[0],align='edge')
ax3.set_title("Edge Feature Capacity")
ax3.set_xlabel("Capacity [MVA]")
fig3.savefig("ac_edge_feature_capacity_distr_4000_subset.png")

fig4,ax4=plt.subplots()
ax4.bar(edgebins2[0:10],edgehist2,width=edgebins2[1]-edgebins2[0],align='edge')
ax4.set_title("Active PF")
ax4.set_xlabel("")
fig4.savefig("ac_edge_feature_active_pf_distr_4000_subset.png")

fig6,ax6=plt.subplots()
ax6.bar(edgebins3[0:10],edgehist3,width=edgebins3[1]-edgebins3[0],align='edge')
ax6.set_title("Edge Feature reactive PF")
ax6.set_xlabel("")
fig6.savefig("ac_edge_feature_reactive_pf_distr_4000_subset.png")

fig7,ax7=plt.subplots()
ax7.bar(edgebins4[0:10],edgehist4,width=edgebins4[1]-edgebins4[0],align='edge')
ax7.set_title("Edge Feature Status")
ax7.set_xlabel("")
fig7.savefig("ac_edge_feature_status_distr_4000_subset.png")

fig8,ax8=plt.subplots()
ax8.bar(edgebins5[0:10],edgehist5,width=edgebins5[1]-edgebins5[0],align='edge')
ax8.set_title("Edge Feature resistance")
ax8.set_xlabel("")
fig8.savefig("ac_edge_feature_resistance_distr_4000_subset.png")

fig9,ax9=plt.subplots()
ax9.bar(edgebins6[0:11],edgehist6, width=edgebins6[1]-edgebins6[0], align='edge')
ax9.set_title("Edge Feature reactance")
ax9.set_xlabel("")
fig9.savefig("ac_edge_feature_reactance_distr_4000_subset.png")

fig10,ax10=plt.subplots()
ax10.bar(edgebins7[0:10],edgehist7,width=edgebins7[1]-edgebins7[0],align='edge')
ax10.set_title("Edge Feature Init Damage")
ax10.set_xlabel("")
fig10.savefig("ac_edge_feature_init_dmg_distr_4000_subset.png")

"""
fig4,ax4=plt.subplots()
ax4.bar(x3bins[0:9],x3hist,width=x3bins[1]-x3bins[0],align='edge')
ax4.set_title("Node Feature Voltage Amplitude")
ax4.set_xlabel("")
fig4.savefig("ac_node_feature_distr_voltage_amplitude.png")"""


fig5,ax5=plt.subplots()
ax5.bar(node_label_bins[0:10],node_label_hist,width=node_label_bins[1]-node_label_bins[0],align='edge')
ax5.set_title("Power Outage at Nodes (NodeLabel)")
ax5.set_xlabel("")
fig5.savefig("ac_node_label_distr_4000_subset.png")

