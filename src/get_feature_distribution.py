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
NO_Va = True
path='normalized/'
NAME = 'nANGF_Vcf_test'
N_NODE_FEATURES = 15    #if NodeIDs are added as features substract 2000 from N_Features
N_EDGE_FEATURES = 2




def get_hist(data, bins):
    hist = np.zeros(len(bins))
    for i in range(len(data)):
        for j in range(len(bins)-1):
            if data[i] <= bins[j+1]:
                hist[j]+=1
                break
    return hist

def get_min_max_features(path, N_node_features, N_edge_features):
    x_max = torch.zeros(N_node_features)
    x_min = torch.zeros(N_node_features)
    edge_attr_max = torch.zeros(N_edge_features)
    edge_attr_min = torch.zeros(N_edge_features)
    for i in range(N_node_features):
        x_max[i] = np.NINF
        x_min[i] = np.Inf
        if i <N_edge_features:
            edge_attr_max[i] =  np.NINF
            edge_attr_min[i] = np.Inf


    node_labels_max = 0
    node_labels_min = 1e6
    y_max = np.NINF
    y_min = np.Inf

    #loop through files
    for file in os.listdir(path):
        if not file.startswith('data'):
            continue
        print(path + file)
        data = torch.load(path+file)
        x, edge_attr, node_labels, y = data.x, data.edge_attr, data.node_labels, data.y
        for i in range(x.shape[0]):
            for j in range(len(x_max)):
                bias = 0
                if x[i,j]>x_max[j]: x_max[j]=x[i,j]
                if x[i,j]<x_min[j]: x_min[j]=x[i,j]

            
        edge_attr = data.edge_attr
        if N_edge_features == 1:
            for i in range(len(edge_attr)):
                if edge_attr[i] > edge_attr_max[0]: edge_attr_max[0] = edge_attr[i]
                if edge_attr[i] < edge_attr_min[0]: edge_attr_min[0] = edge_attr[i]
        else:
            for i in range(edge_attr.shape[0]):
                for j in range(len(edge_attr_max)):
                    if edge_attr[i,j] > edge_attr_max[j]: edge_attr_max[j] = edge_attr[i,j]
                    if edge_attr[i,j] < edge_attr_min[j]: edge_attr_min[j] = edge_attr[i,j]

        node_labels=torch.load(path+file)['node_labels']
        for i in range(len(node_labels)):
            if node_labels[i] > node_labels_max: node_labels_max = node_labels[i]
            if node_labels[i] < node_labels_min: node_labels_min = node_labels[i]

        if y > y_max:   y_max = y
        if y < y_min:   y_min = y
                
    return x_min,x_max,edge_attr_min,edge_attr_max, node_labels_min, node_labels_max, y_min, y_max




if PLOT_ONLY:
    data = np.load(NAME + '.npz')
    x_hists = data['x_hists']
    x_bins = data['x_bins']
    
    edge_hists = data['edge_hists']
    edge_bins = data['edge_bins']
    
    node_label_hist = data['node_label_hist']
    node_label_bins = data['node_label_bins']

    y_hist = data['y_hist']
    y_bins = data['y_bins']

    


        
else:
    x_min, x_max, edge_attr_min, edge_attr_max, node_labels_min, node_labels_max, y_min, y_max = get_min_max_features(path, N_NODE_FEATURES, N_EDGE_FEATURES)
    
    x_bins = np.zeros([len(x_max),10])
    edge_bins = np.zeros([N_EDGE_FEATURES, 10])
    node_label_bins = np.arange(node_labels_min,node_labels_max+node_labels_max/10,(node_labels_max-node_labels_min)/10)
    y_bins = np.arange(y_min, y_max+y_max/10, (y_max-y_min)/10)
    for i in range(len(x_max)):
        print(i)
        print(x_max[i])
        print(x_min[i])
        print(np.arange(x_min[i],x_max[i]+x_max[i]/10,(x_max[i]-x_min[i])/9))
        #if x_max[i]<=0:
        #    x_bins[i] = np.arange(x_min[i],x_max[i]-x_min[i]/10,(x_max[i]-x_min[i])/9)
        #else:
        #    x_bins[i] = np.arange(x_min[i],x_max[i]+x_max[i]/10,(x_max[i]-x_min[i])/9)
        x_bins[i] = np.linspace(x_min[i],x_max[i],10)
    #if N_EDGE_FEATURES == 1:
     #   edge_bins = np.arange(edge_attr_min[0],edge_attr_max[0]+edge_attr_max[0]/10,(edge_attr_max[0]-edge_attr_min[0])/10)

    #else:
    for i in range(N_EDGE_FEATURES):
        edge_bins[i] = np.linspace(edge_attr_min[0],edge_attr_max[0], 10)
        #edgebins6 = np.arange(0,1.1,1/10)
    
    
    first = True
    x_hists = np.zeros([len(x_max),10])
    edge_hists = np.zeros([N_EDGE_FEATURES,10])
    y_hist = np.zeros(len(y_bins))
    for file in os.listdir(path):
        if file.startswith('data'):
            data=torch.load(path+file)

            if first:            
                for i in range(len(x_max)):
                   x_hists[i] = get_hist(data['x'][:,i],x_bins[i])

                if N_EDGE_FEATURES == 1:
                    edge_hists[0] = get_hist(data['edge_attr'],edge_bins[0])   
                else:
                    for i in range(N_EDGE_FEATURES):
                        edge_hists[i] = get_hist(data['edge_attr'][:,i],edge_bins[i])

                node_label_hist=get_hist(data['node_labels'], node_label_bins)

                for j in range(len(y_bins)-1):
                    if data.y <= y_bins[j+1]:
                        y_hist[j]+=1
                        break
                first = False

            else:               
                for i in range(len(x_max)):
                    x_hists[i] += get_hist(data['x'][:,i],x_bins[i])

                if N_EDGE_FEATURES == 1:
                    edge_hists[0] += get_hist(data['edge_attr'][:],edge_bins)
                else:
                    for i in range(N_EDGE_FEATURES):
                        
                        edge_hists[i] += get_hist(np.array(data['edge_attr'][:,i]),edge_bins[i])

                node_label_hist_temp = get_hist(data['node_labels'],node_label_bins)
                node_label_hist += node_label_hist_temp

                for j in range(len(y_bins)-1):
                    if data.y <= y_bins[j+1]:
                        y_hist[j]+=1
                        break

    np.savez(NAME,  x_hists = x_hists,
                    edge_hists = edge_hists,
                    node_label_hist=node_label_hist,
                    y_hist = y_hist,
                    x_bins = x_bins,
                    edge_bins = edge_bins,
                    node_label_bins=node_label_bins,
                    y_bins = y_bins)
    
    
   
if NO_Va:
    x_bins=np.insert(x_bins,3,np.zeros(10),axis=0)
    x_hists=np.insert(x_hists,3,np.zeros(10),axis=0)
plt.rcParams['font.size'] = 16
plt.rcParams['figure.dpi'] = 300

#Plotting
fig0,ax0=plt.subplots()
ax0.bar(y_bins/10, y_hist, width=(y_bins[1]-y_bins[0])/10,align='edge', color='blue')
ax0.set_xlabel("Total Power Outage [GW]")
ax0.set_ylabel('Number of Instances')
fig0.savefig(path+"ac_graph_label_distr_"+NAME+".png", bbox_inches='tight')


fig1,ax1=plt.subplots()
ax1.bar(x_bins[0]/10,x_hists[0],width=(x_bins[0,1]-x_bins[0,0])/10,align='edge', color='green')
#ax1.set_title("Node Feature Apparent Power")
ax1.set_xlabel("Active Power [GW]")
ax1.set_ylabel('Number of Nodes')
#ax1.set_ylim(0,3e7)
#ax1.set_yticks([0,1e7,2e7,3e7])
fig1.savefig(path+"ac_node_feature_distr_P_"+NAME+".png", bbox_inches='tight')

fig2,ax2=plt.subplots()
ax2.bar(x_bins[1]/10,x_hists[1],width=(x_bins[1,1]-x_bins[1,0])/10,align='edge', color='green')
#ax2.set_title("Node Feature Voltage magnitude")
ax2.set_xlabel("Reactive Power [MVAr]")
ax2.set_ylabel('Number of Nodes')

fig2.savefig(path+"ac_node_feature_distr_Q_"+NAME+".png", bbox_inches='tight')

fig11,ax11=plt.subplots()
ax11.bar(x_bins[2],x_hists[2],width=x_bins[2,1]-x_bins[2,0],align='edge', color='green')
#ax1.set_title("Node Feature Apparent Power")
ax11.set_xlabel("Voltage Magnitude [p.u.]")
ax11.set_ylabel('Number of Nodes')
#ax11.set_ylim(0,3e7)
#ax11.set_yticks([0,1e7,2e7,3e7])
fig11.savefig(path+"ac_node_feature_distr_Vm_"+NAME+".png", bbox_inches='tight')

fig12,ax12=plt.subplots()
ax12.bar(x_bins[3],x_hists[3],width=x_bins[3,1]-x_bins[3,0],align='edge', color='green')
#ax2.set_title("Node Feature Voltage magnitude")
ax12.set_xlabel("Voltage Angle [rad]")
ax12.set_ylabel('Number of Nodes')

fig12.savefig(path+"ac_node_feature_distr_Va_"+NAME+".png", bbox_inches='tight')

fig13,ax13=plt.subplots()
ax13.bar(x_bins[4],x_hists[4],width=x_bins[4,1]-x_bins[4,0],align='edge', color='green')
#ax2.set_title("Node Feature Voltage magnitude")
ax13.set_xlabel("Shunt Susceptance")
ax13.set_ylabel('Number of Nodes')

fig13.savefig(path+"ac_node_feature_distr_Bs_"+NAME+".png", bbox_inches='tight')

fig14,ax14=plt.subplots()
ax14.bar(x_bins[5],x_hists[5],width=x_bins[5,1]-x_bins[5,0],align='edge', color='green')
#ax2.set_title("Node Feature Voltage magnitude")
ax14.set_xlabel("Base kV")
ax14.set_ylabel('Number of Nodes')

fig14.savefig(path+"ac_node_feature_distr_basekV_"+NAME+".png", bbox_inches='tight')

fig15,ax15=plt.subplots()
ax15.bar(x_bins[10]/10,x_hists[10],width=(x_bins[10,1]-x_bins[10,0])/10,align='edge', color='turquoise')
#ax2.set_title("Node Feature Voltage magnitude")
ax15.set_xlabel("Generator Active Power Output [GW]")
ax15.set_ylabel('Number of Nodes')

fig15.savefig(path+"ac_node_feature_distr_genP_"+NAME+".png", bbox_inches='tight')

fig16,ax16=plt.subplots()
ax16.bar(x_bins[11],x_hists[11],width=x_bins[11,1]-x_bins[11,0],align='edge', color='turquoise')
#ax2.set_title("Node Feature Voltage magnitude")
ax16.set_xlabel("Generator Reactive Power Output [MVar]")
ax16.set_ylabel('Number of Nodes')

fig16.savefig(path+"ac_node_feature_distr_genQ_"+NAME+".png", bbox_inches='tight')

fig17,ax17=plt.subplots()
ax17.bar(x_bins[12],x_hists[12],width=x_bins[12,1]-x_bins[12,0],align='edge', color='turquoise')
#ax2.set_title("Node Feature Voltage magnitude")
ax17.set_xlabel("Maximum Reactive Power Output [MVar]")
ax17.set_ylabel('Number of Nodes')

fig17.savefig(path+"ac_node_feature_distr_genQMax_"+NAME+".png", bbox_inches='tight')

fig18,ax18=plt.subplots()
ax18.bar(x_bins[13],x_hists[13],width=x_bins[13,1]-x_bins[13,0],align='edge', color='turquoise')
#ax2.set_title("Node Feature Voltage magnitude")
ax18.set_xlabel("Minimum Reactive Power Output [MVar]")
ax18.set_ylabel('Number of Nodes')

fig18.savefig(path+"ac_node_feature_distr_genQMin_"+NAME+".png", bbox_inches='tight')

fig19,ax19=plt.subplots()
ax19.bar(x_bins[14],x_hists[14],width=x_bins[14,1]-x_bins[14,0],align='edge', color='turquoise')
#ax2.set_title("Node Feature Voltage magnitude")
ax19.set_xlabel("Voltage Magnitude Set Point [p.u.]")
ax19.set_ylabel('Number of Nodes')

fig19.savefig(path+"ac_node_feature_distr_Vg_"+NAME+".png", bbox_inches='tight')

fig20,ax20=plt.subplots()
ax20.bar(x_bins[15],x_hists[15],width=x_bins[15,1]-x_bins[15,0],align='edge', color='turquoise')
#ax2.set_title("Node Feature Voltage magnitude")
ax20.set_xlabel("mBase (total MVA base)")
ax20.set_ylabel('Number of Nodes')

fig20.savefig(path+"ac_node_feature_distr_genmBase_"+NAME+".png", bbox_inches='tight')

fig21,ax21=plt.subplots()
ax21.bar(x_bins[16]/10,x_hists[16],width=(x_bins[16,1]-x_bins[16,0])/10,align='edge', color='turquoise')
#ax2.set_title("Node Feature Voltage magnitude")
ax21.set_xlabel("Maximum Active Power Output [GW]")
ax21.set_ylabel('Number of Nodes')

fig21.savefig(path+"ac_node_feature_distr_genPMax_"+NAME+".png", bbox_inches='tight')

fig22,ax22=plt.subplots()
ax22.bar(x_bins[17]/10,x_hists[17],width=(x_bins[17,1]-x_bins[17,0])/10,align='edge', color='turquoise')
#ax2.set_title("Node Feature Voltage magnitude")
ax22.set_xlabel("Minimum Active Power Output [GW]")
ax22.set_ylabel('Number of Nodes')

fig22.savefig(path+"ac_node_feature_distr_genPMin_"+NAME+".png", bbox_inches='tight')


if N_EDGE_FEATURES == 1:
    fig3,ax3=plt.subplots()
    ax3.bar(edge_bins[0],edge_hists[0],width=edge_bins[0,1]-edge_bins[0,0],align='edge', color='orange')
    #ax3.set_title("Edge Feature Capacity")
    ax3.set_xlabel("Admittance")
    ax3.set_ylabel('Number of Edges')
    fig3.savefig(path+"ac_edge_feature_Admittance_distr_"+NAME+".png", bbox_inches='tight')
else:

    fig3,ax3=plt.subplots()
    ax3.bar(edge_bins[0],edge_hists[0],width=edge_bins[0,1]-edge_bins[0,0],align='edge', color='orange')
    #ax3.set_title("Edge Feature Capacity")
    ax3.set_xlabel("Capacity [MVA]")
    ax3.set_ylabel('Number of Nodes')
    fig3.savefig(path+"ac_edge_feature_capacity_distr_"+NAME+".png", bbox_inches='tight')


    fig4,ax4=plt.subplots()
    ax4.bar(edge_bins[1],edge_hists[1],width=edge_bins[1,1]-edge_bins[1,0],align='edge', color='orange')
    #ax4.set_title("Active PF")
    ax4.set_xlabel("Active Power Flow [MVA]")
    ax4.set_ylabel('Number of Nodes')
    fig4.savefig(path+"ac_edge_feature_active_pf_distr_"+NAME+".png", bbox_inches='tight')

    fig6,ax6=plt.subplots()
    ax6.bar(edge_bins[2],edge_hists[2],width=edge_bins[2,1]-edge_bins[2,0],align='edge', color='orange')
    #ax6.set_title("Edge Feature reactive PF")
    ax6.set_xlabel("Reactive Power Flow [MVA]")
    ax6.set_ylabel('Number of Nodes')
    fig6.savefig(path+"ac_edge_feature_reactive_pf_distr_"+NAME+".png", bbox_inches='tight')

    fig7,ax7=plt.subplots()
    ax7.bar(edge_bins[3],edge_hists[3],width=edge_bins[3,1]-edge_bins[3,0],align='edge', color='orange')
    #ax7.set_title("Edge Feature Status")
    ax7.set_xlabel("Resistance [p.U.]")
    ax7.set_ylabel('Number of Nodes')
    fig7.savefig(path+"ac_edge_feature_resistance_distr_"+NAME+".png", bbox_inches='tight')

    fig8,ax8=plt.subplots()
    ax8.bar(edge_bins[4],edge_hists[4],width=edge_bins[4,1]-edge_bins[4,0],align='edge', color='orange')
    #ax8.set_title("Edge Feature resistance")
    ax8.set_xlabel("Reactance [.p.U.]")
    ax8.set_ylabel('Number of Nodes')
    fig8.savefig(path+"ac_edge_feature_reactance_distr_"+NAME+".png", bbox_inches='tight')

    fig9,ax9=plt.subplots()
    ax9.bar(edge_bins[5],edge_hists[5],width=edge_bins[5,1]-edge_bins[5,0],align='edge', color='orange')
    #ax9.set_title("Edge Feature reactance")
    ax9.set_xlabel("Initial Damage [p.U.]")
    ax9.set_ylabel('Number of Nodes')
    fig9.savefig(path+"ac_edge_feature_init_dmg_distr_"+NAME+".png", bbox_inches='tight')




fig5,ax5=plt.subplots()
ax5.bar(node_label_bins/10,node_label_hist,width=(node_label_bins[1]-node_label_bins[0])/10,align='edge', color='red')
#ax5.set_xlim(0,30)
#ax5.set_title("Power Outage at Nodes (NodeLabel)")
ax5.set_xlabel("Load Shed [GW]")
ax5.set_ylabel('Number of Nodes')
fig5.savefig(path+"ac_node_label_distr_"+NAME+".png", bbox_inches='tight')


