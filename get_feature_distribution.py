#-*- coding: utf-8 -*-
"""
Created on Mon Nov 28 09:25:03 2022

@author: tobia
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
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
    x_max = torch.zeros(4)
    x_min = torch.zeros(4)
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
            if x[i,2]>x_max[2]: x_max[2]=x[i,2]
            if x[i,2]<x_min[2]: x_min[2]=x[i,2]
            if x[i,3]>x_max[3]: x_max[3]=x[i,3]
            if x[i,3]<x_min[3]: x_min[3]=x[i,3]
            
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


PLOT_ONLY = False
path='/home/tohlinger/AC-CFM-GNN-1/processed/'
NAME = 'unnormalized_data_histograms_addedFeatures_test'

if PLOT_ONLY:
    data = np.load(path + 'unnormalized_data_histograms_new.npz')
    x1hist = data['x1hist']
    x2hist = data['x2hist']
    x3hist = data['x3hist']
    x4hist = data['x4hist']
    
    x1bins = data['x1bins']
    x2bins = data['x2bins']
    x3bins = data['x3bins']
    x4bins = data['x4bins']
    
    edgehist1 = data['edgehist1']
    edgehist2 = data['edgehist2']
    edgehist3 = data['edgehist3']
    edgehist4 = data['edgehist4']
    edgehist5 = data['edgehist5']
    edgehist6 = data['edgehist6']
    edgehist7 = data['edgehist7']
    
    edgebins1 = data['edgebins1']
    edgebins2 = data['edgebins2']
    edgebins3 = data['edgebins3']
    edgebins4 = data['edgebins4']
    edgebins5 = data['edgebins5']
    edgebins6 = data['edgebins6']
    edgebins7 = data['edgebins7']
    
    node_label_hist = data['node_label_hist']
    
    node_label_bins = data['node_label_bins']
    


        
else:
    x_min, x_max, edge_attr_min, edge_attr_max, node_labels_min, node_labels_max = get_min_max_features(path)
    
    x1bins=np.arange(x_min[0],x_max[0]+x_max[0]/10,(x_max[0]-x_min[0])/10)
    x2bins=np.arange(x_min[1],x_max[1]+x_max[1]/10,(x_max[1]-x_min[1])/10)
    x3bins=np.arange(x_min[2],x_max[2]+x_max[2]/10,(x_max[2]-x_min[2])/10)
    x4bins=np.arange(x_min[3],x_max[3]+x_max[3]/10,(x_max[3]-x_min[3])/10)
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
                x3hist=get_hist(data['x'][:,2],x3bins)
                x4hist = get_hist(data['x'][:,3],x4bins)

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
    np.savez(NAME,x1hist=x1hist, x2hist=x2hist, 
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

plt.rcParams['font.size'] = 16
plt.rcParams['figure.dpi'] = 300
#Plotting
fig1,ax1=plt.subplots()
ax1.bar(x1bins[0:10]/10,x1hist,width=(x1bins[1]-x1bins[0])/10,align='edge')
#ax1.set_title("Node Feature Apparent Power")
ax1.set_xlabel("Active Power [GW]")
ax1.set_ylabel('Number of Nodes')
#ax1.set_ylim(0,3e7)
#ax1.set_yticks([0,1e7,2e7,3e7])
fig1.savefig(path+"ac_node_feature_distr_p_"+NAME+".png", bbox_inches='tight')

fig2,ax2=plt.subplots()
ax2.bar(x2bins[0:11]/10,x2hist,width=(x2bins[1]-x2bins[0])/10,align='edge')
#ax2.set_title("Node Feature Voltage magnitude")
ax2.set_xlabel("Reactive Power")
ax2.set_ylabel('Number of Nodes')

fig2.savefig(path+"ac_node_feature_distr_q_"+NAME+".png", bbox_inches='tight')

fig11,ax11=plt.subplots()
ax11.bar(x3bins[0:11],x3hist,width=x3bins[1]-x3bins[0],align='edge')
#ax1.set_title("Node Feature Apparent Power")
ax11.set_xlabel("Voltage Magnitude [p.u.]")
ax11.set_ylabel('Number of Nodes')
#ax11.set_ylim(0,3e7)
#ax11.set_yticks([0,1e7,2e7,3e7])
fig11.savefig(path+"ac_node_feature_distr_vm_"+NAME+".png", bbox_inches='tight')

fig12,ax12=plt.subplots()
ax12.bar(x4bins[0:10],x4hist,width=x4bins[1]-x4bins[0],align='edge')
#ax2.set_title("Node Feature Voltage magnitude")
ax12.set_xlabel("Voltage Angle [rad]")
ax12.set_ylabel('Number of Nodes')

fig12.savefig(path+"ac_node_feature_distr_va_"+NAME+".png", bbox_inches='tight')

fig3,ax3=plt.subplots()
ax3.bar(edgebins1[0:10],edgehist1,width=edgebins1[1]-edgebins1[0],align='edge')
#ax3.set_title("Edge Feature Capacity")
ax3.set_xlabel("Capacity [MVA]")
ax3.set_ylabel('Number of Nodes')
fig3.savefig(path+"ac_edge_feature_capacity_distr_"+NAME+".png", bbox_inches='tight')

fig4,ax4=plt.subplots()
ax4.bar(edgebins2[0:10],edgehist2,width=edgebins2[1]-edgebins2[0],align='edge')
#ax4.set_title("Active PF")
ax4.set_xlabel("Active Power Flow [MVA]")
ax4.set_ylabel('Number of Nodes')
fig4.savefig(path+"ac_edge_feature_active_pf_distr_"+NAME+".png", bbox_inches='tight')

fig6,ax6=plt.subplots()
ax6.bar(edgebins3[0:10],edgehist3,width=edgebins3[1]-edgebins3[0],align='edge')
#ax6.set_title("Edge Feature reactive PF")
ax6.set_xlabel("Reactive Power Flow [MVA]")
ax6.set_ylabel('Number of Nodes')
fig6.savefig(path+"ac_edge_feature_reactive_pf_distr_"+NAME+".png", bbox_inches='tight')

fig7,ax7=plt.subplots()
ax7.bar(edgebins4[0:10],edgehist4,width=edgebins4[1]-edgebins4[0],align='edge')
#ax7.set_title("Edge Feature Status")
ax7.set_xlabel("Status")
ax7.set_ylabel('Number of Nodes')
fig7.savefig(path+"ac_edge_feature_status_distr_"+NAME+".png", bbox_inches='tight')

fig8,ax8=plt.subplots()
ax8.bar(edgebins5[0:10],edgehist5,width=edgebins5[1]-edgebins5[0],align='edge')
#ax8.set_title("Edge Feature resistance")
ax8.set_xlabel("Resistance [.p.U.]")
ax8.set_ylabel('Number of Nodes')
fig8.savefig(path+"ac_edge_feature_resistance_distr_"+NAME+".png", bbox_inches='tight')

fig9,ax9=plt.subplots()
ax9.bar(edgebins6[0:11],edgehist6, width=edgebins6[1]-edgebins6[0], align='edge')
#ax9.set_title("Edge Feature reactance")
ax9.set_xlabel("Reactance [p.U.]")
ax9.set_ylabel('Number of Nodes')
fig9.savefig(path+"ac_edge_feature_reactance_distr_"+NAME+".png", bbox_inches='tight')

fig10,ax10=plt.subplots()
ax10.bar(edgebins7[0:10],edgehist7,width=edgebins7[1]-edgebins7[0],align='edge')
#ax10.set_title("Edge Feature Init Damage")
ax10.set_xlabel("Initial Damage Indicator")
ax10.set_ylabel('Number of Nodes')
fig10.savefig(path+"ac_edge_feature_init_dmg_distr_"+NAME+".png", bbox_inches='tight')

"""
fig4,ax4=plt.subplots()
ax4.bar(x3bins[0:9],x3hist,width=x3bins[1]-x3bins[0],align='edge')
ax4.set_title("Node Feature Voltage Amplitude")
ax4.set_xlabel("")
fig4.savefig("ac_node_feature_distr_voltage_amplitude.png")"""


fig5,ax5=plt.subplots()
ax5.bar(node_label_bins[0:10]/10,node_label_hist,width=(node_label_bins[1]-node_label_bins[0])/10,align='edge')
ax5.set_xlim(0,30)
#ax5.set_title("Power Outage at Nodes (NodeLabel)")
ax5.set_xlabel("Load Shed [GW]")
ax5.set_ylabel('Number of Nodes')
fig5.savefig(path+"ac_node_label_distr_"+NAME+".png", bbox_inches='tight')

