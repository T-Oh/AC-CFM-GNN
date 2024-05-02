#-*- coding: utf-8 -*-
"""
Created on Mon Nov 28 09:25:03 2022

@author: tobia
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_ONLY = True
NO_Va = True
path='processed/'
NAME = 'plotting_test'


def get_hist(data, bins):
    hist = np.zeros(len(bins))
    for i in range(len(data)):
        for j in range(len(bins)-1):
            if data[i] <= bins[j+1]:
                hist[j]+=1
                break
    return hist

def get_min_max_features(path):
    x_max = torch.zeros(18)
    x_min = torch.zeros(18)
    edge_attr_max = torch.zeros(5)
    edge_attr_min = torch.zeros(5)
    for i in range(18):
        x_max[i] = np.NINF
        x_min[i] = np.Inf
        if i <5:
            edge_attr_max[i] =  np.NINF
            edge_attr_min[i] = np.Inf


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
            for j in range(len(x_max)):
                bias = 0
                if x[i,j]>x_max[j]: x_max[j]=x[i,j]
                if x[i,j]<x_min[j]: x_min[j]=x[i,j]

            
        edge_attr=torch.load(path+file)['edge_attr']
        for i in range(len(edge_attr[:,0])):
            if edge_attr[i,0] > edge_attr_max[0]: edge_attr_max[0] = edge_attr[i,0]
            if edge_attr[i,0] < edge_attr_min[0]: edge_attr_min[0] = edge_attr[i,0]
            if edge_attr[i,1] > edge_attr_max[1]: edge_attr_max[1] = edge_attr[i,1]
            if edge_attr[i,1] < edge_attr_min[1]: edge_attr_min[1] = edge_attr[i,1]
            if edge_attr[i,2] > edge_attr_max[2]: edge_attr_max[2] = edge_attr[i,2]
            if edge_attr[i,2] < edge_attr_min[2]: edge_attr_min[2] = edge_attr[i,2]
            if edge_attr[i,3] > edge_attr_max[3]: edge_attr_max[3] = edge_attr[i,3]
            if edge_attr[i,3] < edge_attr_min[3]: edge_attr_min[3] = edge_attr[i,3]
            if edge_attr[i,4] > edge_attr_max[4]: edge_attr_max[4] = edge_attr[i,4]
            if edge_attr[i,4] < edge_attr_min[4]: edge_attr_min[4] = edge_attr[i,4]
            
        if torch.is_tensor(torch.load(path+file)['node_labels']):
            print('TEST')
            node_labels=torch.load(path+file)['node_labels']
            for i in range(len(node_labels)):
                if node_labels[i] > node_labels_max: node_labels_max = node_labels[i]
                if node_labels[i] < node_labels_min: node_labels_min = node_labels[i]
                
    return x_min,x_max,edge_attr_min,edge_attr_max, node_labels_min, node_labels_max




if PLOT_ONLY:
    data = np.load(NAME + '.npz')
    x_hists = data['x_hists']
    
    x_bins = data['x_bins']
    
    edgehist1 = data['edgehist1']
    edgehist2 = data['edgehist2']
    edgehist3 = data['edgehist3']
    edgehist4 = data['edgehist4']
    edgehist5 = data['edgehist5']
    edgehist6 = data['edgehist6']
    #edgehist7 = data['edgehist7']
    
    edgebins1 = data['edgebins1']
    edgebins2 = data['edgebins2']
    edgebins3 = data['edgebins3']
    edgebins4 = data['edgebins4']
    edgebins5 = data['edgebins5']
    edgebins6 = data['edgebins6']
    #edgebins7 = data['edgebins7']
    
    node_label_hist = data['node_label_hist']
    
    node_label_bins = data['node_label_bins']
    


        
else:
    x_min, x_max, edge_attr_min, edge_attr_max, node_labels_min, node_labels_max = get_min_max_features(path)
    
    x_bins = np.zeros([len(x_max),10])
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
    """x1bins=np.arange(x_min[0],x_max[0]+x_max[0]/10,(x_max[0]-x_min[0])/10)
    x2bins=np.arange(x_min[1],x_max[1]+x_max[1]/10,(x_max[1]-x_min[1])/10)
    x3bins=np.arange(x_min[2],x_max[2]+x_max[2]/10,(x_max[2]-x_min[2])/10)
    x4bins=np.arange(x_min[3],x_max[3]+x_max[3]/10,(x_max[3]-x_min[3])/10)"""
    edgebins1 = np.arange(edge_attr_min[0],edge_attr_max[0]+edge_attr_max[0]/10,(edge_attr_max[0]-edge_attr_min[0])/10)
    edgebins2 = np.arange(edge_attr_min[1],edge_attr_max[1]+edge_attr_max[1]/10,(edge_attr_max[1]-edge_attr_min[1])/10)
    edgebins3 = np.arange(edge_attr_min[2],edge_attr_max[2]+edge_attr_max[2]/10,(edge_attr_max[2]-edge_attr_min[2])/10)
    #edgebins4 = np.arange(0,1.1,1/10)
    edgebins4 = np.arange(edge_attr_min[3],edge_attr_max[3]+edge_attr_max[3]/10,(edge_attr_max[3]-edge_attr_min[3])/10)
    edgebins5 = np.arange(edge_attr_min[4],edge_attr_max[4]+edge_attr_max[4]/10,(edge_attr_max[4]-edge_attr_min[4])/10)
    edgebins6 = np.arange(0,1.1,1/10)
    
    
    #labelbins=np.arange(label_min,label_max,(label_max-label_min)/10)
    node_label_bins = np.arange(node_labels_min,node_labels_max+node_labels_max/10,(node_labels_max-node_labels_min)/10)
    
    
    first = True
    x_hists = np.zeros([len(x_max),10])
    for file in os.listdir(path):
        if file.startswith('data'):
            data=torch.load(path+file)
            if first:
                
                for i in range(len(x_max)):
                   x_hists[i] = get_hist(data['x'][:,i],x_bins[i])
                """x1hist=get_hist(data['x'][:,0],x1bins)
                x2hist = get_hist(data['x'][:,1],x2bins)
                x3hist=get_hist(data['x'][:,2],x3bins)
                x4hist = get_hist(data['x'][:,3],x4bins)"""

                edgehist1 = get_hist(data['edge_attr'][:,0],edgebins1)
                edgehist2 = get_hist(data['edge_attr'][:,1],edgebins2)
                edgehist3 = get_hist(data['edge_attr'][:,2],edgebins3)
                #edgehist4 = get_hist(data['edge_attr'][:,3],edgebins4)
                edgehist4 = get_hist(data['edge_attr'][:,3],edgebins4)
                edgehist5 = get_hist(data['edge_attr'][:,4],edgebins5)
                edgehist6 = get_hist(data['edge_attr'][:,5],edgebins6)
                #labelhist = get_hist(data['y'],labelbins)
                node_label_hist=get_hist(data['node_labels'], node_label_bins)
                first = False
            else:
                
                for i in range(len(x_max)):
                    x_hists[i] += get_hist(data['x'][:,i],x_bins[i])
                """x2hist_temp = get_hist(data['x'][:,1],x2bins)
                x3hist_temp=get_hist(data['x'][:,2],x3bins)
                x4hist_temp = get_hist(data['x'][:,3],x4bins)"""

                edgehist1_temp = get_hist(data['edge_attr'][:,0],edgebins1)
                edgehist2_temp = get_hist(data['edge_attr'][:,1],edgebins2)
                edgehist3_temp = get_hist(data['edge_attr'][:,2],edgebins3)
                edgehist4_temp = get_hist(data['edge_attr'][:,3],edgebins4)
                edgehist5_temp = get_hist(data['edge_attr'][:,4],edgebins5)
                edgehist6_temp = get_hist(data['edge_attr'][:,5],edgebins6)
                #edgehist7_temp = get_hist(data['edge_attr'][:,6],edgebins7)
                #labelhist_temp = get_hist(data['y'],labelbins)
                node_label_hist_temp = get_hist(data['node_labels'],node_label_bins)

                edgehist1 += edgehist1_temp
                edgehist2 += edgehist2_temp
                edgehist3 += edgehist3_temp
                edgehist4 += edgehist4_temp
                edgehist5 += edgehist5_temp
                edgehist6 += edgehist6_temp
                #edgehist7 += edgehist7_temp
                #labelhist += labelhist_temp
                node_label_hist += node_label_hist_temp
    np.savez(NAME,x_hists = x_hists, edgehist1=edgehist1, edgehist2=edgehist2, edgehist3=edgehist3, edgehist4=edgehist4, edgehist5=edgehist5, edgehist6=edgehist6, #edgehist7=edgehist7,
                                     node_label_hist=node_label_hist,
                                     x_bins = x_bins,
                                     edgebins1=edgebins1, edgebins2=edgebins2, edgebins3=edgebins3, edgebins4=edgebins4, edgebins5=edgebins5, edgebins6=edgebins6, #edgebins7=edgebins7,
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
if NO_Va:
    x_bins=np.insert(x_bins,3,np.zeros(10),axis=0)
    x_hists=np.insert(x_hists,3,np.zeros(10),axis=0)
plt.rcParams['font.size'] = 16
plt.rcParams['figure.dpi'] = 300
#Plotting
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



fig3,ax3=plt.subplots()
ax3.bar(edgebins1,edgehist1,width=edgebins1[1]-edgebins1[0],align='edge', color='orange')
#ax3.set_title("Edge Feature Capacity")
ax3.set_xlabel("Capacity [MVA]")
ax3.set_ylabel('Number of Nodes')
fig3.savefig(path+"ac_edge_feature_capacity_distr_"+NAME+".png", bbox_inches='tight')

fig4,ax4=plt.subplots()
ax4.bar(edgebins2,edgehist2,width=edgebins2[1]-edgebins2[0],align='edge', color='orange')
#ax4.set_title("Active PF")
ax4.set_xlabel("Active Power Flow [MVA]")
ax4.set_ylabel('Number of Nodes')
fig4.savefig(path+"ac_edge_feature_active_pf_distr_"+NAME+".png", bbox_inches='tight')

fig6,ax6=plt.subplots()
ax6.bar(edgebins3,edgehist3,width=edgebins3[1]-edgebins3[0],align='edge', color='orange')
#ax6.set_title("Edge Feature reactive PF")
ax6.set_xlabel("Reactive Power Flow [MVA]")
ax6.set_ylabel('Number of Nodes')
fig6.savefig(path+"ac_edge_feature_reactive_pf_distr_"+NAME+".png", bbox_inches='tight')

fig7,ax7=plt.subplots()
ax7.bar(edgebins4,edgehist4,width=edgebins4[1]-edgebins4[0],align='edge', color='orange')
#ax7.set_title("Edge Feature Status")
ax7.set_xlabel("Resistance [p.U.]")
ax7.set_ylabel('Number of Nodes')
fig7.savefig(path+"ac_edge_feature_resistance_distr_"+NAME+".png", bbox_inches='tight')

fig8,ax8=plt.subplots()
ax8.bar(edgebins5,edgehist5,width=edgebins5[1]-edgebins5[0],align='edge', color='orange')
#ax8.set_title("Edge Feature resistance")
ax8.set_xlabel("Reactance [.p.U.]")
ax8.set_ylabel('Number of Nodes')
fig8.savefig(path+"ac_edge_feature_reactance_distr_"+NAME+".png", bbox_inches='tight')

fig9,ax9=plt.subplots()
ax9.bar(edgebins6,edgehist6, width=edgebins6[1]-edgebins6[0], align='edge', color='orange')
#ax9.set_title("Edge Feature reactance")
ax9.set_xlabel("Initial Damage [p.U.]")
ax9.set_ylabel('Number of Nodes')
fig9.savefig(path+"ac_edge_feature_init_dmg_distr_"+NAME+".png", bbox_inches='tight')

"""
fig10,ax10=plt.subplots()
ax10.bar(edgebins7[0:10],edgehist7,width=edgebins7[1]-edgebins7[0],align='edge')
#ax10.set_title("Edge Feature Init Damage")
ax10.set_xlabel("Initial Damage Indicator")
ax10.set_ylabel('Number of Nodes')
fig10.savefig(path+"ac_edge_feature_init_dmg_distr_"+NAME+".png", bbox_inches='tight')


fig4,ax4=plt.subplots()
ax4.bar(x3bins[0:9],x3hist,width=x3bins[1]-x3bins[0],align='edge')
ax4.set_title("Node Feature Voltage Amplitude")
ax4.set_xlabel("")
fig4.savefig("ac_node_feature_distr_voltage_amplitude.png")"""


fig5,ax5=plt.subplots()
ax5.bar(node_label_bins/10,node_label_hist,width=(node_label_bins[1]-node_label_bins[0])/10,align='edge', color='red')
#ax5.set_xlim(0,30)
#ax5.set_title("Power Outage at Nodes (NodeLabel)")
ax5.set_xlabel("Load Shed [GW]")
ax5.set_ylabel('Number of Nodes')
fig5.savefig(path+"ac_node_label_distr_"+NAME+".png", bbox_inches='tight')


