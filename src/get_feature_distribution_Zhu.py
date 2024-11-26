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
NAME = 'addY_unnormalized'
N_NODE_FEATURES = 8   #if NodeIDs are added as features subtract 2000 from N_Features
N_EDGE_FEATURES = 2
N_TARGETS = 2
N_GRAPH_LABELS = 0

def get_hist(data, bins):
    hist = np.zeros(len(bins))
    for i in range(len(data)):
        for j in range(len(bins)-1):
            if data[i] <= bins[j+1]:
                hist[j] += 1
                break
    return hist


def get_min_max_features(processed_dir, n_node_features, n_edge_features, n_targets):
    #identifies and saves the min and max values as well as the mean values of all features and labels of the data
    
    #Variables to save the min/max/means
    x_max=torch.zeros(n_node_features)
    x_min=torch.zeros(n_node_features)
    edge_attr_max=torch.zeros(n_edge_features)
    edge_attr_min=torch.zeros(n_edge_features)
    node_labels_max = torch.zeros(n_targets)
    node_labels_min = torch.zeros(n_targets)




    #Initialize mins and max values
    for i in range(len(x_max)):
        #Nodefeatures
        x_max[i] = -np.Inf
        x_min[i] = np.Inf
        #EdgeFeatures
        if i <len(edge_attr_max):
            edge_attr_max[i] = -np.Inf
            edge_attr_min[i] = np.Inf
        #NodeLabels
        if i < len(node_labels_min):
            node_labels_max[i] = -np.Inf
            node_labels_min[i] = np.Inf

    #Same for Graph Label

    graph_labels_min = np.inf
    graph_labels_max = -np.inf

    
    #Loop through files
    for file in os.listdir(processed_dir):
        if file.startswith('data'): #only process data files
            data = torch.load(processed_dir +'/' + file)
            #Nodes
            x = data['x']
            for i in range(x.shape[0]): #node_loop
                for j in range(len(x_max)): #feature_loop
                    if x[i,j]>x_max[j]: x_max[j]=x[i,j]
                    if x[i,j]<x_min[j]: x_min[j]=x[i,j]

            #Edges
            edge_attr = data['edge_attr']
            if edge_attr.dim() == 1: edge_attr = edge_attr.unsqueeze(1)
            for i in range(len(edge_attr)):
                for j in range(len(edge_attr_max)):
                    if edge_attr[i,j]>edge_attr_max[j]: edge_attr_max[j]=edge_attr[i,j]
                    if edge_attr[i,j]<edge_attr_min[j]: edge_attr_min[j]=edge_attr[i,j]


                
            #Node Labels
            node_labels = data['node_labels']
            for i in range(len(node_labels)):
                for j in range(len(node_labels_min)):
                    if node_labels[i,j] > node_labels_max[j]: node_labels_max[j] = node_labels[i,j]
                    if node_labels[i,j] < node_labels_min[j]: node_labels_min[j] = node_labels[i,j]

            #Graph Labels
            if 'y' in data.keys:   
                graph_label = data['y']
                if graph_label > graph_labels_max: graph_labels_max = graph_label
                if graph_label < graph_labels_min: graph_labels_min = graph_label

            
        

    return x_min, x_max, edge_attr_min, edge_attr_max, node_labels_min, node_labels_max,  graph_labels_min, graph_labels_max



if PLOT_ONLY:
    data = np.load(NAME + '.npz')
    x_hists = data['x_hists']
    x_bins = data['x_bins']
    edge_hists = data['edge_hists']
    edge_bins = data['edge_bins']
    node_label_hists = data['node_label_hist']
    node_label_bins = data['node_label_bins']
    y_hist = data['y_hist']
    y_bins = data['y_bins']

else:
    x_min, x_max, edge_attr_min, edge_attr_max, node_labels_min, node_labels_max, y_min, y_max = get_min_max_features(path, N_NODE_FEATURES, N_EDGE_FEATURES, N_TARGETS)

    x_bins = np.zeros([len(x_max), 10])
    edge_bins = np.zeros([N_EDGE_FEATURES, 10])
    node_label_bins = np.zeros([len(node_labels_max), 10])
    y_bins = np.linspace(y_min, y_max, 10)

    for i in range(len(x_max)):
        x_bins[i] = np.linspace(x_min[i], x_max[i], 10)

    for i in range(N_EDGE_FEATURES):
        edge_bins[i] = np.linspace(edge_attr_min[i], edge_attr_max[i], 10)

    for i in range(len(node_labels_max)):
        print(node_labels_min[i])
        print(node_labels_max[i])
        node_label_bins[i] = np.linspace(node_labels_min[i], node_labels_max[i], 10)


    first = True
    x_hists = np.zeros([len(x_max), 10])
    edge_hists = np.zeros([N_EDGE_FEATURES, 10])
    node_label_hists = np.zeros([len(node_labels_max), 10])
    y_hist = np.zeros(len(y_bins))

    for file in os.listdir(path):
        if file.startswith('data'):
            data = torch.load(path+file)

            if first:
                for i in range(len(x_max)):
                    x_hists[i] = get_hist(data['x'][:, i], x_bins[i])

                if N_EDGE_FEATURES == 1:
                    edge_hists[0] = get_hist(data['edge_attr'], edge_bins[0])
                else:
                    for i in range(N_EDGE_FEATURES):
                        edge_hists[i] = get_hist(data['edge_attr'][:, i], edge_bins[i])

                for i in range(N_TARGETS):
                    node_label_hists[i] = get_hist(data['node_labels'][:,i], node_label_bins[i])

                if 'y' in data.keys:
                    for j in range(len(y_bins)-1):
                        if data.y <= y_bins[j+1]:
                            y_hist[j] += 1
                            break

                first = False

            else:
                for i in range(len(x_max)):
                    x_hists[i] += get_hist(data['x'][:, i], x_bins[i])

                if N_EDGE_FEATURES == 1:
                    edge_hists[0] += get_hist(data['edge_attr'][:], edge_bins)
                else:
                    for i in range(N_EDGE_FEATURES):
                        edge_hists[i] += get_hist(data['edge_attr'][:, i], edge_bins[i])

                for i in range(len(node_labels_max)):
                    node_label_hists[i] += get_hist(data['node_labels'][:,i], node_label_bins[i])

                if 'y' in data.keys:
                    for j in range(len(y_bins)-1):
                        if data.y <= y_bins[j+1]:
                            y_hist[j] += 1
                            break

    np.savez(NAME, x_hists=x_hists, edge_hists=edge_hists, node_label_hist=node_label_hists, y_hist=y_hist, x_bins=x_bins, edge_bins=edge_bins, node_label_bins=node_label_bins, y_bins=y_bins)


plt.rcParams['font.size'] = 16
plt.rcParams['figure.dpi'] = 300

# Plotting
fig1, ax1 = plt.subplots()
ax1.bar(x_bins[0], x_hists[0], width=(x_bins[0, 1]-x_bins[0, 0]), align='edge', color='green')
ax1.set_xlabel("Active Power [GW]")
ax1.set_ylabel('Number of Nodes')
fig1.savefig(path+"ac_node_feature_distr_P_"+NAME+".png", bbox_inches='tight')

fig2, ax2 = plt.subplots()
ax2.bar(x_bins[1], x_hists[1], width=(x_bins[1, 1]-x_bins[1, 0]), align='edge', color='green')
ax2.set_xlabel("Reactive Power [MVAr]")
ax2.set_ylabel('Number of Nodes')
fig2.savefig(path+"ac_node_feature_distr_Q_"+NAME+".png", bbox_inches='tight')

fig11, ax11 = plt.subplots()
ax11.bar(x_bins[2], x_hists[2], width=x_bins[2, 1]-x_bins[2, 0], align='edge', color='green')
ax11.set_xlabel("Voltage real [p.u.]")
ax11.set_ylabel('Number of Nodes')
fig11.savefig(path+"ac_node_feature_distr_Vreal_"+NAME+".png", bbox_inches='tight')

fig12, ax12 = plt.subplots()
ax12.bar(x_bins[3], x_hists[3], width=x_bins[3, 1]-x_bins[3, 0], align='edge', color='green')
ax12.set_xlabel("Voltage imag")
ax12.set_ylabel('Number of Nodes')
fig12.savefig(path+"ac_node_feature_distr_Vimag_"+NAME+".png", bbox_inches='tight')

#Edge Features
fig3, ax3 = plt.subplots()
ax3.bar(edge_bins[0], edge_hists[0], width=(edge_bins[0, 1]-edge_bins[0, 0]), align='edge', color='orange')
ax3.set_xlabel("Y real")
ax3.set_ylabel('Number of Edges')
fig3.savefig(path+"ac_edge_feature_distr_Yreal"+NAME+".png", bbox_inches='tight')

fig4, ax4 = plt.subplots()
ax4.bar(edge_bins[1], edge_hists[1], width=(edge_bins[1, 1]-edge_bins[1, 0]), align='edge', color='orange')
ax4.set_xlabel("Y real")
ax4.set_ylabel('Number of Edges')
fig4.savefig(path+"ac_edge_feature_distr_Yimag"+NAME+".png", bbox_inches='tight')


fig5, ax5 = plt.subplots()
ax5.bar(y_bins, y_hist, width=(y_bins[1]-y_bins[0]), align='edge', color='orange')
ax5.set_xlabel("ls_tot")
ax5.set_ylabel('N Scenarios')
fig5.savefig(path+"ac_graph_label_distr"+NAME+".png", bbox_inches='tight')

for i in range(node_label_bins.shape[0]):
    fig, ax = plt.subplots()
    ax.bar(node_label_bins[i], node_label_hists[i], width=(node_label_bins[i][1] - node_label_bins[i][0]), align='edge', color='red')
    ax.set_xlabel(f"Label {i} Voltage [p.u.]")
    ax.set_ylabel('Number of Nodes')

    # Save each plot with a different name
    fig.savefig(path + f"ac_node_label_distr_{NAME}_label_{i}.png", bbox_inches='tight')
    plt.close(fig)  # Close the figure to free up memory

