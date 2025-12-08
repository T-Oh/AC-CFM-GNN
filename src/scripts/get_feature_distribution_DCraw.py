import scipy
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

    

FOLDER = 'raw_DC/'
PLOT_ONLY = False

if not PLOT_ONLY:
    init_data = scipy.io.loadmat(FOLDER + 'pwsdata.mat')
    edge_data = init_data['ans'][0,0][4]
    bus_from = edge_data[:,0]
    bus_to = edge_data[:,1]
    scenario = 1 
    y_hist = np.zeros(100)
    yclass_hist = np.zeros(4)
    for raw_path in os.listdir(FOLDER):
        #skip damage file and pws file 
            #used to create unique file identifiers 
        if 'Hurricane' in raw_path or 'pwsdata' in raw_path:
            continue
        #scenario = self.get_scenario_of_file(raw_path)
        with open(FOLDER+raw_path, 'rb') as f:
            data = json.load(f)['result']
        for key in data.keys(): #every file contains 125 scenarios
    

            y = torch.tensor(data[key]['final_MW_load'])*100
            y_hist_temp, y_bins = np.histogram(y,bins=100, range=(0,6.7109e4))
            y_hist += y_hist_temp

            if y/6.7109e4 < 0.18:       y_class = 0
            elif y/6.7109e4 < 0.65:     y_class = 1
            elif y/6.7109e4 < 0.88:     y_class = 2
            else:                       y_class = 3
            yclass_hist_temp, yclass_bins = np.histogram(y_class, bins=4, range=(0,4))
            yclass_hist += yclass_hist_temp

        np.save('hist_DCraw.npy', {'y_hist': y_hist,
                                    'y_bins':    y_bins,
                                    'yclass_hist': yclass_hist,
                                    'yclass_bins':   yclass_bins})
else:

    hists = np.load('hist_DCraw.npy', allow_pickle=True).item()
    y_hist = hists['y_hist']
    y_bins = hists['y_bins']
    yclass_hist = hists['yclass_hist']
    yclass_bins = hists['yclass_bins']

fig1,ax1=plt.subplots()
ax1.bar(yclass_bins[0:4], yclass_hist, width=(yclass_bins[1]-yclass_bins[0]),align='center', color='blue')
ax1.set_xlabel("Total Power Outage [GW]")
ax1.set_ylabel('Number of Instances')
ax1.set_xticks([0,1,2,3])
ax1.set_xlim([-0.5, 3.5])
fig1.savefig("DC_graph_class_label_distr.png", bbox_inches='tight')

fig2,ax2=plt.subplots()
ax2.bar(y_bins[0:100], y_hist, width=(y_bins[1]-y_bins[0]),align='center', color='blue')
ax2.set_xlabel("Total Power Outage [GW]")
ax2.set_ylabel('Number of Instances')
fig2.savefig("DC_graph_label_distr.png", bbox_inches='tight')

