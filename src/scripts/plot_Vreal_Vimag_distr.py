import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy

PATH_DATASET = 'raw/'
PLOT_ONLY = True

Vreal_max = np.NINF
Vreal_min = np.inf
Vimag_max = np.NINF
Vimag_min = np.inf

def get_hist(data, bins):
    hist = np.zeros(len(bins))
    for i in range(len(data)):
        for j in range(len(bins)-1):
            if data[i] <= bins[j+1]:
                hist[j]+=1
                break
    return hist

def calc_Vreal_Vimag(Vm, Va):
    Va_radians = torch.deg2rad(Va)
    Vreal = Vm * torch.cos(Va_radians)
    Vimag = Vm * torch.sin(Va_radians)
    return Vreal, Vimag

if not PLOT_ONLY:

    #Get min and max
    if PATH_DATASET == 'raw/':
        for file in os.listdir(PATH_DATASET):
            if file.startswith('pfresult'):
                file=scipy.io.loadmat('raw/'+file)
                for i in range(len(file['pf_result'][0,:])):
                    node_data_post = file['pf_result'][0,i][2]
                    Vm = torch.tensor(node_data_post[:,7]) #Voltage magnitude of all buses at initial condition - Node feature 
                    basekV = torch.tensor(node_data_post[:,9]) #Base Voltage
                    Va = torch.tensor(node_data_post[:,8]) #Voltage angle of all buses at initial condition - Node feature
                    Vreal, Vimag = calc_Vreal_Vimag(Vm*basekV, Va)
                    for i in range(len(Vreal)):
                        if Vreal[i] > Vreal_max:   Vreal_max = Vreal[i]
                        if Vreal[i] < Vreal_min:   Vreal_min = Vreal[i]
                        if Vimag[i] > Vimag_max:   Vimag_max = Vimag[i]
                        if Vimag[i] < Vimag_min:   Vimag_min = Vimag[i]
    else:
        for file in os.listdir(PATH_DATASET):
            if file.startswith('data'):
                x = torch.load(PATH_DATASET+file)['x']
                Vm = x[:,2]
                Va = x[:,3]
                Vreal, Vimag = calc_Vreal_Vimag(Vm, Va)
                for i in range(len(Vreal)):
                    if Vreal[i] > Vreal_max:   Vreal_max = Vreal[i]
                    if Vreal[i] < Vreal_min:   Vreal_min = Vreal[i]
                    if Vimag[i] > Vimag_max:   Vimag_max = Vimag[i]
                    if Vimag[i] < Vimag_min:   Vimag_min = Vimag[i]

    Vreal_bins = np.linspace(Vreal_min, Vreal_max, 10)
    Vimag_bins = np.linspace(Vimag_min, Vimag_max, 10)

    first = True
    if PATH_DATASET == 'raw/':
        for file in os.listdir(PATH_DATASET):
            if file.startswith('pfresult'):
                data=scipy.io.loadmat('raw/'+file)
                for i in range(len(data['pf_result'][0,:])):
                    node_data_post = data['pf_result'][0,i][2]
                    Vm = torch.tensor(node_data_post[:,7]) #Voltage magnitude of all buses at initial condition - Node feature 
                    basekV = torch.tensor(node_data_post[:,9]) #Base Voltage
                    Va = torch.tensor(node_data_post[:,8]) #Voltage angle of all buses at initial condition - Node feature
                    Vreal, Vimag = calc_Vreal_Vimag(Vm*basekV, Va)
                    if any(Vreal>550) or any(Vreal<-550) or any(Vimag>550) or any(Vimag<-550):
                        print(file)
                        print('Step: ', i)
                        for bus in range(len(Vreal)):
                            if Vreal[bus]>550 or Vreal[bus]<-550 or Vimag[bus]>550 or Vimag[bus]<-550:
                                print('Bus: ', bus)
                    if first:
                        Vreal_hist = get_hist(Vreal, Vreal_bins)
                        Vimag_hist = get_hist(Vimag, Vimag_bins)
                        first = False
                    else:
                        Vreal_hist += get_hist(Vreal, Vreal_bins)
                        Vimag_hist += get_hist(Vimag, Vimag_bins)
                    
    else:
        for file in os.listdir(PATH_DATASET):
            if file.startswith('data'):
                x = torch.load(PATH_DATASET+file)['x']
                Vm = x[:,2]
                Va = x[:,3]
                Va_radians = torch.deg2rad(Va)
                Vreal = Vm * torch.cos(Va_radians)
                Vimag = Vm * torch.sin(Va_radians)
                if first:
                    Vreal_hist = get_hist(Vreal, Vreal_bins)
                    Vimag_hist = get_hist(Vimag, Vimag_bins)
                    first = False
                else:
                    Vreal_hist += get_hist(Vreal, Vreal_bins)
                    Vimag_hist += get_hist(Vimag, Vimag_bins)


    np.savez('Vreal_imag_hists.npz', Vreal_hist = Vreal_hist, Vimag_hist=Vimag_hist, Vreal_bins=Vreal_bins, Vimag_bins=Vimag_bins)

else:
    data = np.load('Vreal_imag_hists.npz')
    Vreal_hist = data['Vreal_hist']
    Vimag_hist = data['Vimag_hist']
    Vreal_bins = data['Vreal_bins']
    Vimag_bins = data['Vimag_bins']

#PLOTTING
plt.rcParams['font.size'] = 16
plt.rcParams['figure.dpi'] = 300

fig1, ax1 = plt.subplots()
ax1.bar(Vreal_bins, Vreal_hist, width=(Vreal_bins[9]-Vreal_bins[8]), align='edge')
ax1.set_xlabel('Vreal')
ax1.set_ylabel('Number of Nodes')
fig1.savefig(PATH_DATASET+'Vreal.png', bbox_inches='tight')

fig2, ax2 = plt.subplots()
ax2.bar(Vimag_bins, Vimag_hist, width=(Vimag_bins[9]-Vimag_bins[8]), align='edge')
ax2.set_xlabel('Vimag')
ax2.set_ylabel('Number of Nodes')
fig2.savefig(PATH_DATASET+'Vimag.png', bbox_inches='tight')






