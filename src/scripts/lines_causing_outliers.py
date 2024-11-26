import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy

PATH_DATASET = 'raw/'


Vreal_max = np.NINF
Vreal_min = np.inf
Vimag_max = np.NINF
Vimag_min = np.inf



def calc_Vreal_Vimag(Vm, Va):
    Va_radians = torch.deg2rad(Va)
    Vreal = Vm * torch.cos(Va_radians)
    Vimag = Vm * torch.sin(Va_radians)
    return Vreal, Vimag

def get_initial_damages():
    '''
    returns the sorted initial damages of each scenario in damages [N_scenarios][step,line_id]
    where step is reassigned so that it starts with 0 and increments by 1 except if two lines 
    were destroyed in the same step
    '''
    
    #load scenario file which stores the initial damages
    f = open('raw/Hurricane_Ike_gamma8.3e-5_scenarios.txt','r')
    lines= f.readlines()
    damages = []
    for i in range(len(lines)):
        lines[i] = lines[i].replace("[", '')
        lines[i] = lines[i].replace(']', '')
        lines[i] = lines[i].replace('(', '')
        lines[i] = lines[i].replace(')', '')
        lines[i] = lines[i].replace('"', '')
        lines[i] = lines[i].replace(',', '')
        line = np.array(list(map(int, lines[i].split())))
        scenario_dmgs=np.reshape(line,(-1,2))
        scenario_dmgs=scenario_dmgs[scenario_dmgs[:,0].argsort(axis=0)]
        #rewrite the time steps to count in steps from 0 for easier handling of multiple damages in the same time step
        index = 0
        for j in range(0,len(scenario_dmgs)-1):
            increment = 0
            if scenario_dmgs[j,0] != scenario_dmgs[j+1,0]:
                increment = 1
            scenario_dmgs[j,0] = index
            index += increment
        scenario_dmgs[-1,0] = index
        damages.append(scenario_dmgs)
    return damages


def get_scenario_of_file(name):
    """
    Input:
    name        name of the processed data file
    
    Returns:
    scenario    index of the scenario of which the datafile stems
    """
    if name.startswith('./processed'):
        name=name[17:]
    else:
        name=name[10:]
    i=1
    while name[i].isnumeric():
        i+=1
    scenario=int(name[0:i])
    
    return scenario

N_files_containing_outliers = 0
damages = get_initial_damages()

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
                scenario = get_scenario_of_file(file)
                N_files_containing_outliers += 1
                print(file)
                print('Step: ', i)
                print('Line: ', damages[scenario][i,1])
                print('Lines destroyed in earlier steps:')
                for j in range(i):
                    print(damages[scenario][j,1])


                    












