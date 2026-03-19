import torch
import numpy as np
import os

from torch_geometric.data import Data

from utils.processing import get_initial_damages, get_scenario_of_file, load_mat_file, get_ls_tot


def process_ldtsf(cfg):
    '''
    Processes the data so that the input is the sequence of initial damages and the output is the total load shed of each scenario
    '''
    damages = get_initial_damages()
    KEY = 'clusterresult_'

    init_data, filetype = load_mat_file('raw/' + 'pwsdata.mat')
    edge_data = init_data[KEY][0,0][4]
    bus_from = edge_data[:,0]
    bus_to = edge_data[:,1]
    for raw_path in cfg.raw_paths:
        #skip damage file and pws file 
        if 'Hurricane' in raw_path or 'pwsdata' in raw_path:
            continue
        scenario = get_scenario_of_file(raw_path)
        file, filetype = load_mat_file(raw_path)  #loads a full scenario 
        #_, _, edge_data, _, _ = self.get_data(file[KEY][0,0][4]
        
        #get total loadshed of each step
        N_steps, ls_tot = get_ls_tot(KEY, filetype, file)
        #N_steps = len(file[KEY][0,:])
        N_damages = len(damages[scenario][:,0])
        scenario = get_scenario_of_file(raw_path)

        x = torch.zeros([len(damages[scenario][:,1]),3206])
        #for i in range(len(damages[scenario][:,1])):
            

        #add parallel lines to the damaged lines to x
        dmg_idx = 0
        shifted_step = 0
        while dmg_idx < N_damages:
            x[shifted_step,damages[scenario][dmg_idx,1]-1] = 1    #sequence of initial damages
            j=0
            while bus_from[damages[scenario][dmg_idx,1]-1] == bus_from[damages[scenario][dmg_idx,1]-1-j] and bus_to[damages[scenario][dmg_idx,1]-1] == bus_to[damages[scenario][dmg_idx,1]-1-j]:
                x[shifted_step][damages[scenario][dmg_idx,1]-1-j] = 1 
                j = j+1
            j = 0
            while bus_from[damages[scenario][dmg_idx,1]-1] == bus_from[damages[scenario][dmg_idx,1]-1+j] and bus_to[damages[scenario][dmg_idx,1]-1] == bus_to[damages[scenario][dmg_idx,1]-1+j]:
                x[shifted_step][damages[scenario][dmg_idx,1]-1+j] = 1 
                j = j+1

            #check if there is another damage in the same time step
            while dmg_idx < len(damages[scenario])-1 and damages[scenario][dmg_idx,0] == damages[scenario][dmg_idx+1,0]:  
                x[shifted_step][damages[scenario][dmg_idx+1,1]-1] = 1    
                dmg_idx = dmg_idx+1  
            dmg_idx = dmg_idx+1
            shifted_step = shifted_step+1
        x = x[:shifted_step]

        #y = torch.tensor(6.7109e4 - file[KEY][0,-1][17][99]) #17 stores the 'load' array which contains the load after each PF in ACCFM 99 in the last cell of the array containing the final load of this tsep  6.7109e4 is the full load without contingency
        remaining_load = 1
        for i in range(N_steps):
            if i == 0:  y_seq = ls_tot[i]*remaining_load    #file[KEY][0,i][21]*remaining_load
            else:       y_seq = np.append(y_seq, ls_tot[i]*remaining_load)  #np.append(y_seq, file[KEY][0,i][21]*remaining_load)
            remaining_load = remaining_load - ls_tot[i]*remaining_load  #file[KEY][0,i][21]* remaining_load
        y_seq_class = [0 if y_ < 0.15 else 1 for y_ in y_seq]
        y = torch.sum(torch.tensor(y_seq))
        
        if y/6.7109e4 < 0.18:       y_class = 0
        elif y/6.7109e4 < 0.65:     y_class = 1
        elif y/6.7109e4 < 0.88:     y_class = 2
        else:                       y_class = 3

        #y = torch.log(y+1)/torch.log(torch.tensor(6.7109e4+1))    #log normalization
        data = Data(x=x, y=y, y_class=y_class, y_seq=y_seq, y_seq_class=y_seq_class)
        torch.save(data, os.path.join(cfg.processed_dir, f'data_{scenario}_{N_steps}.pt'))
