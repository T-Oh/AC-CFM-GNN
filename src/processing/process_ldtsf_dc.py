import scipy.io
import json
import torch
import numpy as np
import os

from torch_geometric.data import Data

from utils.processing import get_initial_damages_dc

def process_ldtsf_dc(cfg):
    '''
    Processes the data so that the input is the sequence of initial damages and the output is the total load shed of each scenario
    '''
    init_data = scipy.io.loadmat('raw/' + 'pwsdata.mat')
    edge_data = init_data['clusterresult_'][0,0][4]
    bus_from = edge_data[:,0]
    bus_to = edge_data[:,1]
    scenario = 1 
    for raw_path in cfg.raw_paths:
        #skip damage file and pws file 
            #used to create unique file identifiers 
        if 'Hurricane' in raw_path or 'pwsdata' in raw_path:
            continue
        #scenario = self.get_scenario_of_file(raw_path)
        with open(raw_path, 'rb') as f:
            data = json.load(f)['result']
        for key in data.keys(): #every file contains 125 scenarios
            damages = get_initial_damages_dc(data[key]['primary_dmg'])

            N_steps = len(np.unique(damages[:,0]))

            x = torch.zeros([N_steps,3206])
            i = 0   #index for original damage array (damages in the same time steps appear after one another)
            updated_index = 0   #index for the resulting array where damages in the same time step appear in the same row
            while i < N_steps:
                x[updated_index,damages[i,1]-1] = 1    #sequence of initial damages
                #add parallel lines
                j=0
                while bus_from[damages[i,1]-1] == bus_from[damages[i,1]-1-j] and bus_to[damages[i,1]-1] == bus_to[damages[i,1]-1-j]:
                    x[updated_index][damages[i,1]-1-j] = 1 
                    j = j+1
                j = 0
                while bus_from[damages[i,1]-1] == bus_from[damages[i,1]-1+j] and bus_to[damages[i,1]-1] == bus_to[damages[i,1]-1+j]:
                    x[updated_index][damages[i,1]-1+j] = 1 
                    j = j+1

                #check if there is another damage in the same time step
                while i < len(damages[:,0])-1 and damages[i,0] == damages[i+1,0]:  
                    x[updated_index,damages[i+1,1]-1] = 1    
                    i = i+1
                updated_index = updated_index+1
                i = i+1

            y = torch.tensor(6.7109e4 -data[key]['final_MW_load']*100)

            
            if y/6.7109e4 < 0.18:       y_class = 0
            elif y/6.7109e4 < 0.65:     y_class = 1
            elif y/6.7109e4 < 0.88:     y_class = 2
            else:                       y_class = 3

            y = torch.log(y+1)/torch.log(torch.tensor(6.7109e4+1))    #log normalization
            processed_data = Data(x=x, y=y, y_class=y_class)
            torch.save(processed_data, os.path.join(self.processed_dir, f'data_{scenario}_{N_steps}.pt'))
            scenario += 1

